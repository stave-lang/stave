//! Type queries of one body (Copy, drop glue, ref-likeness, field types) and the adapter to the
//! signature origins (`from` / `from copy`) of a callee.
//!
//! **Origins** ([`call_flows`], [`fn_value_flows`]): stave-sema's `origins::of_sig` /
//! `of_fn_type`: a callee without a `from` clause declares the empty origin set, so its
//! result holds no loan; `from x` (`Arg`) passes the argument value's origins, `from copy x`
//! (`Pointee`) the origins of the view it points to. A call through a fn value takes every
//! ref/view argument as `from` (the weakening of a fn item's `from copy` is covered by that).
//!
//! **Ref-likeness** ([`Tys::ref_like`]): a type whose values carry an origin set: references, fn
//! values, view ADTs, and anything containing one (`Option<&T>`, tuples, arrays), plus generic
//! params bounded `T: view`. `stave_sema::kinds::is_view_in` is the same rule.

use rustc_hash::FxHashMap;
use stave_sema::sigs::{FnSig, GenericParam, Sigs};
use stave_sema::traits::World;
use stave_sema::types::{GArg, Interner, Overlay, TypeId, TypeKind, TypeTable};
use stave_sema::{DeclId, Program};

/// How a dereference reaches its pointee.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DerefKind {
    Shared,
    Mut,
    Raw,
}

/// How a call result's origin set is formed from the argument values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Flows {
    /// The result carries no origins (its type is not ref-like).
    None,
    /// Per named parameter: (argument index, `from copy`).
    Named(Vec<(u32, bool)>),
    /// The union of every argument's origins (calls through fn values, built-ins, and callees
    /// without a usable `from` clause).
    AllArgs,
}

/// The flows of a direct call of `sig` (the result type is ref-like).
pub fn call_flows<I: Interner + ?Sized>(prog: &Program, tys: &I, sig: &FnSig) -> Flows {
    let o = stave_sema::origins::of_sig(prog, tys, sig);
    named(&o, 0)
}

/// The flows of a call through a fn value of type `fn(params) -> ret`; `shift`: the index of the
/// first argument among the call's operands.
pub fn fn_value_flows<I: Interner + ?Sized>(
    prog: &Program,
    tys: &I,
    params: &[TypeId],
    ret: TypeId,
    generics: &[GenericParam],
    shift: u32,
) -> Flows {
    let o = stave_sema::origins::of_fn_type(prog, tys, params, ret, generics);
    named(&o, shift)
}

fn named(o: &stave_sema::origins::SigOrigins, shift: u32) -> Flows {
    if o.origins.is_empty() {
        return Flows::None;
    }
    let v = o
        .origins
        .iter()
        .map(|r| (u32::from(r.param) + shift, r.flow == stave_sema::origins::Flow::Pointee))
        .collect();
    Flows::Named(v)
}

/// Per-body type queries over an overlay of the canonical table.
pub struct Tys<'a> {
    pub prog: &'a Program,
    pub sigs: &'a Sigs,
    pub ov: Overlay<'a>,
    pub env: &'a [GenericParam],
    /// Answers for local (overlay) types of this body.
    local: Memo,
    /// Answers for canonical types, shared by the non-generic bodies a thread checks in one run.
    shared: Memo,
    share: Option<u64>,
    drop_impls: &'a [DeclId],
    /// The last Copy and drop-glue answers, by the low bits of the type id: the builder asks for
    /// every operand and temporary, and a body uses a handful of types.
    hot_copy: [(TypeId, bool); HOT],
    hot_drop: [(TypeId, bool); HOT],
}

/// Slots of the per-body answer caches of [`Tys`].
const HOT: usize = 8;

/// Memoized type answers.
#[derive(Default)]
struct Memo {
    copy: FxHashMap<TypeId, bool>,
    drop: FxHashMap<TypeId, bool>,
    view: FxHashMap<TypeId, bool>,
}

thread_local! {
    static SHARED: std::cell::RefCell<(u64, Memo)> = std::cell::RefCell::new((u64::MAX, Memo::default()));
}

impl Drop for Tys<'_> {
    fn drop(&mut self) {
        if let Some(run) = self.share {
            let m = std::mem::take(&mut self.shared);
            SHARED.with(|c| *c.borrow_mut() = (run, m));
        }
    }
}

impl<'a> Tys<'a> {
    /// `run`: an id of this borrow-check run (answers for canonical types of non-generic bodies
    /// are reused across the bodies one thread checks in that run).
    pub fn new(
        prog: &'a Program,
        sigs: &'a Sigs,
        types: &'a TypeTable,
        env: &'a [GenericParam],
        run: u64,
    ) -> Tys<'a> {
        let drop_impls = sigs.impls_by_trait.get(&sigs.lang.drop).map_or(&[][..], |v| &v[..]);
        let share = env.is_empty().then_some(run);
        let shared = match share {
            Some(r) => SHARED.with(|c| {
                let mut c = c.borrow_mut();
                if c.0 == r { std::mem::take(&mut c.1) } else { Memo::default() }
            }),
            None => Memo::default(),
        };
        Tys {
            prog,
            sigs,
            ov: Overlay::new(types),
            env,
            local: Memo::default(),
            shared,
            share,
            drop_impls,
            hot_copy: [(TypeId::NONE, false); HOT],
            hot_drop: [(TypeId::NONE, false); HOT],
        }
    }

    fn memo(&mut self, t: TypeId) -> &mut Memo {
        if t.is_local() || self.share.is_none() { &mut self.local } else { &mut self.shared }
    }

    pub fn kind(&self, t: TypeId) -> TypeKind {
        self.ov.kind(t).clone()
    }

    pub fn is_unknown(&self, t: TypeId) -> bool {
        t.is_none() || self.ov.is_error_ty(t)
    }

    /// Whether values of `t` carry an origin set (unknown types: yes).
    pub fn ref_like(&mut self, t: TypeId) -> bool {
        if self.is_unknown(t) {
            return true;
        }
        if let Some(&b) = self.memo(t).view.get(&t) {
            return b;
        }
        let sigs = self.sigs;
        let env = self.env;
        let mut stack = vec![t];
        let mut kids = Vec::new();
        let mut steps = 0u32;
        let mut r = false;
        while let Some(x) = stack.pop() {
            steps += 1;
            if steps > 100_000 {
                r = true;
                break;
            }
            match self.ov.kind(x) {
                TypeKind::Ref { .. } | TypeKind::Fn { .. } => {
                    r = true;
                    break;
                }
                TypeKind::Ptr { .. } => {}
                TypeKind::Param(i) => {
                    if env.get(*i as usize).is_some_and(|g| g.view) {
                        r = true;
                        break;
                    }
                }
                TypeKind::Adt { def, args } => {
                    if sigs.adt(*def).is_some_and(|a| a.is_view) {
                        r = true;
                        break;
                    }
                    for a in args.iter() {
                        if let GArg::Type(t) = a {
                            stack.push(*t);
                        }
                    }
                }
                k => {
                    kids.clear();
                    k.children(&mut kids);
                    stack.extend_from_slice(&kids);
                }
            }
        }
        self.memo(t).view.insert(t, r);
        r
    }

    /// Whether `t` is Copy (unknown types: yes, so nothing is reported for them).
    pub fn is_copy(&mut self, t: TypeId) -> bool {
        if self.is_unknown(t) {
            return true;
        }
        let h = t.0 as usize % HOT;
        if self.hot_copy[h].0 == t {
            return self.hot_copy[h].1;
        }
        let b = match self.memo(t).copy.get(&t) {
            Some(&b) => b,
            None => {
                let w = World { prog: self.prog, sigs: self.sigs };
                let b = w.is_copy(&mut self.ov, t, self.env);
                self.memo(t).copy.insert(t, b);
                b
            }
        };
        self.hot_copy[h] = (t, b);
        b
    }

    /// Whether dropping a value of `t` runs code (a `Drop` impl inside, not behind a pointer); a
    /// generic param that is not Copy may.
    pub fn needs_drop(&mut self, t: TypeId) -> bool {
        if self.is_unknown(t) {
            return false;
        }
        let h = t.0 as usize % HOT;
        if self.hot_drop[h].0 == t {
            return self.hot_drop[h].1;
        }
        let r = self.needs_drop_d(t, 0);
        self.hot_drop[h] = (t, r);
        r
    }

    fn needs_drop_d(&mut self, t: TypeId, depth: u32) -> bool {
        if self.is_unknown(t) || depth > 64 {
            return false;
        }
        if let Some(&b) = self.memo(t).drop.get(&t) {
            return b;
        }
        self.memo(t).drop.insert(t, false);
        let r = match self.kind(t) {
            TypeKind::Adt { def, args } => {
                let has_impl = self.drop_impls.iter().any(|&i| {
                    self.sigs.impl_def(i).is_some_and(|d| d.head == Some(stave_sema::TypeHead::Adt(def)))
                });
                has_impl || {
                    let fields: Vec<TypeId> = self
                        .sigs
                        .adt(def)
                        .map(|a| a.variants.iter().flat_map(|v| v.fields.iter().map(|f| f.ty)).collect())
                        .unwrap_or_default();
                    fields.into_iter().any(|f| {
                        let ft = self.ov.subst(f, &args, None);
                        self.needs_drop_d(ft, depth + 1)
                    })
                }
            }
            TypeKind::Tuple(ts) => ts.iter().any(|&x| self.needs_drop_d(x, depth + 1)),
            TypeKind::Array { elem, .. } => self.needs_drop_d(elem, depth + 1),
            TypeKind::Param(_) => !self.is_copy(t),
            _ => false,
        };
        self.memo(t).drop.insert(t, r);
        r
    }

    /// The pointee of a reference or raw pointer.
    pub fn deref(&self, t: TypeId) -> Option<(TypeId, DerefKind)> {
        match self.ov.kind(t) {
            TypeKind::Ref { mutable, inner } => {
                Some((*inner, if *mutable { DerefKind::Mut } else { DerefKind::Shared }))
            }
            TypeKind::Ptr { inner, .. } => Some((*inner, DerefKind::Raw)),
            _ => None,
        }
    }

    /// The element type of an array or slice (`NONE` otherwise).
    pub fn elem(&self, t: TypeId) -> TypeId {
        match self.ov.kind(t) {
            TypeKind::Array { elem, .. } | TypeKind::Slice { elem, .. } => *elem,
            _ => TypeId::NONE,
        }
    }

    /// Field `index` of variant `variant` of `t` (struct: variant 0; tuple element; range end).
    pub fn field_ty(&mut self, t: TypeId, variant: u32, index: u32) -> TypeId {
        match self.kind(t) {
            TypeKind::Tuple(ts) => ts.get(index as usize).copied().unwrap_or(TypeId::NONE),
            TypeKind::Range { elem, .. } => elem,
            TypeKind::Adt { def, args } => {
                let f = self
                    .sigs
                    .adt(def)
                    .and_then(|a| a.variants.get(variant as usize))
                    .and_then(|v| v.fields.get(index as usize))
                    .map(|f| f.ty);
                match f {
                    Some(f) => self.ov.subst(f, &args, None),
                    None => TypeId::NONE,
                }
            }
            _ => TypeId::NONE,
        }
    }

    /// The index of the field named `name` (text) of struct type `t`, with its type.
    pub fn field_by_name(&mut self, t: TypeId, name: &str) -> Option<(u32, TypeId)> {
        let TypeKind::Adt { def, .. } = self.kind(t) else { return None };
        let i = self
            .sigs
            .adt(def)?
            .variants
            .first()?
            .fields
            .iter()
            .position(|f| self.prog.strings.get(f.name) == name)?;
        Some((i as u32, self.field_ty(t, 0, i as u32)))
    }

    /// The name of field `index` of variant `variant` of `t` (`None`: a tuple element).
    pub fn field_name(&self, t: TypeId, variant: u32, index: u32) -> Option<String> {
        let TypeKind::Adt { def, .. } = self.ov.kind(t) else { return None };
        let f = self.sigs.adt(*def)?.variants.get(variant as usize)?.fields.get(index as usize)?;
        if f.name.is_none() {
            return None;
        }
        Some(self.prog.strings.get(f.name).to_string())
    }

    pub fn subst(&mut self, t: TypeId, args: &[GArg], self_ty: Option<TypeId>) -> TypeId {
        if t.is_none() {
            return t;
        }
        self.ov.subst(t, args, self_ty)
    }

    /// The payload type of `Option<T>` / `Result<T, E>` (`NONE` otherwise).
    pub fn payload(&self, t: TypeId) -> TypeId {
        match self.ov.kind(t) {
            TypeKind::Adt { def, args } if *def == self.sigs.lang.option || *def == self.sigs.lang.result => {
                match args.first() {
                    Some(GArg::Type(x)) => *x,
                    _ => TypeId::NONE,
                }
            }
            _ => TypeId::NONE,
        }
    }

    pub fn is_never(&self, t: TypeId) -> bool {
        matches!(self.ov.kind(t), TypeKind::Never)
    }
}
