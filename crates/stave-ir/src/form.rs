//! The duplicate form of a function ([`form`]): the text two functions of one canonical module
//! share exactly when their code is the same. `stave dup` groups functions by it
//! (contracts/cli.md).
//!
//! The form is the function's canonical IR text ([`crate::text`]) with what names the function
//! itself left out:
//!
//! - the header keeps the param and return types, `needs`, `variadic`, and `altstack`; it drops
//!   the instance id, the path, the type args (every type the body uses is in the body), the
//!   name site, the link symbol, and the `known` and `unbuilt` marks;
//! - the body prints as the module printer prints it, without the `!row` site suffixes and with
//!   the site row a call passes to the user-site store, an error-trace hook, or the `ensures`
//!   report spelled `site` ([`crate::known::site_param`]): sites are positions; values,
//!   blocks, constants, strings, statics, and intrinsic names are already canonical module-wide
//!   ([`crate::canon`]), so equal text is equal code;
//! - a reference to the function itself is `@self`, to its closure `k` `@self.k`, and each
//!   closure's body follows the function's, in closure order;
//! - every other function is its instance (`@<instance>`), every type its id (`t<N>`): one
//!   module numbers both, so two functions that call different callees or use different types
//!   never share a form. With [`Callees::same`], a callee is spelled by the instance of the
//!   function standing for its class of equal forms: calls of two functions with the same code
//!   are the same code ([`same_code`] computes the classes).
//!
//! **Up to `Self`** ([`SelfTypes`]): for a method whose `Self` type is known, a second form
//! spells that type `Self`, each type that contains it by its key with components spelled the
//! same way (`(ref Self)`), and each call of a method of the same `Self` type `@Self.<label>`
//! with its own type args (not its impl's, which `Self` decides). Equal forms then mean equal code after renaming one `Self` type, and its
//! methods, to the other.

use std::fmt::Write as _;

use rustc_hash::FxHashMap;

use crate::ir::{FuncId, InstanceId, Module};
use crate::text::{BodyNames, body_text};
use crate::types::{GenericArg, TypeId, feature_generators, key_text_with};

/// What a function is, for the forms up to `Self`: a method whose `Self` type is `self_ty`,
/// called `label` among the methods of that type (its trait's path or `-` for an inherent
/// method, then its name), whose first `impl_args` type args are its impl's (`Self` decides
/// them).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MethodOf {
    pub self_ty: TypeId,
    pub label: String,
    pub impl_args: usize,
}

/// What a form knows about the other functions of the module, per function (an empty table
/// knows nothing).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Callees {
    /// The method each function is, for the forms up to `Self`.
    pub methods: Vec<Option<MethodOf>>,
    /// The function standing for each function's class of equal forms.
    pub same: Vec<FuncId>,
}

/// The form of one function.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Form {
    pub text: String,
    /// Instructions plus blocks (one terminator each), the function's and its closures'.
    pub size: u32,
}

/// The spelling of every type of a module that contains one `Self` type, the others being `t<N>`.
pub struct SelfTypes {
    self_ty: TypeId,
    text: FxHashMap<TypeId, String>,
}

impl SelfTypes {
    /// Walks the key components of every type of `m` once, without recursion: a type's text is
    /// made after its components'.
    pub fn new(m: &Module, self_ty: TypeId) -> SelfTypes {
        let n = m.types.len();
        // 0: not seen, 1: components pending, 2: done
        let mut state = vec![0u8; n];
        let mut text: FxHashMap<TypeId, String> = FxHashMap::default();
        let mut stack: Vec<(u32, bool)> = Vec::new();
        for root in 0..n as u32 {
            if state[root as usize] != 0 {
                continue;
            }
            stack.push((root, false));
            while let Some((t, ready)) = stack.pop() {
                let ti = t as usize;
                if state[ti] == 2 {
                    continue;
                }
                let Some(key) = m.types.key(TypeId(t)) else {
                    state[ti] = 2;
                    continue;
                };
                if !ready {
                    state[ti] = 1;
                    stack.push((t, true));
                    key.for_each_component(|c| {
                        if state.get(c.index()) == Some(&0) {
                            stack.push((c.0, false));
                        }
                    });
                    continue;
                }
                state[ti] = 2;
                if TypeId(t) == self_ty {
                    text.insert(TypeId(t), "Self".to_string());
                    continue;
                }
                let mut has = false;
                key.for_each_component(|c| has |= text.contains_key(&c));
                if has {
                    let inner = key_text_with(key, &mut |s, c| spell(&text, s, c));
                    text.insert(TypeId(t), format!("({inner})"));
                }
            }
        }
        SelfTypes { self_ty, text }
    }

    pub fn self_ty(&self) -> TypeId {
        self.self_ty
    }
}

fn spell(text: &FxHashMap<TypeId, String>, out: &mut String, t: TypeId) {
    match text.get(&t) {
        Some(s) => out.push_str(s),
        None => {
            let _ = write!(out, "t{}", t.0);
        }
    }
}

struct Names<'a> {
    m: &'a Module,
    own: FuncId,
    own_instance: &'a InstanceId,
    selfs: Option<&'a SelfTypes>,
    callees: &'a Callees,
}

impl Names<'_> {
    fn args(&self, out: &mut String, args: &[GenericArg]) {
        if args.is_empty() {
            return;
        }
        out.push('<');
        for (i, a) in args.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            match a {
                GenericArg::Type(t) => self.ty(out, *t),
                GenericArg::Const(c) => {
                    let _ = write!(out, "{c}");
                }
            }
        }
        out.push('>');
    }
}

impl BodyNames for Names<'_> {
    fn func(&self, out: &mut String, f: FuncId) {
        let Some(g) = self.m.func(f) else {
            out.push_str("@?");
            return;
        };
        if f == self.own {
            out.push_str("@self");
            return;
        }
        if let Some(rest) = g.instance.0.strip_prefix(self.own_instance.0.as_slice())
            && !rest.is_empty()
        {
            out.push_str("@self");
            for k in rest {
                let _ = write!(out, ".{k}");
            }
            return;
        }
        if let (Some(s), Some(Some(mo))) = (self.selfs, self.callees.methods.get(f.index()))
            && mo.self_ty == s.self_ty
        {
            let _ = write!(out, "@Self.{}", mo.label);
            self.args(out, g.type_args.get(mo.impl_args..).unwrap_or(&[]));
            return;
        }
        let g = self.callees.same.get(f.index()).and_then(|&r| self.m.func(r)).unwrap_or(g);
        let _ = write!(out, "@{}", g.instance);
    }

    fn ty(&self, out: &mut String, t: TypeId) {
        match self.selfs {
            Some(s) => spell(&s.text, out, t),
            None => {
                let _ = write!(out, "t{}", t.0);
            }
        }
    }
}

/// The closures of `f`: the functions after it whose instance id extends its own by one number
/// (canonical function order puts them right after it).
pub fn closures_of(m: &Module, f: FuncId) -> Vec<FuncId> {
    let Some(own) = m.func(f) else { return Vec::new() };
    let mut out = Vec::new();
    for (i, g) in m.funcs.iter().enumerate().skip(f.index() + 1) {
        let Some(rest) = g.instance.0.strip_prefix(own.instance.0.as_slice()) else { break };
        if rest.is_empty() {
            break;
        }
        if rest.len() == 1 && rest[0] != InstanceId::UNCHECKED_ENTRY {
            out.push(FuncId(i as u32));
        }
    }
    out
}

/// The form of function `f` of canonical module `m` (`None`: it has no body). `selfs`: the form
/// up to that `Self` type, with [`Callees::methods`] naming the calls of its methods.
pub fn form(m: &Module, f: FuncId, selfs: Option<&SelfTypes>, callees: &Callees) -> Option<Form> {
    let func = m.func(f)?;
    let body = func.body.as_ref()?;
    let names = Names { m, own: f, own_instance: &func.instance, selfs, callees };
    let mut text = String::from("fn(");
    for (i, p) in func.params.iter().enumerate() {
        if i > 0 {
            text.push_str(", ");
        }
        names.ty(&mut text, *p);
    }
    text.push(')');
    if let Some(r) = func.ret {
        text.push_str(" -> ");
        names.ty(&mut text, r);
    }
    let needs = feature_generators(func.features);
    if !needs.is_empty() {
        let list: Vec<&str> = needs.iter().map(|x| x.name()).collect();
        let _ = write!(text, " needs({})", list.join(", "));
    }
    for (on, word) in [(func.variadic, "variadic"), (func.alt_stack, "altstack")] {
        if on {
            text.push(' ');
            text.push_str(word);
        }
    }
    text.push('\n');
    let mut size = 0u32;
    let mut add = |text: &mut String, b: &crate::ir::Body| {
        text.push_str(&body_text(m, b, &names));
        size = size.saturating_add(b.insts.op.len() as u32).saturating_add(b.blocks.len() as u32);
    };
    add(&mut text, body);
    for c in closures_of(m, f) {
        let Some(cf) = m.func(c) else { continue };
        let k = cf.instance.0.last().copied().unwrap_or(0);
        let _ = write!(text, "closure {k}(");
        for (i, p) in cf.params.iter().enumerate() {
            if i > 0 {
                text.push_str(", ");
            }
            names.ty(&mut text, *p);
        }
        text.push(')');
        if let Some(r) = cf.ret {
            text.push_str(" -> ");
            names.ty(&mut text, r);
        }
        text.push('\n');
        if let Some(b) = &cf.body {
            add(&mut text, b);
        }
    }
    Some(Form { text, size })
}

/// The classes of equal code among `funcs` (functions of `m` with a body, ascending): the least
/// fixpoint of "equal forms, with each callee spelled by its class". Round 0 spells every callee
/// by itself; each round spells it by the first function of its class in the round before and
/// joins the functions whose forms are then equal, until a round joins none. A class only grows,
/// so this ends; two functions of a class call, at each call, functions with the same code, so
/// they have the same code. Functions that only call each other in a cycle stay apart.
/// `each` maps the forms of a round (any order, in parallel).
///
/// Returns [`Callees::same`] for all of `m`: a function outside `funcs` stands for itself.
pub fn same_code<E>(m: &Module, funcs: &[FuncId], methods: &[Option<MethodOf>], each: E) -> Vec<FuncId>
where
    E: Fn(&[FuncId], &(dyn Fn(FuncId) -> Option<String> + Sync)) -> Vec<Option<String>>,
{
    let mut cx = Callees { methods: methods.to_vec(), same: (0..m.funcs.len() as u32).map(FuncId).collect() };
    loop {
        let texts = each(funcs, &|f| form(m, f, None, &cx).map(|x| x.text));
        let mut first: FxHashMap<&str, FuncId> = FxHashMap::default();
        let mut next = cx.same.clone();
        for (&f, t) in funcs.iter().zip(&texts) {
            let Some(t) = t else { continue };
            let r = *first.entry(t.as_str()).or_insert(f);
            next[f.index()] = r;
        }
        if next == cx.same {
            return next;
        }
        cx.same = next;
    }
}
