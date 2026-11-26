//! IR verifier: structure (spans, value definitions), SSA dominance, block
//! param arity and types, terminators, per-op operand / result typing, and pointer mutability.
//! Returns every error found; never panics, whatever the input.
//!
//! It is the gate for modules the compiler did not build: a module that verifies must not panic
//! a back end, the release passes, or the interpreter. So every id is in range and of its kind
//! (sites included), every intrinsic is an entry of [`crate::intrinsics`] used with its
//! signature, every known item the stages call on their own has the signature they assume, and
//! every type has a shape the stages can size. A form the verifier does not know is an error.

use std::fmt;

mod mutability;
mod simd;

use crate::arith::{self, CastTy, IntTy};
use crate::intrinsics::{self, ITy, Kind};
use crate::ir::{
    Block, Body, CastKind, Const, Func, FuncId, Module, Op, Operand, Site, Target, Term, Value, ValueDef,
};
use crate::known;
use crate::text::Level;
use crate::types::{AdtKind, FloatBits, GenericArg, IntBits, TypeId, TypeKey, TypeTable};
use crate::witness::Term as RowTerm;

/// One verifier finding, located as precisely as known.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifyError {
    pub func: Option<FuncId>,
    pub block: Option<Block>,
    /// Instruction index (body-wide); `None` for a terminator or a block / function error.
    pub inst: Option<u32>,
    pub msg: String,
}

impl fmt::Display for VerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(func) = self.func {
            write!(f, "fn #{}", func.0)?;
        } else {
            f.write_str("module")?;
        }
        if let Some(b) = self.block {
            write!(f, " b{}", b.0)?;
        }
        if let Some(i) = self.inst {
            write!(f, " inst {i}")?;
        }
        write!(f, ": {}", self.msg)
    }
}

/// Verifies the whole module by the rules of levels `lowered` and `checked`.
pub fn verify_module(m: &Module) -> Result<(), Vec<VerifyError>> {
    verify_module_at(m, Level::Checked)
}

/// Verifies the whole module by the rules of `level` (contracts/checked-ir.md "Loading" step 3:
/// only level `opt` differs, in the `switch` scrutinee).
pub fn verify_module_at(m: &Module, level: Level) -> Result<(), Vec<VerifyError>> {
    let mut errs = verify_types(m)?;
    for fi in 0..m.funcs.len() {
        errs.extend(verify_func_body(m, FuncId(fi as u32), level));
    }
    if errs.is_empty() { Ok(()) } else { Err(errs) }
}

/// The module-level part of [`verify_module`]: the type table and the ADT shapes. `Ok`: the shape
/// errors (the functions can still be verified); `Err`: the type table is malformed and the
/// functions cannot be verified.
pub fn verify_types(m: &Module) -> Result<Vec<VerifyError>, Vec<VerifyError>> {
    let mut errs = Vec::new();
    let module_err = |msg: String| VerifyError { func: None, block: None, inst: None, msg };
    let types_ok = match m.types.check_well_formed() {
        Ok(()) => true,
        Err(e) => {
            errs.push(module_err(format!("type table: {e:?}")));
            false
        }
    };
    let nt = m.types.len();
    for i in 0..nt {
        if let Some(s) = m.types.shape(TypeId(i as u32)) {
            if s.kind == AdtKind::Struct && s.variants.len() != 1 {
                errs.push(module_err(format!("t{i}: struct shape needs exactly one variant")));
            }
            for f in s.variants.iter().flatten() {
                if f.index() >= nt {
                    errs.push(module_err(format!("t{i}: shape field type t{} out of range", f.0)));
                }
            }
        }
    }
    errs.extend(module_faults(m).into_iter().map(module_err));
    errs.extend(verify_static_refs(m).into_iter().map(module_err));
    if types_ok {
        errs.extend(verify_lanes(&m.types).into_iter().map(module_err));
        let layouts = crate::layout::Layouts::compute(&m.types);
        errs.extend(verify_layouts(&m.types, &layouts).into_iter().map(module_err));
        errs.extend(verify_static_layout(m, &layouts).into_iter().map(module_err));
    }
    if types_ok { Ok(errs) } else { Err(errs) }
}

/// The keys the text can spell and the decls they name: no generic parameter (the module is
/// monomorphized), every `adt` key's decl and `cell` in `decls`, `decls` ascending by id with
/// none that no key or `cell` names, and an ADT shape of its decl's kind; `main` names a fn. The
/// printer refuses a module with any of these too: its text would not load.
pub(crate) fn module_faults(m: &Module) -> Vec<String> {
    let mut errs = Vec::new();
    if let Some(f) = m.main.filter(|f| f.index() >= m.funcs.len()) {
        errs.push(format!("main: fn #{} out of range", f.0));
    }
    if !m.decls.windows(2).all(|w| w[0].id < w[1].id) {
        errs.push("decls: not ascending by id, or an id twice".to_string());
    }
    let decl = |d: crate::types::DeclId| m.decls.binary_search_by_key(&d, |x| x.id).ok().map(|k| &m.decls[k]);
    let mut named = vec![false; m.decls.len()];
    let mut name = |d: crate::types::DeclId| {
        if let Ok(k) = m.decls.binary_search_by_key(&d, |x| x.id) {
            named[k] = true;
        }
    };
    for (i, k) in m.types.keys().iter().enumerate() {
        match k {
            TypeKey::Param(_) | TypeKey::ConstParam(_) => {
                errs.push(format!("t{i}: a generic parameter in a monomorphized module"));
            }
            TypeKey::Adt { def, .. } => {
                name(*def);
                match (decl(*def), m.types.shape(TypeId(i as u32))) {
                    (None, _) => errs.push(format!("t{i}: adt d{} has no decl", def.0)),
                    (Some(d), Some(s)) if s.kind != d.kind => {
                        errs.push(format!("t{i}: a {:?} shape of {:?} decl d{}", s.kind, d.kind, def.0));
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
    if let Some(c) = m.cell {
        name(c);
        if decl(c).is_none() {
            errs.push(format!("cell: d{} has no decl", c.0));
        }
    }
    for (k, d) in m.decls.iter().enumerate() {
        if !named[k] {
            errs.push(format!("decl d{}: no type or `cell` names it", d.id.0));
        }
    }
    errs
}

/// A vector type has a lane type ([`crate::simd::lane_ty`]) and a lane count of
/// [`crate::simd::valid_lanes`] or the native one; a mask type a lane width of 8, 16, 32, or 64
/// bits and such a count.
fn verify_lanes(types: &TypeTable) -> Vec<String> {
    use crate::types::Lanes;
    let count_ok = |n: &Lanes| match n {
        Lanes::Fixed(n) => crate::simd::valid_lanes(u32::from(*n)),
        Lanes::Native => true,
    };
    let mut errs = Vec::new();
    for i in 0..types.len() {
        let ok = match types.key(TypeId(i as u32)) {
            Some(TypeKey::Vec { lane, n }) => crate::simd::lane_ty(types, *lane).is_some() && count_ok(n),
            Some(TypeKey::Mask { width, n }) => matches!(width, 8 | 16 | 32 | 64) && count_ok(n),
            _ => true,
        };
        if !ok {
            errs.push(format!("t{i}: not a vector or mask shape ({})", types.key_text(TypeId(i as u32))));
        }
    }
    errs
}

/// No type contains itself by value, nests deeper than the layout bound, or has a size beyond
/// `u64`: every stage that sizes values computes these layouts. A type may still have no layout
/// for being unsized, generic, or shapeless; only values need one, and the stages report that.
fn verify_layouts(types: &TypeTable, layouts: &crate::layout::Layouts) -> Vec<String> {
    use crate::layout::LayoutError as E;
    let mut errs = Vec::new();
    for i in 0..types.len() {
        if let Err(e @ (E::Infinite | E::TooDeep | E::Overflow)) = layouts.get_layout(TypeId(i as u32)) {
            errs.push(format!("t{i}: {e}"));
        }
    }
    errs
}

/// Every static's `init` holds exactly its type's size, and a static with address words has a
/// type aligned to a multiple of 8: each word is 8-aligned within the static, and the back ends
/// place a static at its type's alignment, so only then is the word 8-aligned in memory.
fn verify_static_layout(m: &Module, layouts: &crate::layout::Layouts) -> Vec<String> {
    let mut errs = Vec::new();
    for (i, st) in m.statics.iter().enumerate() {
        let l = match layouts.get_layout(st.ty) {
            Ok(l) => l,
            Err(e) => {
                errs.push(format!("static {i}: type t{} has no layout ({e})", st.ty.0));
                continue;
            }
        };
        if st.init.len() as u64 != l.size {
            errs.push(format!("static {i}: {} initial bytes for a type of size {}", st.init.len(), l.size));
        }
        if !st.refs.is_empty() && !l.align.is_multiple_of(8) {
            errs.push(format!("static {i}: address words in a type aligned to {}", l.align));
        }
    }
    errs
}

/// The address words of every static: ascending, non-overlapping, 8-aligned words wholly inside
/// zero bytes of `init`, each into an existing string or static, at most one past its end.
fn verify_static_refs(m: &Module) -> Vec<String> {
    let mut errs = Vec::new();
    for (i, st) in m.statics.iter().enumerate() {
        let mut next = 0usize;
        for r in &st.refs {
            let at = r.offset as usize;
            let word = st.init.get(at..at.saturating_add(8));
            if at < next || !at.is_multiple_of(8) || word.is_none_or(|w| w.iter().any(|&x| x != 0)) {
                errs.push(format!("static {i}: address word at {at} is misplaced"));
            }
            next = at.saturating_add(8);
            let len = match r.target {
                crate::DataTarget::Str(s) => m.strings.get(s.index()).map(|b| b.len()),
                crate::DataTarget::Static(k) => m.statics.get(k as usize).map(|t| t.init.len()),
            };
            match len {
                Some(n) if r.addend as usize <= n => {}
                Some(_) => errs.push(format!("static {i}: address word at {at} points past its target")),
                None => errs.push(format!("static {i}: address word at {at} has no target")),
            }
        }
    }
    errs
}

/// The per-function part of [`verify_module_at`]: the errors of function `fid` (none for an
/// unknown id) by the rules of `level`. The type table must be well formed ([`verify_types`]).
/// Functions are independent, so a caller may verify them in parallel; [`verify_module_at`]'s
/// errors are [`verify_types`]'s, then every function's in id order.
pub fn verify_func_body(m: &Module, fid: FuncId, level: Level) -> Vec<VerifyError> {
    let mut errs = Vec::new();
    if let Some(f) = m.func(fid) {
        FnVerifier { m, f, fid, level, errs: &mut errs, body: None }.run();
    }
    errs
}

/// The part of verification that needs the target: every `intrinsic.<name>` is built for `arch`
/// ([`intrinsics::IArch::builds_for`], the rule the front end applies to a declaration). A name
/// outside the table is [`verify_func_body`]'s error, not this one's.
pub fn verify_target(m: &Module, arch: stave_target::Arch) -> Vec<VerifyError> {
    let mut errs = Vec::new();
    for (k, f) in m.funcs.iter().enumerate() {
        let Some(b) = &f.body else { continue };
        for (i, op) in b.insts.op.iter().enumerate() {
            if let Op::Intrinsic(id) = op
                && let Some(kind) = m.intrinsic_kind(*id)
                && !kind.arch().builds_for(arch)
            {
                errs.push(VerifyError {
                    func: Some(FuncId(k as u32)),
                    block: None,
                    inst: Some(i as u32),
                    msg: format!(
                        "`intrinsic.{}` is not built for the target arch {}",
                        m.intrinsic_name(*id).unwrap_or("?"),
                        arch.name()
                    ),
                });
            }
        }
    }
    errs
}

/// Verifies one function of the module by the rules of levels `lowered` and `checked`.
pub fn verify_func(m: &Module, fid: FuncId) -> Result<(), Vec<VerifyError>> {
    let mut errs = Vec::new();
    if let Err(e) = m.types.check_well_formed() {
        errs.push(VerifyError { func: None, block: None, inst: None, msg: format!("type table: {e:?}") });
        return Err(errs);
    }
    match m.func(fid) {
        None => errs.push(VerifyError {
            func: Some(fid),
            block: None,
            inst: None,
            msg: "unknown function".into(),
        }),
        Some(f) => FnVerifier { m, f, fid, level: Level::Checked, errs: &mut errs, body: None }.run(),
    }
    if errs.is_empty() { Ok(()) } else { Err(errs) }
}

/// Classification of a type for op typing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Cls {
    Int(IntTy),
    Bool,
    Char,
    Float(FloatBits),
    /// Thin ref / raw pointer.
    Ptr,
    /// Fat ref / raw pointer (to a slice or `str`).
    Fat,
    Fn,
    Agg,
    Other,
}

pub(crate) fn classify(types: &TypeTable, t: TypeId) -> Cls {
    match types.key(t) {
        Some(TypeKey::Int { signed, bits }) => Cls::Int(IntTy::new(
            match bits {
                IntBits::Size => 64,
                b => b.width(),
            },
            *signed,
        )),
        Some(TypeKey::Bool) => Cls::Bool,
        Some(TypeKey::Char) => Cls::Char,
        Some(TypeKey::Float(f)) => Cls::Float(*f),
        Some(TypeKey::Ref { inner, .. } | TypeKey::RawPtr { inner, .. }) => {
            if crate::layout::is_unsized_pointee(types, *inner) {
                Cls::Fat
            } else {
                Cls::Ptr
            }
        }
        Some(TypeKey::FnType { .. }) => Cls::Fn,
        Some(TypeKey::Tuple(_) | TypeKey::Array { .. } | TypeKey::Adt { .. }) => Cls::Agg,
        _ => Cls::Other,
    }
}

pub(crate) fn cast_ty(c: Cls) -> Option<CastTy> {
    Some(match c {
        Cls::Int(t) => CastTy::Int(t),
        Cls::Bool => CastTy::Bool,
        Cls::Char => CastTy::Char,
        Cls::Float(f) => CastTy::Float(f),
        // A fn item operand casts as a pointer (its code address); callers map `@f` themselves,
        // so a fn *value* (code + environment) and a fn-type target have no cast type.
        Cls::Ptr => CastTy::Ptr,
        _ => return None,
    })
}

/// Pointee of a ref / raw pointer type.
pub(crate) fn pointee(types: &TypeTable, t: TypeId) -> Option<TypeId> {
    match types.key(t) {
        Some(TypeKey::Ref { inner, .. } | TypeKey::RawPtr { inner, .. }) => Some(*inner),
        _ => None,
    }
}

/// Fields of a tuple / struct type (not enums, not arrays).
pub(crate) fn record_fields(types: &TypeTable, t: TypeId) -> Option<Vec<TypeId>> {
    match types.key(t) {
        Some(TypeKey::Tuple(ts)) => Some(ts.clone()),
        Some(TypeKey::Adt { .. }) => match types.shape(t) {
            Some(s) if s.kind == AdtKind::Struct => s.variants.first().cloned(),
            _ => None,
        },
        _ => None,
    }
}

pub(crate) fn enum_variants(types: &TypeTable, t: TypeId) -> Option<&Vec<Vec<TypeId>>> {
    match types.shape(t) {
        Some(s) if s.kind == AdtKind::Enum => Some(&s.variants),
        _ => None,
    }
}

pub(crate) fn option_shape(types: &TypeTable, t: TypeId) -> Option<(u32, u32, TypeId)> {
    types.option_shape(t)
}

struct FnVerifier<'a> {
    m: &'a Module,
    f: &'a Func,
    fid: FuncId,
    level: Level,
    errs: &'a mut Vec<VerifyError>,
    body: Option<&'a Body>,
}

impl<'a> FnVerifier<'a> {
    fn err(&mut self, block: Option<Block>, inst: Option<u32>, msg: impl Into<String>) {
        self.errs.push(VerifyError { func: Some(self.fid), block, inst, msg: msg.into() });
    }

    fn ty_ok(&self, t: TypeId) -> bool {
        t.index() < self.m.types.len()
    }

    fn key(&self, t: TypeId) -> Option<&'a TypeKey> {
        self.m.types.key(t)
    }

    fn cls(&self, t: TypeId) -> Cls {
        classify(&self.m.types, t)
    }

    fn tname(&self, t: TypeId) -> String {
        format!("t{} ({})", t.0, self.m.types.key_text(t))
    }

    fn run(&mut self) {
        let f = self.f;
        let targs = f.type_args.iter().filter_map(|a| match a {
            GenericArg::Type(t) => Some(t),
            GenericArg::Const(_) => None,
        });
        for &t in f.params.iter().chain(targs).chain(f.ret.iter()) {
            if !self.ty_ok(t) {
                self.err(None, None, format!("signature type t{} out of range", t.0));
                return;
            }
        }
        if !self.site_ok(f.name_site) {
            self.err(None, None, format!("name site {} out of range", f.name_site.0));
        }
        self.known_signature();
        self.claim_params();
        let Some(body) = &f.body else { return };
        self.body = Some(body);
        if !self.structure(body) {
            return;
        }
        let dom = Dominators::compute(body);
        self.dominance(body, &dom);
        self.typing(body);
    }

    /// `Site::NONE` or a row of the site table.
    fn site_ok(&self, s: Site) -> bool {
        s == Site::NONE || s.index() < self.m.sites.len()
    }

    /// Every `p i` and `lenp i` of the fn's summary rows (its `claim` lines) names one of its
    /// params: the audit reads a row at a call by the call's argument `i`.
    fn claim_params(&mut self) {
        let Some(s) = self.m.stated.get(self.fid.0 as usize).and_then(|s| s.as_deref()) else { return };
        let n = self.f.params.len();
        let rows = s.pre.iter().map(|p| &p.row).chain(&s.post);
        let bad = rows.flat_map(|r| [r.a, r.b]).find_map(|t| match t {
            RowTerm::Param(i) | RowTerm::LenParam(i) if i as usize >= n => Some(i),
            _ => None,
        });
        if let Some(i) = bad {
            self.err(None, None, format!("claim names param {i} of a fn with {n} params"));
        }
    }

    /// A known item's signature is the one the stages that call it themselves assume
    /// ([`known::abi_signature`]), and only a panic report entry runs on the alternate stack. A
    /// report entry that does not is no error: its code runs on the stack it was called on.
    fn known_signature(&mut self) {
        let f = self.f;
        if f.alt_stack && !f.known.is_some_and(known::reports_on_alt_stack) {
            let k = f.known.map_or("no known item".to_string(), |k| format!("`{k}`"));
            self.err(None, None, format!("alt_stack is {} for {k}", f.alt_stack));
        }
        let Some(k) = f.known else { return };
        let Some((params, ret)) = known::abi_signature(k) else { return };
        let ty = |x: ITy| ity_type(&self.m.types, x);
        let ps_ok =
            params.len() == f.params.len() && params.iter().zip(&f.params).all(|(&p, &t)| ty(p) == Some(t));
        let ret_ok = match ret {
            ITy::Unit => f.ret.is_none(),
            r => f.ret.is_some() && ty(r) == f.ret,
        };
        if !ps_ok || !ret_ok || f.variadic {
            self.err(None, None, format!("`{k}` must have the signature {params:?} -> {ret:?}"));
        }
    }

    /// Spans, ids, and definitions. Returns false if later passes cannot run safely.
    fn structure(&mut self, b: &Body) -> bool {
        let n0 = self.errs.len();
        if b.blocks.is_empty() {
            self.err(None, None, "body has no blocks");
            return false;
        }
        let ni = b.insts.op.len();
        if b.insts.ty.len() != ni
            || b.insts.result.len() != ni
            || b.insts.args.len() != ni
            || b.insts.site.len() != ni
        {
            self.err(None, None, "instruction arrays differ in length");
            return false;
        }
        let nv = b.values.len();
        for (i, v) in b.values.iter().enumerate() {
            if !self.ty_ok(v.ty) {
                self.err(None, None, format!("%{i}: type t{} out of range", v.ty.0));
            }
        }
        // Block spans: params in range, insts consecutive and covering all instructions.
        let mut next = 0u32;
        // Each operand and switch case belongs to one span at most: the passes that rewrite
        // operands in place rewrite each span on its own.
        let mut operand_owned = vec![false; b.operands.len()];
        let mut case_owned = vec![false; b.cases.len()];
        let claim = |owned: &mut [bool], s: crate::ir::Span| {
            let mut ok = true;
            for k in s.indices() {
                if let Some(o) = owned.get_mut(k as usize) {
                    ok &= !*o;
                    *o = true;
                }
            }
            ok
        };
        let mut defined = vec![false; nv];
        let mut define =
            |s: &mut Self, v: Value, blk: Block, inst: Option<u32>| match defined.get_mut(v.index()) {
                None => s.err(Some(blk), inst, format!("%{} out of range", v.0)),
                Some(d) if *d => s.err(Some(blk), inst, format!("%{} defined twice", v.0)),
                Some(d) => *d = true,
            };
        for (bi, bd) in b.blocks.iter().enumerate() {
            let blk = Block(bi as u32);
            if !bd.params.in_range(b.block_params.len()) {
                self.err(Some(blk), None, "param span out of range");
            } else {
                for (pi, &v) in bd.params.of(&b.block_params).iter().enumerate() {
                    define(self, v, blk, None);
                    match b.values.get(v.index()) {
                        Some(d) if d.def == (ValueDef::Param { block: blk, index: pi as u32 }) => {}
                        Some(_) => self.err(Some(blk), None, format!("%{}: definition record mismatch", v.0)),
                        None => {}
                    }
                }
            }
            if bd.insts.start != next || !bd.insts.in_range(ni) {
                self.err(Some(blk), None, "instruction span not consecutive / out of range");
                return false;
            }
            next = bd.insts.start + bd.insts.len;
            for i in bd.insts.indices() {
                let a = b.insts.args[i as usize];
                if !a.in_range(b.operands.len()) {
                    self.err(Some(blk), Some(i), "operand span out of range");
                } else if !claim(&mut operand_owned, a) {
                    self.err(Some(blk), Some(i), "operand span shared with another instruction or target");
                }
                let s = b.insts.site[i as usize];
                if !self.site_ok(s) {
                    self.err(Some(blk), Some(i), format!("site {} out of range", s.0));
                }
                match (b.insts.result[i as usize], b.insts.ty[i as usize]) {
                    (Some(v), Some(t)) => {
                        define(self, v, blk, Some(i));
                        match b.values.get(v.index()) {
                            Some(d) if d.def == ValueDef::Inst(i) && d.ty == t => {}
                            Some(_) => {
                                self.err(Some(blk), Some(i), format!("%{}: definition record mismatch", v.0))
                            }
                            None => {}
                        }
                    }
                    (None, None) => {}
                    _ => {
                        self.err(Some(blk), Some(i), "result and result type must both be present or absent")
                    }
                }
                if let Some(t) = b.insts.ty[i as usize]
                    && !self.ty_ok(t)
                {
                    self.err(Some(blk), Some(i), format!("result type t{} out of range", t.0));
                }
            }
            let targets: Vec<Target> = b.successors(blk);
            if let Term::Switch { cases, .. } = bd.term {
                if !cases.in_range(b.cases.len()) {
                    self.err(Some(blk), None, "switch case span out of range");
                } else if !claim(&mut case_owned, cases) {
                    self.err(Some(blk), None, "switch case span shared with another switch");
                }
            }
            for t in targets {
                if !t.args.in_range(b.operands.len()) {
                    self.err(Some(blk), None, "target arg span out of range");
                } else if !claim(&mut operand_owned, t.args) {
                    self.err(Some(blk), None, "target arg span shared with another instruction or target");
                }
                if t.block.index() >= b.blocks.len() {
                    self.err(Some(blk), None, format!("target b{} out of range", t.block.0));
                } else if t.block.0 == 0 {
                    self.err(Some(blk), None, "b0 (entry) cannot be a branch target");
                }
            }
        }
        if next as usize != ni {
            self.err(None, None, "instructions not covered by any block");
        }
        for (i, d) in defined.iter().enumerate() {
            if !d {
                self.err(None, None, format!("%{i} has no definition"));
            }
        }
        // Operand ids in range.
        for (oi, o) in b.operands.iter().enumerate() {
            let bad = match o {
                Operand::Value(v) => v.index() >= nv,
                Operand::Const(c) => c.index() >= b.consts.len(),
                Operand::Str(s) => s.index() >= self.m.strings.len(),
                Operand::Func(fi) => fi.index() >= self.m.funcs.len(),
                Operand::Type(t) => !self.ty_ok(*t),
            };
            if bad {
                self.err(None, None, format!("operand #{oi} ({o:?}) out of range"));
            }
        }
        let mut terms = Vec::new();
        for (bi, bd) in b.blocks.iter().enumerate() {
            match bd.term {
                Term::Cbr { cond, .. } => terms.push((bi, cond)),
                Term::Switch { scrut, .. } => terms.push((bi, scrut)),
                Term::Ret(Some(o)) => terms.push((bi, o)),
                _ => {}
            }
        }
        for (bi, o) in terms {
            let bad = match o {
                Operand::Value(v) => v.index() >= nv,
                Operand::Const(c) => c.index() >= b.consts.len(),
                Operand::Str(s) => s.index() >= self.m.strings.len(),
                Operand::Func(fi) => fi.index() >= self.m.funcs.len(),
                Operand::Type(t) => !self.ty_ok(t),
            };
            if bad {
                self.err(Some(Block(bi as u32)), None, format!("terminator operand {o:?} out of range"));
            }
        }
        // Entry params = fn params.
        let p0: Vec<TypeId> = b.params(Block(0)).iter().filter_map(|v| b.value_ty(*v)).collect();
        if p0 != self.f.params {
            self.err(Some(Block(0)), None, "b0 params do not match the fn params");
        }
        self.errs.len() == n0
    }

    fn dominance(&mut self, b: &Body, dom: &Dominators) {
        // Position of each definition: (block, index) with params at 0 and inst k at k+1.
        let mut pos = vec![(0u32, 0u32); b.values.len()];
        for (bi, bd) in b.blocks.iter().enumerate() {
            for &v in bd.params.of(&b.block_params) {
                pos[v.index()] = (bi as u32, 0);
            }
            for i in bd.insts.indices() {
                if let Some(v) = b.insts.result[i as usize] {
                    pos[v.index()] = (bi as u32, i - bd.insts.start + 1);
                }
            }
        }
        for (bi, bd) in b.blocks.iter().enumerate() {
            if !dom.reachable(bi) {
                continue;
            }
            let blk = Block(bi as u32);
            let check = |s: &mut Self, o: &Operand, at: u32, inst: Option<u32>| {
                if let Operand::Value(v) = o {
                    let (db, dp) = pos[v.index()];
                    let ok = if db as usize == bi { dp < at } else { dom.dominates(db as usize, bi) };
                    if !ok {
                        s.err(Some(blk), inst, format!("use of %{} not dominated by its definition", v.0));
                    }
                }
            };
            for i in bd.insts.indices() {
                for o in b.args(i) {
                    check(self, o, i - bd.insts.start + 1, Some(i));
                }
            }
            let end = bd.insts.len + 1;
            match bd.term {
                Term::Cbr { cond, .. } => check(self, &cond, end, None),
                Term::Switch { scrut, .. } => check(self, &scrut, end, None),
                Term::Ret(Some(o)) => check(self, &o, end, None),
                _ => {}
            }
            for t in b.successors(blk) {
                for o in b.target_args(&t) {
                    check(self, o, end, None);
                }
            }
        }
    }

    // ---- typing ----

    fn operand_ty(&self, b: &Body, o: Operand) -> Option<TypeId> {
        match o {
            Operand::Value(v) => b.value_ty(v),
            _ => None,
        }
    }

    /// Checks that `o` can stand where the op rules name a scalar type by its key (`bool` for a
    /// condition, `int u size` for a length or index). The table need not hold the type: a
    /// module whose only operand there is a constant has no value of it, and a canonical type
    /// table holds only the types something reaches.
    pub(crate) fn check_scalar_operand(&self, b: &Body, o: Operand, want: &TypeKey) -> Result<(), String> {
        if let Some(t) = self.m.types.find(want) {
            return self.check_operand(b, o, t);
        }
        let fits = match (o, want) {
            (Operand::Const(c), TypeKey::Bool) => matches!(b.konst(c), Some(Const::Int(0 | 1))),
            (Operand::Const(c), TypeKey::Int { signed, bits }) => {
                let width = if *bits == IntBits::Size { 64 } else { bits.width() };
                matches!(b.konst(c), Some(Const::Int(x)) if IntTy::new(width, *signed).fits(x))
            }
            _ => false,
        };
        if fits { Ok(()) } else { Err(format!("operand cannot have type {}", crate::types::key_text(want))) }
    }

    /// Checks that `o` can stand where a value of type `want` is expected.
    fn check_operand(&self, b: &Body, o: Operand, want: TypeId) -> Result<(), String> {
        let key = self.key(want).ok_or("expected type out of range")?;
        match o {
            Operand::Value(v) => {
                let t = b.value_ty(v).ok_or("value out of range")?;
                if t == want || self.same_fn_shape(t, key) {
                    Ok(())
                } else {
                    Err(format!("%{} has type {}, expected {}", v.0, self.tname(t), self.tname(want)))
                }
            }
            Operand::Const(c) => match (b.konst(c), key) {
                (Some(Const::Int(x)), TypeKey::Int { .. }) => match self.cls(want) {
                    Cls::Int(it) if it.fits(x) => Ok(()),
                    _ => Err(format!("constant {x} out of range of {}", self.tname(want))),
                },
                (Some(Const::Int(x)), TypeKey::Bool) if x == 0 || x == 1 => Ok(()),
                (Some(Const::Int(x)), TypeKey::Char)
                    if char::from_u32(x as u32).is_some() && (0..=0x10ffff).contains(&x) =>
                {
                    Ok(())
                }
                (Some(Const::F16(_)), TypeKey::Float(FloatBits::F16))
                | (Some(Const::BF16(_)), TypeKey::Float(FloatBits::B16))
                | (Some(Const::F32(_)), TypeKey::Float(FloatBits::F32))
                | (Some(Const::F64(_)), TypeKey::Float(FloatBits::F64)) => Ok(()),
                (k, _) => Err(format!("constant {k:?} cannot have type {}", self.tname(want))),
            },
            Operand::Str(_) => match key {
                // a string's bytes are read-only data: only a shared pointer to them
                TypeKey::Ref { inner, mutable: false } | TypeKey::RawPtr { inner, mutable: false }
                    if matches!(self.key(*inner), Some(TypeKey::Str)) =>
                {
                    Ok(())
                }
                _ => Err(format!("string operand cannot have type {}", self.tname(want))),
            },
            Operand::Func(fi) => {
                let g = self.m.func(fi).ok_or("function out of range")?;
                match key {
                    TypeKey::FnType { params, ret, .. }
                        if *params == g.params && self.ret_matches(g.ret, *ret) =>
                    {
                        Ok(())
                    }
                    _ => Err(format!("fn @{} does not have type {}", g.instance, self.tname(want))),
                }
            }
            Operand::Type(_) => Err("a type operand is not a value".into()),
        }
    }

    /// Whether `t` is a fn type with the params and return of `want`: fn types that differ only
    /// in their features share one representation, and the type checker has already decided
    /// that the value may drop or gain them (E0707), as for a fn item operand.
    fn same_fn_shape(&self, t: TypeId, want: &TypeKey) -> bool {
        match (self.key(t), want) {
            (
                Some(TypeKey::FnType { params: p1, ret: r1, .. }),
                TypeKey::FnType { params: p2, ret: r2, .. },
            ) => p1 == p2 && r1 == r2,
            _ => false,
        }
    }

    fn ret_matches(&self, r: Option<TypeId>, t: TypeId) -> bool {
        match r {
            Some(r) => r == t,
            None => self.key(t).is_some_and(|k| k.is_unit()),
        }
    }

    /// The class of an operand: from a value's type, or from a constant's form (`None` = an int
    /// constant, typed only by context).
    fn operand_cls(&self, b: &Body, o: Operand) -> Option<Cls> {
        match o {
            Operand::Value(v) => b.value_ty(v).map(|t| self.cls(t)),
            Operand::Const(c) => match b.konst(c) {
                Some(Const::F16(_)) => Some(Cls::Float(FloatBits::F16)),
                Some(Const::BF16(_)) => Some(Cls::Float(FloatBits::B16)),
                Some(Const::F32(_)) => Some(Cls::Float(FloatBits::F32)),
                Some(Const::F64(_)) => Some(Cls::Float(FloatBits::F64)),
                _ => None,
            },
            Operand::Str(_) => Some(Cls::Fat),
            Operand::Func(_) => Some(Cls::Fn),
            Operand::Type(_) => Some(Cls::Other),
        }
    }

    fn typing(&mut self, b: &Body) {
        for (bi, bd) in b.blocks.iter().enumerate() {
            let blk = Block(bi as u32);
            for i in bd.insts.indices() {
                if let Err(e) = self.check_inst(b, blk, i).and_then(|()| self.check_mutability(b, i)) {
                    self.err(Some(blk), Some(i), e);
                }
            }
            if let Err(e) = self.check_term(b, blk) {
                self.err(Some(blk), None, e);
            }
        }
    }

    fn check_args(&self, b: &Body, args: &[Operand], tys: &[TypeId]) -> Result<(), String> {
        if args.len() != tys.len() {
            return Err(format!("{} arguments, expected {}", args.len(), tys.len()));
        }
        for (k, (o, t)) in args.iter().zip(tys).enumerate() {
            self.check_operand(b, *o, *t).map_err(|e| format!("argument {k}: {e}"))?;
        }
        Ok(())
    }

    fn check_target(&self, b: &Body, t: &Target) -> Result<(), String> {
        let tys: Vec<TypeId> = b.params(t.block).iter().filter_map(|v| b.value_ty(*v)).collect();
        self.check_args(b, b.target_args(t), &tys).map_err(|e| format!("branch to b{}: {e}", t.block.0))
    }

    fn check_term(&self, b: &Body, blk: Block) -> Result<(), String> {
        let bd = b.block(blk).ok_or("block out of range")?;
        for t in b.successors(blk) {
            self.check_target(b, &t)?;
        }
        match bd.term {
            Term::Cbr { cond, .. } => {
                self.check_scalar_operand(b, cond, &TypeKey::Bool).map_err(|e| format!("cbr condition: {e}"))
            }
            Term::Switch { scrut, cases, .. } => {
                // At level `opt` an int constant may stand here: inlining substitutes arguments
                // for params, and only the branch folding after it removes such a switch. A
                // constant has no type, so its cases are checked for duplicates only.
                let t = match self.operand_ty(b, scrut) {
                    Some(t) => Some(t),
                    None if self.level == Level::Opt && b.int_const(scrut).is_some() => None,
                    None => return Err("switch scrutinee must be a value".into()),
                };
                if let Some(t) = t
                    && !matches!(self.cls(t), Cls::Int(_) | Cls::Bool | Cls::Char)
                {
                    return Err(format!("switch on {}", self.tname(t)));
                }
                let mut seen = rustc_hash::FxHashSet::default();
                for c in b.switch_cases(cases) {
                    if let Some(t) = t {
                        let ok = match self.cls(t) {
                            Cls::Int(it) => it.fits(c.value),
                            Cls::Bool => c.value == 0 || c.value == 1,
                            _ => (0..=0x10ffff).contains(&c.value),
                        };
                        if !ok {
                            return Err(format!("switch case {} out of range of {}", c.value, self.tname(t)));
                        }
                    }
                    if !seen.insert(c.value) {
                        return Err(format!("duplicate switch case {}", c.value));
                    }
                }
                Ok(())
            }
            Term::Ret(v) => match (v, self.f.ret) {
                (None, None) => Ok(()),
                (Some(o), Some(t)) => self.check_operand(b, o, t).map_err(|e| format!("ret: {e}")),
                (None, Some(_)) => Err("ret without a value in a fn with a return type".into()),
                (Some(_), None) => Err("ret with a value in a unit fn".into()),
            },
            Term::Br(_) | Term::Unreachable => Ok(()),
        }
    }

    fn check_inst(&self, b: &Body, blk: Block, i: u32) -> Result<(), String> {
        let op = b.insts.op[i as usize];
        let ty = b.insts.ty[i as usize];
        let args = b.args(i);
        let n = args.len();
        let arity = |k: usize| if n == k { Ok(()) } else { Err(format!("{n} operands, expected {k}")) };
        let res = || ty.ok_or_else(|| "missing result type".to_string());
        let no_res = || if ty.is_none() { Ok(()) } else { Err("must have no result".to_string()) };
        let tn = |t: TypeId| self.tname(t);
        let types = &self.m.types;
        match op {
            Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Rem => {
                arity(2)?;
                let t = res()?;
                match self.cls(t) {
                    Cls::Int(_) | Cls::Float(FloatBits::F32 | FloatBits::F64) => {}
                    _ => return Err(format!("arithmetic on {}", tn(t))),
                }
                self.check_args(b, args, &[t, t])
            }
            Op::AddWrap | Op::SubWrap | Op::MulWrap | Op::AddSat | Op::SubSat | Op::MulSat => {
                arity(2)?;
                let t = res()?;
                let Cls::Int(_) = self.cls(t) else { return Err(format!("int op on {}", tn(t))) };
                self.check_args(b, args, &[t, t])
            }
            Op::Neg => {
                arity(1)?;
                let t = res()?;
                match self.cls(t) {
                    Cls::Int(it) if it.signed => {}
                    Cls::Float(FloatBits::F32 | FloatBits::F64) => {}
                    _ => return Err(format!("neg on {}", tn(t))),
                }
                self.check_args(b, args, &[t])
            }
            Op::Shl | Op::Shr | Op::ShlWrap => {
                arity(2)?;
                let t = res()?;
                let Cls::Int(_) = self.cls(t) else { return Err(format!("shift of {}", tn(t))) };
                self.check_operand(b, args[0], t)?;
                match self.operand_cls(b, args[1]) {
                    Some(Cls::Int(_)) => Ok(()),
                    None if b.int_const(args[1]).is_some() => Ok(()),
                    _ => Err("shift amount must be an int".into()),
                }
            }
            Op::And | Op::Or | Op::Xor => {
                arity(2)?;
                let t = res()?;
                if !matches!(self.cls(t), Cls::Int(_) | Cls::Bool) {
                    return Err(format!("bitwise op on {}", tn(t)));
                }
                self.check_args(b, args, &[t, t])
            }
            Op::Not => {
                arity(1)?;
                let t = res()?;
                if !matches!(self.cls(t), Cls::Int(_) | Cls::Bool) {
                    return Err(format!("not on {}", tn(t)));
                }
                self.check_args(b, args, &[t])
            }
            Op::Popcount | Op::Clz | Op::Ctz => {
                arity(1)?;
                let t = res()?;
                let Cls::Int(_) = self.cls(t) else { return Err(format!("bit count of {}", tn(t))) };
                self.check_args(b, args, &[t])
            }
            Op::MulWide => {
                arity(2)?;
                let t = res()?;
                match self.key(t) {
                    Some(TypeKey::Tuple(ts)) if ts.len() == 2 && ts[0] == ts[1] => {
                        match self.cls(ts[0]) {
                            Cls::Int(it) if it.bits == 32 || it.bits == 64 => {}
                            _ => return Err("mul.wide needs a 32- or 64-bit int".into()),
                        }
                        self.check_args(b, args, &[ts[0], ts[0]])
                    }
                    _ => Err("mul.wide result must be a pair tuple".into()),
                }
            }
            Op::Eq | Op::Ne | Op::Lt | Op::Le | Op::Gt | Op::Ge => {
                arity(2)?;
                let t = res()?;
                if self.key(t) != Some(&TypeKey::Bool) {
                    return Err("compare result must be bool".into());
                }
                let ot = self.operand_ty(b, args[0]).or(self.operand_ty(b, args[1]));
                match ot {
                    Some(ot) => {
                        let ok = match self.cls(ot) {
                            Cls::Int(_) | Cls::Bool | Cls::Char | Cls::Float(_) => true,
                            Cls::Ptr | Cls::Fn => matches!(op, Op::Eq | Op::Ne),
                            _ => false,
                        };
                        if !ok {
                            return Err(format!("compare of {}", tn(ot)));
                        }
                        self.check_args(b, args, &[ot, ot])
                    }
                    None => match (self.operand_cls(b, args[0]), self.operand_cls(b, args[1])) {
                        (None, None) => Ok(()),
                        (Some(Cls::Float(x)), Some(Cls::Float(y))) if x == y => Ok(()),
                        (Some(Cls::Fn), Some(Cls::Fn)) if matches!(op, Op::Eq | Op::Ne) => Ok(()),
                        _ => Err("compare of mismatched constants".into()),
                    },
                }
            }
            Op::Select => {
                arity(3)?;
                let t = res()?;
                self.check_scalar_operand(b, args[0], &TypeKey::Bool)
                    .map_err(|e| format!("argument 0: {e}"))?;
                for k in 1..3 {
                    self.check_operand(b, args[k], t).map_err(|e| format!("argument {k}: {e}"))?;
                }
                Ok(())
            }
            Op::Cast(kind) => {
                arity(1)?;
                let t = res()?;
                let to_t = if kind == CastKind::Try {
                    let (_, _, p) =
                        option_shape(types, t).ok_or("cast.try result must be an Option-shaped enum")?;
                    p
                } else {
                    t
                };
                let to = cast_ty(self.cls(to_t)).ok_or_else(|| format!("cast to {}", tn(to_t)))?;
                match self.operand_cls(b, args[0]) {
                    None => {
                        let x = b.int_const(args[0]).ok_or("cast operand")?;
                        let src = if x < 0 { IntTy::new(64, true) } else { IntTy::new(64, false) };
                        if x > u64::MAX as i128 || x < i64::MIN as i128 {
                            return Err("cast constant out of range".into());
                        }
                        let ok = match kind {
                            CastKind::Exact => {
                                arith::convert(CastKind::Try, CastTy::Int(src), to, x as u64).is_some()
                            }
                            _ => arith::cast_allowed(kind, CastTy::Int(src), to),
                        };
                        if ok { Ok(()) } else { Err(format!("constant {x} cannot be cast to {}", tn(to_t))) }
                    }
                    Some(c) => {
                        let from = match args[0] {
                            Operand::Func(_) => CastTy::Ptr,
                            _ if c == Cls::Fn => return Err("cast of a fn value".into()),
                            _ => cast_ty(c).ok_or("cast of a non-scalar")?,
                        };
                        if arith::cast_allowed(kind, from, to) {
                            Ok(())
                        } else {
                            Err(format!("{:?} cast from {from:?} to {}", kind, tn(to_t)))
                        }
                    }
                }
            }
            Op::Make => {
                let t = res()?;
                match self.key(t) {
                    Some(TypeKey::Array { elem, len }) => {
                        if *len != n as u64 {
                            return Err(format!("make of {} with {n} elements", tn(t)));
                        }
                        let tys = vec![*elem; n];
                        self.check_args(b, args, &tys)
                    }
                    _ => {
                        let fs = record_fields(types, t).ok_or_else(|| format!("make of {}", tn(t)))?;
                        self.check_args(b, args, &fs)
                    }
                }
            }
            Op::Get | Op::Set => {
                arity(if op == Op::Get { 2 } else { 3 })?;
                let t = res()?;
                let at =
                    if op == Op::Get { self.operand_ty(b, args[0]).ok_or("get of a non-value")? } else { t };
                self.check_operand(b, args[0], at)?;
                let ft = match self.key(at) {
                    Some(TypeKey::Array { elem, len }) => {
                        match (b.int_const(args[1]), self.operand_cls(b, args[1])) {
                            (Some(k), _) if k >= 0 && (k as u128) < *len as u128 => {}
                            (Some(_), _) => return Err("array index constant out of range".into()),
                            (None, Some(Cls::Int(it))) if !it.signed => {}
                            _ => return Err("array index must be an unsigned int".into()),
                        }
                        *elem
                    }
                    _ => {
                        let fs = record_fields(types, at).ok_or_else(|| format!("field of {}", tn(at)))?;
                        let k = b.int_const(args[1]).ok_or("field index must be a constant")?;
                        *usize::try_from(k).ok().and_then(|k| fs.get(k)).ok_or("field index out of range")?
                    }
                };
                if op == Op::Get {
                    if ft != t {
                        return Err(format!("get result {} but field is {}", tn(t), tn(ft)));
                    }
                    Ok(())
                } else {
                    self.check_operand(b, args[2], ft)
                }
            }
            Op::EnumMake => {
                let t = res()?;
                let vs = enum_variants(types, t).ok_or_else(|| format!("enum.make of {}", tn(t)))?;
                if n == 0 {
                    return Err("enum.make needs a variant index".into());
                }
                let v = b.int_const(args[0]).ok_or("variant index must be a constant")?;
                let fs =
                    usize::try_from(v).ok().and_then(|v| vs.get(v)).ok_or("variant index out of range")?;
                self.check_args(b, &args[1..], fs)
            }
            Op::EnumTag => {
                arity(1)?;
                let t = res()?;
                let Cls::Int(it) = self.cls(t) else { return Err("enum.tag result must be an int".into()) };
                let et = self.operand_ty(b, args[0]).ok_or("enum.tag of a non-value")?;
                let vs = enum_variants(types, et).ok_or_else(|| format!("enum.tag of {}", tn(et)))?;
                if !it.fits(vs.len().saturating_sub(1) as i128) {
                    return Err("enum.tag result type too narrow".into());
                }
                Ok(())
            }
            Op::EnumGet => {
                arity(3)?;
                let t = res()?;
                let et = self.operand_ty(b, args[0]).ok_or("enum.get of a non-value")?;
                let ft = self.enum_field(b, et, args[1], args[2])?;
                if ft != t {
                    Err(format!("enum.get result {} but field is {}", tn(t), tn(ft)))
                } else {
                    Ok(())
                }
            }
            Op::Slot => {
                arity(0)?;
                let t = res()?;
                if blk.0 != 0 {
                    return Err("slot outside b0".into());
                }
                let first = b.block(blk).map_or(i, |bd| bd.insts.start);
                if i > first && b.insts.op[i as usize - 1] != Op::Slot {
                    return Err("slot after a non-slot instruction of b0 (slots are a prefix of b0)".into());
                }
                let p = pointee(types, t).ok_or("slot result must be a pointer type")?;
                if crate::layout::is_unsized_pointee(types, p) {
                    return Err("slot of an unsized type".into());
                }
                Ok(())
            }
            Op::Load => {
                arity(1)?;
                let t = res()?;
                let pt = self.operand_ty(b, args[0]).ok_or("load from a non-value")?;
                if pointee(types, pt) != Some(t) {
                    return Err(format!("load of {} through {}", tn(t), tn(pt)));
                }
                Ok(())
            }
            Op::Store => {
                arity(2)?;
                no_res()?;
                let pt = self.operand_ty(b, args[0]).ok_or("store to a non-value")?;
                let p = pointee(types, pt).ok_or("store through a non-pointer")?;
                self.check_operand(b, args[1], p)
            }
            Op::FieldAddr => {
                arity(2)?;
                let t = res()?;
                let pt = self.operand_ty(b, args[0]).ok_or("addr.field of a non-value")?;
                let rt = pointee(types, pt).ok_or("addr.field through a non-pointer")?;
                let fs = record_fields(types, rt).ok_or_else(|| format!("addr.field of {}", tn(rt)))?;
                let k = b.int_const(args[1]).ok_or("field index must be a constant")?;
                let ft =
                    *usize::try_from(k).ok().and_then(|k| fs.get(k)).ok_or("field index out of range")?;
                if pointee(types, t) != Some(ft) {
                    Err("addr.field result must point to the field type".into())
                } else {
                    Ok(())
                }
            }
            Op::EnumFieldAddr => {
                arity(3)?;
                let t = res()?;
                let pt = self.operand_ty(b, args[0]).ok_or("addr.enum_field of a non-value")?;
                let et = pointee(types, pt).ok_or("addr.enum_field through a non-pointer")?;
                let ft = self.enum_field(b, et, args[1], args[2])?;
                if pointee(types, t) != Some(ft) {
                    Err("addr.enum_field result must point to the field type".into())
                } else {
                    Ok(())
                }
            }
            Op::ElemAddr => {
                arity(2)?;
                let t = res()?;
                let pt = self.operand_ty(b, args[0]).ok_or("addr.elem of a non-value")?;
                let inner = pointee(types, pt).ok_or("addr.elem through a non-pointer")?;
                let et = match self.key(inner) {
                    Some(TypeKey::Array { elem, .. } | TypeKey::Slice { elem, .. }) => *elem,
                    _ if matches!(self.key(pt), Some(TypeKey::RawPtr { .. })) => inner,
                    _ => return Err(format!("addr.elem of {}", tn(pt))),
                };
                match (b.int_const(args[1]), self.operand_cls(b, args[1])) {
                    (Some(k), _) if fits_word(k) => {}
                    (None, Some(Cls::Int(_))) => {}
                    _ => return Err("element index must be an int (a constant in 64 bits)".into()),
                }
                if pointee(types, t) != Some(et) {
                    Err("addr.elem result must point to the element type".into())
                } else {
                    Ok(())
                }
            }
            Op::SliceMake => {
                arity(2)?;
                let t = res()?;
                let inner = pointee(types, t).ok_or("slice.make result must be a pointer")?;
                let pt = self.operand_ty(b, args[0]).ok_or("slice.make pointer must be a value")?;
                let elem_ok = match (self.key(inner), pointee(types, pt)) {
                    (Some(TypeKey::Slice { elem, .. }), Some(p)) => p == *elem,
                    (Some(TypeKey::Str), Some(p)) => {
                        self.key(p) == Some(&TypeKey::Int { signed: false, bits: IntBits::B8 })
                    }
                    (Some(TypeKey::Slice { .. } | TypeKey::Str), None) => false,
                    _ => return Err("slice.make result must point to a slice or str".into()),
                };
                if !elem_ok || self.cls(pt) != Cls::Ptr {
                    return Err("slice.make pointer must point to the element type".into());
                }
                self.check_scalar_operand(b, args[1], &TypeKey::Int { signed: false, bits: IntBits::Size })
            }
            Op::SliceLen | Op::SlicePtr => {
                arity(1)?;
                let t = res()?;
                if self.operand_cls(b, args[0]) != Some(Cls::Fat) {
                    return Err("slice.len / slice.ptr of a thin value".into());
                }
                if op == Op::SliceLen {
                    if self.key(t) != Some(&TypeKey::Int { signed: false, bits: IntBits::Size }) {
                        return Err("slice.len result must be usize".into());
                    }
                } else if self.cls(t) != Cls::Ptr {
                    return Err("slice.ptr result must be a thin pointer".into());
                }
                Ok(())
            }
            Op::FnMake => {
                // `fn.make @f env`: `@f` has the fn type's params plus a trailing environment
                // param (a thin pointer or `usize`), which `env` fills.
                arity(2)?;
                let t = res()?;
                let Some(TypeKey::FnType { params, ret, .. }) = self.key(t) else {
                    return Err("fn.make result must be a fn type".into());
                };
                let Operand::Func(fi) = args[0] else { return Err("fn.make code must be @fn".into()) };
                let g = self.m.func(fi).ok_or("fn.make callee out of range")?;
                let Some((&et, ps)) = g.params.split_last() else {
                    return Err("fn.make code has no environment param".into());
                };
                if ps != &params[..] || !self.ret_matches(g.ret, *ret) {
                    return Err(format!("fn.make: @{} does not match {}", g.instance, tn(t)));
                }
                match self.cls(et) {
                    Cls::Ptr => {}
                    Cls::Int(it) if !it.signed && it.bits == 64 => {}
                    _ => return Err("fn.make environment param must be a thin pointer or usize".into()),
                }
                self.check_operand(b, args[1], et)
            }
            Op::BoundsCheck => {
                arity(2)?;
                no_res()?;
                for &a in args {
                    match (b.int_const(a), self.operand_cls(b, a)) {
                        (Some(k), _) if (0..=u64::MAX as i128).contains(&k) => {}
                        (None, Some(Cls::Int(it))) if !it.signed => {}
                        _ => {
                            return Err(
                                "check.bounds operands must be unsigned ints (constants below 2^64)".into()
                            );
                        }
                    }
                }
                Ok(())
            }
            Op::Prefetch(_) => {
                arity(2)?;
                no_res()?;
                let pt = self.operand_ty(b, args[0]).ok_or("prefetch of a non-value")?;
                if pointee(types, pt).is_none() || self.cls(pt) != Cls::Ptr {
                    return Err("prefetch address must be a thin pointer".into());
                }
                match (b.int_const(args[1]), self.operand_cls(b, args[1])) {
                    (Some(k), _) if fits_word(k) => Ok(()),
                    (None, Some(Cls::Int(_))) => Ok(()),
                    _ => Err("prefetch index must be an int (a constant in 64 bits)".into()),
                }
            }
            Op::StoreStream => {
                arity(2)?;
                no_res()?;
                let pt = self.operand_ty(b, args[0]).ok_or("mem.store_stream to a non-value")?;
                let p = pointee(types, pt).ok_or("mem.store_stream through a non-pointer")?;
                match self.key(p) {
                    Some(TypeKey::Int { bits, .. }) if *bits != IntBits::Size => {}
                    Some(TypeKey::Float(_) | TypeKey::Vec { .. }) => {}
                    _ => return Err(format!("mem.store_stream of {}", tn(p))),
                }
                self.check_operand(b, args[1], p)
            }
            Op::FenceStream => {
                arity(0)?;
                no_res()
            }
            Op::Call => {
                if n == 0 {
                    return Err("call needs a callee".into());
                }
                let Operand::Func(fi) = args[0] else {
                    return Err("call callee must be @fn (use call.ind)".into());
                };
                let g = self.m.func(fi).ok_or("callee out of range")?;
                if ty != g.ret {
                    return Err("call result type differs from the callee's return type".into());
                }
                if g.variadic && args.len() > g.params.len() + 1 {
                    let (fixed, extra) = args[1..].split_at(g.params.len());
                    self.check_args(b, fixed, &g.params)?;
                    for (k, &o) in extra.iter().enumerate() {
                        // the C-passable variadic args (E0813); an int constant is typed by context
                        let ok = match self.operand_cls(b, o) {
                            None => matches!(o, Operand::Const(_)),
                            Some(Cls::Int(it)) => it.bits == 32 || it.bits == 64,
                            Some(c) => matches!(c, Cls::Float(FloatBits::F64) | Cls::Ptr),
                        };
                        if !ok {
                            let at = g.params.len() + k;
                            return Err(format!("variadic argument {at}: not a C variadic type"));
                        }
                    }
                    return Ok(());
                }
                self.check_args(b, &args[1..], &g.params)
            }
            Op::CallInd => {
                if n == 0 {
                    return Err("call.ind needs a callee".into());
                }
                if let Operand::Func(fi) = args[0] {
                    // a fn item as the fn value (code, null environment)
                    let g = self.m.func(fi).ok_or("callee out of range")?;
                    if ty != g.ret {
                        return Err("call.ind result type differs from the callee's return type".into());
                    }
                    return self.check_args(b, &args[1..], &g.params);
                }
                let ft = self.operand_ty(b, args[0]).ok_or("call.ind callee must be a value")?;
                let Some(TypeKey::FnType { params, ret, .. }) = self.key(ft) else {
                    return Err(format!("call.ind of {}", tn(ft)));
                };
                let ok = match ty {
                    Some(t) => t == *ret,
                    None => self.key(*ret).is_some_and(|k| k.is_unit()),
                };
                if !ok {
                    return Err("call.ind result type differs from the fn type's return".into());
                }
                self.check_args(b, &args[1..], params)
            }
            Op::Intrinsic(id) => {
                let name = self.m.intrinsic_name(id).ok_or("intrinsic id out of range")?;
                self.check_intrinsic(b, name, args, ty)
            }
            Op::Vec(v) => self.check_vec(b, v, ty, args),
            Op::Panic(_) => {
                arity(0)?;
                no_res()
            }
        }
    }

    fn enum_field(&self, b: &Body, et: TypeId, v: Operand, f: Operand) -> Result<TypeId, String> {
        let vs =
            enum_variants(&self.m.types, et).ok_or_else(|| format!("variant field of {}", self.tname(et)))?;
        let v = b.int_const(v).ok_or("variant index must be a constant")?;
        let f = b.int_const(f).ok_or("field index must be a constant")?;
        let fs = usize::try_from(v).ok().and_then(|v| vs.get(v)).ok_or("variant index out of range")?;
        usize::try_from(f)
            .ok()
            .and_then(|f| fs.get(f))
            .copied()
            .ok_or_else(|| "field index out of range".into())
    }

    /// `intrinsic.<name>` against its entry of [`crate::intrinsics`], or [`Kind::StaticAddr`] against
    /// the module's statics. A name the table lacks, or one without a checked signature, is
    /// rejected: no back end builds it.
    fn check_intrinsic(
        &self,
        b: &Body,
        name: &str,
        args: &[Operand],
        ty: Option<TypeId>,
    ) -> Result<(), String> {
        let t = &self.m.types;
        let kind = intrinsics::kind(name).ok_or_else(|| format!("unknown intrinsic `{name}`"))?;
        match kind {
            Kind::StaticAddr => {
                let [k] = args else {
                    return Err(format!("intrinsic.{name}: {} operands, expected 1", args.len()));
                };
                let k = b
                    .int_const(*k)
                    .ok_or_else(|| format!("intrinsic.{name}: the index must be an int constant"))?;
                let st = usize::try_from(k)
                    .ok()
                    .and_then(|k| self.m.statics.get(k))
                    .ok_or_else(|| format!("intrinsic.{name}: static {k} out of range"))?;
                return match ty.and_then(|r| t.key(r)) {
                    Some(TypeKey::RawPtr { inner, .. }) if *inner == st.ty => Ok(()),
                    _ => Err(format!("intrinsic.{name}: the result must be a raw pointer to t{}", st.ty.0)),
                };
            }
            Kind::CpuBaseline => {
                return Err(format!("intrinsic.{name}: lowering replaces it by a constant"));
            }
            Kind::Ptx => {
                return Err(format!("intrinsic.{name}: no checked signature (the {:?} set)", kind.arch()));
            }
            Kind::Syscall(_)
            | Kind::Ccall(_)
            | Kind::Cpuid
            | Kind::Xgetbv
            | Kind::ThreadStartAddr
            | Kind::MainAddr
            | Kind::SignalHandlerAddr
            | Kind::SiteTable
            | Kind::PcTable
            | Kind::StackTable
            | Kind::ThreadBlock
            | Kind::SetThreadBlock
            | Kind::CallOnStack
            | Kind::CpuInit
            | Kind::ThreadClone
            | Kind::Atomic { .. }
            | Kind::CopyBytes => {}
        }
        let Some((params, ret)) = kind.sig() else {
            return Err(format!("intrinsic.{name}: no checked signature"));
        };
        let resolve =
            |x: ITy| ity_type(t, x).ok_or_else(|| format!("intrinsic.{name}: {x:?} is not in the module"));
        match (ret, ty) {
            (ITy::Unit, None) => {}
            (ITy::Unit, Some(_)) => return Err(format!("intrinsic.{name}: must have no result")),
            (r, Some(rt)) if resolve(r)? == rt => {}
            _ => return Err(format!("intrinsic.{name}: wrong result type")),
        }
        let ps = params.iter().map(|&p| resolve(p)).collect::<Result<Vec<_>, _>>()?;
        self.check_args(b, args, &ps).map_err(|e| format!("intrinsic.{name}: {e}"))
    }
}

/// Whether an int constant typed only by context fits a 64-bit int of either signedness.
fn fits_word(k: i128) -> bool {
    (i64::MIN as i128..=u64::MAX as i128).contains(&k)
}

/// The id of intrinsic table type `x` in `types`; `None` for `()` or a type the table lacks.
fn ity_type(types: &TypeTable, x: ITy) -> Option<TypeId> {
    let int = |signed, bits| types.find(&TypeKey::Int { signed, bits });
    let ptr_to =
        |mutable, bits| int(false, bits).and_then(|inner| types.find(&TypeKey::RawPtr { mutable, inner }));
    let ptr = |mutable| ptr_to(mutable, IntBits::B8);
    match x {
        ITy::ConstPtrUsize => ptr_to(false, IntBits::Size),
        ITy::Usize => int(false, IntBits::Size),
        ITy::Isize => int(true, IntBits::Size),
        ITy::U32 => int(false, IntBits::B32),
        ITy::U64 => int(false, IntBits::B64),
        ITy::Unit => None,
        ITy::U32x4 => int(false, IntBits::B32).and_then(|u| types.find(&TypeKey::Tuple(vec![u; 4]))),
        ITy::MutPtrU8 => ptr(true),
        ITy::ConstPtrU8 => ptr(false),
    }
}

/// Dominator tree of the reachable blocks (Cooper, Harvey, Kennedy), with DFS intervals for O(1)
/// queries.
pub struct Dominators {
    idom: Vec<Option<usize>>,
    tin: Vec<u32>,
    tout: Vec<u32>,
}

impl Dominators {
    pub fn compute(b: &Body) -> Dominators {
        let n = b.blocks.len();
        let succ: Vec<Vec<usize>> = (0..n)
            .map(|i| {
                let mut v = Vec::new();
                b.for_each_successor(Block(i as u32), |t| {
                    if t.block.index() < n {
                        v.push(t.block.index());
                    }
                });
                v
            })
            .collect();
        // Iterative DFS postorder from b0.
        let mut post = Vec::with_capacity(n);
        let mut seen = vec![false; n];
        if n > 0 {
            let mut stack = vec![(0usize, 0usize)];
            seen[0] = true;
            while let Some((node, k)) = stack.last_mut() {
                if let Some(&s) = succ[*node].get(*k) {
                    *k += 1;
                    if !seen[s] {
                        seen[s] = true;
                        stack.push((s, 0));
                    }
                } else {
                    post.push(*node);
                    stack.pop();
                }
            }
        }
        let mut po_num = vec![usize::MAX; n];
        for (i, &bk) in post.iter().enumerate() {
            po_num[bk] = i;
        }
        let mut preds = vec![Vec::new(); n];
        for (i, ss) in succ.iter().enumerate() {
            if seen[i] {
                for &s in ss {
                    preds[s].push(i);
                }
            }
        }
        let mut idom: Vec<Option<usize>> = vec![None; n];
        if n > 0 {
            idom[0] = Some(0);
        }
        let intersect = |idom: &[Option<usize>], mut a: usize, mut b: usize| -> usize {
            while a != b {
                while po_num[a] < po_num[b] {
                    a = idom[a].unwrap_or(0);
                }
                while po_num[b] < po_num[a] {
                    b = idom[b].unwrap_or(0);
                }
            }
            a
        };
        let mut changed = true;
        while changed {
            changed = false;
            for &bk in post.iter().rev() {
                if bk == 0 {
                    continue;
                }
                let mut new: Option<usize> = None;
                for &p in &preds[bk] {
                    if idom[p].is_some() {
                        new = Some(match new {
                            None => p,
                            Some(x) => intersect(&idom, p, x),
                        });
                    }
                }
                if new.is_some() && idom[bk] != new {
                    idom[bk] = new;
                    changed = true;
                }
            }
        }
        // DFS intervals over the dominator tree.
        let mut children = vec![Vec::new(); n];
        for (i, d) in idom.iter().enumerate() {
            if let Some(d) = *d
                && i != 0
            {
                children[d].push(i);
            }
        }
        let mut tin = vec![0u32; n];
        let mut tout = vec![0u32; n];
        let mut clock = 0u32;
        if n > 0 {
            let mut stack = vec![(0usize, 0usize)];
            tin[0] = clock;
            clock += 1;
            while let Some((node, k)) = stack.last_mut() {
                if let Some(&c) = children[*node].get(*k) {
                    *k += 1;
                    tin[c] = clock;
                    clock += 1;
                    stack.push((c, 0));
                } else {
                    tout[*node] = clock;
                    clock += 1;
                    stack.pop();
                }
            }
        }
        Dominators { idom, tin, tout }
    }

    pub fn reachable(&self, b: usize) -> bool {
        self.idom.get(b).is_some_and(|d| d.is_some())
    }

    /// `a` dominates `b` (both reachable).
    pub fn dominates(&self, a: usize, b: usize) -> bool {
        self.reachable(a) && self.reachable(b) && self.tin[a] <= self.tin[b] && self.tout[b] <= self.tout[a]
    }

    pub fn idom(&self, b: usize) -> Option<usize> {
        self.idom.get(b).copied().flatten()
    }

    /// The position of reachable block `b` in a preorder walk of the dominator tree: a block
    /// comes after every block that dominates it.
    pub fn preorder(&self, b: usize) -> u32 {
        self.tin.get(b).copied().unwrap_or(0)
    }
}
