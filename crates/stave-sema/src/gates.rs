//! SIMD lowering gates (contracts/simd-ops.md). The class of an op on the build target is its
//! row of the target's lowering table at the target's baseline features
//! (`stave_target::lowering`):
//! - E0700: a class U op outside every `emulate { }` block;
//! - E0701: an op inside a `strict { }` block that is class M or U, or that the table does not list;
//! - E0702: a vector or mask wider than the target's vector registers (128 bits; 256 on
//!   `x86_64-linux-v3`, 512 on `x86_64-linux-v4`), where a type is written and where a built-in
//!   returns a value wider than its receiver.
//!
//! The blocks are lexical: a closure inside one is inside it. A fn with `needs(...)` is not gated
//! here: its features, not the target baseline, decide what it may use (the values its body holds:
//! [`crate::inherit::gate_bodies`]). Ops on a vector with a generic
//! lane or a sizeless lane count are not gated. A gather or scatter with 8- or 16-bit data or
//! index lanes is class U; otherwise it takes the worse class of its data and index lane widths.
//! On `gpu-sm86` only E0702 applies (at 128 bits). A class U op inside `emulate` is recorded for
//! the build summary ([`Emulated`]).

use stave_diag::Span;
use stave_syntax::Term;
use stave_target::lowering::{self, Class, Tier};
use stave_target::{Arch, Feature, FeatureGroup, FeatureSet, Triple};

use crate::diag::mk;
use crate::prim::Lanes;
use crate::typeck::Ck;
use crate::types::{self, FloatBits, Interner, TypeId, TypeKind};

/// A class U op inside `emulate { }`: it runs as a scalar loop over the lanes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Emulated {
    pub span: Span,
    /// The IR name (`vec.gather`).
    pub op: String,
    pub ty: String,
}

/// The gate state of a body.
#[derive(Debug, Default)]
pub(crate) struct Gates {
    /// The number of enclosing `emulate` blocks.
    emulate: u32,
    /// The `strict` token of the innermost enclosing `strict` block.
    strict: Option<u32>,
    pub emulated: Vec<Emulated>,
    /// E0702 spans already reported (a type node may be lowered more than once).
    wide: Vec<Span>,
}

impl Gates {
    /// Enters a `strict` (`strict` true) or `emulate` block whose keyword is `kw`; the result is
    /// the state to give [`Gates::leave`].
    pub fn enter(&mut self, kw: u32, strict: bool) -> (u32, Option<u32>) {
        let saved = (self.emulate, self.strict);
        if strict {
            self.strict = Some(kw);
        } else {
            self.emulate += 1;
        }
        saved
    }

    pub fn leave(&mut self, saved: (u32, Option<u32>)) {
        (self.emulate, self.strict) = saved;
    }
}

/// The widest vector or mask, in bits, that target `t` has without `needs(...)`.
pub fn max_vector_bits(t: Triple) -> u32 {
    feature_vector_bits(t.baseline())
}

/// The features 512-bit vectors and masks take: the AVX-512 part of `v4`, not `avx512f` alone.
fn wide512() -> &'static [Feature] {
    FeatureGroup::V4.own_members()
}

/// The widest vector or mask, in bits, that code with features `fs` may use.
pub fn feature_vector_bits(fs: FeatureSet) -> u32 {
    if wide512().iter().all(|f| fs.contains(*f)) {
        512
    } else if fs.contains(Feature::Avx2) {
        256
    } else {
        128
    }
}

/// The first fixed vector or mask in `t` wider than `max` bits, with its width. Looks through
/// refs, pointers, arrays, slices, tuples, and generic arguments, not into fn types (each fn type
/// is checked where it is written).
pub(crate) fn wide_part<I: Interner + ?Sized>(cx: &I, t: TypeId, max: u32) -> Option<(TypeId, u32)> {
    // Each distinct type is visited once: exact, with no cutoff.
    let mut seen: rustc_hash::FxHashSet<TypeId> = rustc_hash::FxHashSet::default();
    let mut stack = vec![t];
    while let Some(t) = stack.pop() {
        if !seen.insert(t) {
            continue;
        }
        match cx.kind(t) {
            TypeKind::Vector { lane, n: Lanes::Fixed(k) } => {
                let bits = types::lane_bits(cx.kind(*lane)) * *k as u32;
                if bits > max {
                    return Some((t, bits));
                }
            }
            TypeKind::Mask { width, n: Lanes::Fixed(k) } => {
                let bits = *width as u32 * *k as u32;
                if bits > max {
                    return Some((t, bits));
                }
            }
            TypeKind::Ref { inner, .. } | TypeKind::Ptr { inner, .. } => stack.push(*inner),
            TypeKind::Array { elem, .. } | TypeKind::Slice { elem, .. } => stack.push(*elem),
            TypeKind::Tuple(ts) => stack.extend(ts.iter().rev()),
            TypeKind::Adt { args, .. } => stack.extend(args.iter().rev().filter_map(|a| match a {
                types::GArg::Type(x) => Some(*x),
                types::GArg::Const(_) => None,
            })),
            _ => {}
        }
    }
    None
}

/// The first vector in signature type `t` that a signature with features `features` cannot hold
/// on `target` (the baseline counts), with its width.
pub(crate) fn sig_wide_part<I: Interner + ?Sized>(
    cx: &I,
    target: Triple,
    t: TypeId,
    features: u64,
) -> Option<(TypeId, u32)> {
    let fs = FeatureSet::from_bits(features).unwrap_or_default();
    wide_part(cx, t, feature_vector_bits(target.baseline().union(fs)))
}

/// The features whose registers hold a `bits`-wide vector or mask on `target`, beyond 128 bits
/// (empty: none of its arch's features is that wide).
pub(crate) fn width_features(target: Triple, bits: u32) -> Vec<Feature> {
    match target.arch() {
        Arch::X86_64 if bits <= 256 => vec![Feature::Avx2],
        Arch::X86_64 if bits <= 512 => wide512().to_vec(),
        _ => Vec::new(),
    }
}

/// E0703 for vector type `ty` (`bits` wide) in a signature at `span` whose features `features`
/// (with the target baseline) do not cover it, naming the missing features (those not implied by
/// another, in feature order); E0702 where no feature of the target's arch is that wide.
pub(crate) fn sig_too_wide(
    target: Triple,
    features: u64,
    ty: &str,
    bits: u32,
    span: Span,
) -> Option<stave_diag::Diagnostic> {
    let need = width_features(target, bits);
    let have = target.baseline().union(FeatureSet::from_bits(features).unwrap_or_default());
    let missing = FeatureSet::from_features(need.into_iter().filter(|f| !have.contains(*f)));
    let names = types::feature_names(missing.bits());
    if names.is_empty() {
        return too_wide(ty, bits, max_vector_bits(target), span);
    }
    mk("E0703", None, &[("type", ty), ("feature", &names.join(", "))], span, &[])
}

/// E0702 for a `bits`-wide vector or mask type `ty` at `span`, if it is wider than `max`.
pub(crate) fn too_wide(ty: &str, bits: u32, max: u32, span: Span) -> Option<stave_diag::Diagnostic> {
    if bits <= max {
        return None;
    }
    let (b, m) = (bits.to_string(), max.to_string());
    mk("E0702", None, &[("type", ty), ("bits", &b), ("max", &m)], span, &[])
}

/// The table key of a lane type (`i32`, `bf16`); `None` for a type that is no lane type.
fn lane_key(k: &TypeKind) -> Option<String> {
    Some(match k {
        TypeKind::Int { signed, bits } => format!("{}{}", if *signed { 'i' } else { 'u' }, bits.bits()),
        TypeKind::Float(FloatBits::F16) => "f16".into(),
        TypeKind::Float(FloatBits::B16) => "bf16".into(),
        TypeKind::Float(FloatBits::F32) => "f32".into(),
        TypeKind::Float(FloatBits::F64) => "f64".into(),
        _ => return None,
    })
}

/// The IR op of binary operator `op` (a token kind) on vectors, or on masks.
fn binop_name(op: u8, mask: bool) -> Option<&'static str> {
    let names: &[(Term, &str, &str)] = &[
        (Term::Plus, "vec.add", ""),
        (Term::Minus, "vec.sub", ""),
        (Term::Star, "vec.mul", ""),
        (Term::Slash, "vec.div", ""),
        (Term::PlusPercent, "vec.add.wrap", ""),
        (Term::MinusPercent, "vec.sub.wrap", ""),
        (Term::StarPercent, "vec.mul.wrap", ""),
        (Term::PlusPipe, "vec.add.sat", ""),
        (Term::MinusPipe, "vec.sub.sat", ""),
        (Term::StarPipe, "vec.mul.sat", ""),
        (Term::LtLt, "vec.shl", ""),
        (Term::GtGt, "vec.shr", ""),
        (Term::LtLtPercent, "vec.shl.wrap", ""),
        (Term::Amp, "vec.and", "mask.and"),
        (Term::Pipe, "vec.or", "mask.or"),
        (Term::Caret, "vec.xor", "mask.xor"),
    ];
    let &(_, v, m) = names.iter().find(|x| x.0 as u8 == op)?;
    let name = if mask { m } else { v };
    if name.is_empty() { None } else { Some(name) }
}

/// The IR op of compare operator `op` (a token kind) on vectors.
fn compare_name(op: u8) -> Option<&'static str> {
    let names: &[(Term, &str)] = &[
        (Term::EqEq, "vec.eq"),
        (Term::BangEq, "vec.ne"),
        (Term::Lt, "vec.lt"),
        (Term::LtEq, "vec.le"),
        (Term::Gt, "vec.gt"),
        (Term::GtEq, "vec.ge"),
    ];
    names.iter().find(|x| x.0 as u8 == op).map(|x| x.1)
}

/// The IR op of cast operator `op` (a token kind) on vectors.
pub(crate) fn cast_name(op: u8) -> Option<&'static str> {
    let names: &[(Term, &str)] = &[
        (Term::KwAs, "vec.cast"),
        (Term::AsPercent, "vec.cast.wrap"),
        (Term::AsPipe, "vec.cast.sat"),
        (Term::AsTilde, "vec.cast.round"),
        (Term::AsQuestion, "vec.cast.try"),
    ];
    names.iter().find(|x| x.0 as u8 == op).map(|x| x.1)
}

fn rank(c: Option<Class>) -> u8 {
    match c {
        Some(Class::N) => 0,
        Some(Class::M) | None => 1,
        Some(Class::U) => 2,
    }
}

impl Ck<'_> {
    /// The tier whose table gates this body: `None` in a `needs(...)` fn and on the GPU.
    fn gate_tier(&self) -> Option<Tier> {
        if self.features != 0 { None } else { Tier::of(self.prog.target) }
    }

    /// The table key of fixed vector or mask type `t` (`f32`, `m32`).
    pub(crate) fn vec_key(&self, t: TypeId) -> Option<String> {
        match self.tys.kind(t) {
            TypeKind::Vector { lane, n: Lanes::Fixed(_) } => lane_key(self.tys.kind(*lane)),
            TypeKind::Mask { width, n: Lanes::Fixed(_) } => Some(format!("m{width}")),
            _ => None,
        }
    }

    fn table_class(&self, tier: Tier, op: &str, key: &str, form: &str) -> Option<Class> {
        lowering::lookup_with(op, key, form, tier, self.prog.target.baseline()).map(|l| l.class)
    }

    /// Gates op `op` (IR name) at token `tok` on fixed vector or mask type `t` (keyed by its lane
    /// type, or its width for a mask). `form`: the shift amount form, else empty.
    pub(crate) fn gate(&mut self, tok: u32, op: &str, t: TypeId, form: &str) {
        let Some(key) = self.vec_key(t) else { return };
        self.gate_key(tok, op, &key, form, t);
    }

    /// [`Ck::gate`] with an explicit table key (a cast's `i32>f32`).
    pub(crate) fn gate_key(&mut self, tok: u32, op: &str, key: &str, form: &str, t: TypeId) {
        let Some(tier) = self.gate_tier() else { return };
        let class = self.table_class(tier, op, key, form);
        self.gate_class(tok, op, class, t);
    }

    /// The cast `from` → `to` (both vectors) of IR op `op`.
    pub(crate) fn gate_cast(&mut self, tok: u32, op: &str, from: TypeId, to: TypeId) {
        let (Some(f), Some(t)) = (self.vec_key(from), self.vec_key(to)) else { return };
        self.gate_key(tok, op, &format!("{f}>{t}"), "", from);
    }

    /// Binary operator `op` (a token kind) on vector or mask type `t`; `form`: the shift amount.
    pub(crate) fn gate_binop(&mut self, tok: u32, op: u8, t: TypeId, form: &str) {
        let mask = match self.tys.kind(t) {
            TypeKind::Vector { .. } => false,
            TypeKind::Mask { .. } => true,
            _ => return,
        };
        if let Some(name) = binop_name(op, mask) {
            self.gate(tok, name, t, form);
        }
    }

    /// Compare operator `op` (a token kind) on vector or mask operands of type `t`.
    pub(crate) fn gate_compare(&mut self, tok: u32, op: u8, t: TypeId) {
        let name = match self.tys.kind(t) {
            TypeKind::Vector { .. } => compare_name(op),
            TypeKind::Mask { .. } => Some("mask.xor"),
            _ => None,
        };
        if let Some(name) = name {
            self.gate(tok, name, t, "");
        }
    }

    /// Built-in method `name` (name token `tok`) on receiver `recv` with result `ret` and argument
    /// types `args`; also E0702 for a result wider than the receiver.
    pub(crate) fn gate_method(&mut self, tok: u32, name: &str, recv: TypeId, ret: TypeId, args: &[TypeId]) {
        match self.tys.kind(recv) {
            TypeKind::Vector { .. } if name == "scatter" => {
                if let Some(&idx) = args.get(1) {
                    self.gate_indexed(tok, "vec.scatter", recv, idx);
                }
            }
            TypeKind::Vector { .. } => self.gate(tok, &format!("vec.{name}"), recv, ""),
            TypeKind::Mask { .. } if name == "select" => self.gate(tok, "mask.select", ret, ""),
            TypeKind::Mask { .. } => self.gate(tok, &format!("mask.{name}"), recv, ""),
            _ => return,
        }
        if self.vec_bits(ret) > self.vec_bits(recv) {
            let span = self.icx.tok_span(tok);
            self.gate_width(span, ret);
        }
    }

    /// Type-relative built-in `name` (`f32x4.splat`, name token `tok`) of type `t`.
    pub(crate) fn gate_assoc(&mut self, tok: u32, name: &str, t: TypeId, ret: TypeId, args: &[TypeId]) {
        match self.tys.kind(t) {
            TypeKind::Vector { .. } if name == "gather" => {
                if let Some(&idx) = args.get(1) {
                    self.gate_indexed(tok, "vec.gather", t, idx);
                }
            }
            TypeKind::Vector { .. } => self.gate(tok, &format!("vec.{name}"), t, ""),
            TypeKind::Mask { .. } => self.gate(tok, &format!("mask.{name}"), t, ""),
            _ => return,
        }
        if self.vec_bits(ret) > self.vec_bits(t) {
            let span = self.icx.tok_span(tok);
            self.gate_width(span, ret);
        }
    }

    /// A gather or scatter of data vector `t` with index vector `idx` (the index of a well-typed
    /// call: unsigned lanes, as many as `t` has).
    pub(crate) fn gate_indexed(&mut self, tok: u32, op: &str, t: TypeId, idx: TypeId) {
        let Some(tier) = self.gate_tier() else { return };
        let (TypeKind::Vector { n, .. }, TypeKind::Vector { lane: il, n: inn }) =
            (self.tys.kind(t), self.tys.kind(idx))
        else {
            return;
        };
        if n != inn || !matches!(self.tys.kind(*il), TypeKind::Int { signed: false, .. }) {
            return;
        }
        let (Some(k), Some(ik)) = (self.vec_key(t), self.vec_key(idx)) else { return };
        let narrow = |s: &Ck<'_>, t: TypeId| match s.tys.kind(t) {
            TypeKind::Vector { lane, .. } => types::lane_bits(s.tys.kind(*lane)) < 32,
            _ => true,
        };
        let class = if narrow(self, t) || narrow(self, idx) {
            Some(Class::U)
        } else {
            let (a, b) = (self.table_class(tier, op, &k, ""), self.table_class(tier, op, &ik, ""));
            if rank(a) >= rank(b) { a } else { b }
        };
        self.gate_class(tok, op, class, t);
    }

    fn gate_class(&mut self, tok: u32, op: &str, class: Option<Class>, t: TypeId) {
        let span = self.icx.tok_span(tok);
        if let Some(s) = self.gates.strict {
            if class == Some(Class::N) {
                return;
            }
            let c = if class == Some(Class::U) { "U" } else { "M" };
            let tn = self.show(t);
            let sp = self.icx.tok_span(s);
            let d = mk("E0701", None, &[("op", op), ("type", &tn), ("class", c)], span, &[("strict", sp)]);
            self.report(d);
        } else if class == Some(Class::U) {
            let tn = self.show(t);
            if self.gates.emulate > 0 {
                self.gates.emulated.push(Emulated { span, op: op.to_string(), ty: tn });
            } else {
                let target = self.prog.target.name();
                let d = mk("E0700", None, &[("op", op), ("type", &tn), ("target", target)], span, &[]);
                self.report(d);
            }
        }
    }

    /// E0702 at `span` if fixed vector or mask type `t` is wider than the target's registers.
    pub(crate) fn gate_width(&mut self, span: Span, t: TypeId) {
        let bits = self.vec_bits(t);
        let tn = self.show(t);
        self.gate_width_of(span, &tn, bits);
    }

    /// The bits of a fixed vector or mask type (0 for any other type).
    pub(crate) fn vec_bits(&self, t: TypeId) -> u32 {
        match self.tys.kind(t) {
            TypeKind::Vector { lane, n: Lanes::Fixed(k) } => {
                types::lane_bits(self.tys.kind(*lane)) * *k as u32
            }
            TypeKind::Mask { width, n: Lanes::Fixed(k) } => *width as u32 * *k as u32,
            _ => 0,
        }
    }

    /// E0703 for signature type `t` at `span` of a closure or fn type with features `features`,
    /// in a body with `needs(...)` (elsewhere E0702 has gated every written vector).
    pub(crate) fn sig_width_at(&mut self, span: Span, t: TypeId, features: u64) {
        if self.features == 0 || self.gates.wide.contains(&span) {
            return;
        }
        let target = self.prog.target;
        let Some((v, bits)) = sig_wide_part(&self.tys, target, t, features) else { return };
        let name = self.show(v);
        self.gates.wide.push(span);
        let d = sig_too_wide(target, features, &name, bits, span);
        self.report(d);
    }

    pub(crate) fn gate_width_of(&mut self, span: Span, ty: &str, bits: u32) {
        if self.features != 0 || self.gates.wide.contains(&span) {
            return;
        }
        let d = too_wide(ty, bits, max_vector_bits(self.prog.target), span);
        if d.is_some() {
            self.gates.wide.push(span);
            self.report(d);
        }
    }
}
