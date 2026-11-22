//! Inputs of a `test fn` with parameters (contracts/cli.md "Test fn inputs").
//!
//! **Domain** ([`Ck::test_domain`]): the product of the parameter types, each a [`Gen`]. A
//! parameter type the runner can generate is an int (`i8`..`i64`, `u8`..`u64`, `isize`, `usize`),
//! `bool`, `char`, `f32`, `f64`, an array, tuple, or non-view struct of those (nested freely), or,
//! at the top level only, `&[T]` of those. A struct also needs every field, and every struct type
//! of the test's package inside, visible from the test's module (case `private`). Anything else
//! is E0214 (case `field`: the struct field that is not generatable, and its type; a struct of
//! another package counts as one such type); so is a non-`()` return type (case `ret`), a slice
//! parameter without a length bound (case `unbounded`), and a value of more than [`MAX_SCALARS`]
//! scalars (case `size`).
//!
//! **Ranges**: each `requires` clause is split at `&&` (through parentheses); a conjunct that
//! compares an int parameter, an int field (`p.a.b`) of a struct parameter, or `p.len()` of a
//! slice parameter, with a constant (an int literal, its negation, or an int `const`) with `<`,
//! `<=`, `>`, `>=`, or `==` narrows that range. Every other conjunct is a filter: the lowered test
//! fn evaluates all its clauses at entry and skips an input that fails one (lowering,
//! [`crate::contracts`]).
//!
//! **Inputs** ([`inputs`]): the edge phase, then the sampled phase, an input equal to an earlier one
//! dropped. Everything here is a pure function of the domain, so every target and both compilers
//! produce the same list.

use std::collections::HashSet;

use stave_syntax::{NodeKind, Term};

use crate::decls::DeclFlags;
use crate::diag::mk;
use crate::resolve::Res;
use crate::sigs::FnSig;
use crate::typeck::{Ck, Target};
use crate::types::{ConstArg, FloatBits, GArg, IntBits, Interner, Names, TypeId, TypeKind, ids};

/// The most scalars one generated parameter value may hold (a slice counts its length bound).
pub const MAX_SCALARS: u64 = 4096;

/// How many inputs the sampled phase draws.
pub const SAMPLES: usize = 256;

/// The generator's initial state.
pub const SEED: u64 = 0;

/// The values of one parameter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Gen {
    /// An int type (`ty` its name, the literal suffix) narrowed to `lo..=hi`.
    Int {
        ty: &'static str,
        lo: i128,
        hi: i128,
    },
    Bool,
    Char,
    F32,
    F64,
    Array(Box<Gen>, u64),
    Tuple(Vec<Gen>),
    /// A struct: the path of its literal from the test's module (`geo.Point`, `Pair::<u32>`) and
    /// its fields in declaration order.
    Struct {
        path: String,
        fields: Vec<(String, Gen)>,
    },
    /// `&[T]` with a length in `lo..=hi` (`hi` = `i128::MAX` until a clause bounds it).
    Slice(Box<Gen>, i128, i128),
}

/// One parameter of the domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Param {
    pub name: String,
    pub values: Gen,
}

/// The input domain of a `test fn` with parameters ([`crate::TypedBody::test_domain`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Domain {
    pub params: Vec<Param>,
    /// The `requires` conjuncts that did not become ranges (each input is filtered by them).
    pub filters: u32,
}

/// A generated value. Arrays, tuples, structs, and slices are lists; floats are their bits.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Value {
    Int(i128),
    Bool(bool),
    Char(u32),
    F32(u32),
    F64(u64),
    List(Vec<Value>),
}

/// SplitMix64: `state += 0x9E3779B97F4A7C15`, then the output mix of the new state.
#[derive(Clone, Debug)]
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Edge values of `char`: the ends of each UTF-8 length and of the surrogate gap.
pub const CHAR_EDGES: [u32; 10] = [0, 0x7F, 0x80, 0x7FF, 0x800, 0xD7FF, 0xE000, 0xFFFF, 0x1_0000, 0x10_FFFF];

/// Edge values of `f32` (bits): ±0, ±1, the least subnormal, the least normal, ±max, ±inf, NaN.
pub const F32_EDGES: [u32; 11] = [
    0x0000_0000,
    0x8000_0000,
    0x3F80_0000,
    0xBF80_0000,
    0x0000_0001,
    0x0080_0000,
    0x7F7F_FFFF,
    0xFF7F_FFFF,
    0x7F80_0000,
    0xFF80_0000,
    0x7FC0_0000,
];

/// Edge values of `f64` (bits), as [`F32_EDGES`].
pub const F64_EDGES: [u64; 11] = [
    0x0000_0000_0000_0000,
    0x8000_0000_0000_0000,
    0x3FF0_0000_0000_0000,
    0xBFF0_0000_0000_0000,
    0x0000_0000_0000_0001,
    0x0010_0000_0000_0000,
    0x7FEF_FFFF_FFFF_FFFF,
    0xFFEF_FFFF_FFFF_FFFF,
    0x7FF0_0000_0000_0000,
    0xFFF0_0000_0000_0000,
    0x7FF8_0000_0000_0000,
];

/// The edge values of the int range `lo..=hi`, ascending without duplicates: `lo`, `lo + 1`,
/// `hi - 1`, `hi`, `0`, `1`, `-1`, and `2^k - 1`, `2^k`, `2^k + 1` and their negations for
/// `k` in 1..=64, those inside the range. Empty iff `lo > hi`.
pub fn int_edges(lo: i128, hi: i128) -> Vec<i128> {
    if lo > hi {
        return Vec::new();
    }
    let mut c = vec![lo, lo + 1, hi - 1, hi, 0, 1, -1];
    for k in 1..=64 {
        let p = 1i128 << k;
        c.extend([p - 1, p, p + 1, -p - 1, -p, -p + 1]);
    }
    c.retain(|v| lo <= *v && *v <= hi);
    c.sort_unstable();
    c.dedup();
    c
}

impl Gen {
    /// How many edge inputs this generator contributes before it repeats (at least 1).
    pub fn period(&self) -> usize {
        match self {
            Gen::Int { lo, hi, .. } => int_edges(*lo, *hi).len().max(1),
            Gen::Bool => 2,
            Gen::Char => CHAR_EDGES.len(),
            Gen::F32 => F32_EDGES.len(),
            Gen::F64 => F64_EDGES.len(),
            Gen::Array(e, n) => {
                if *n == 0 {
                    1
                } else {
                    e.period()
                }
            }
            Gen::Tuple(gs) => gs.iter().map(Gen::period).max().unwrap_or(1),
            Gen::Struct { fields, .. } => fields.iter().map(|(_, g)| g.period()).max().unwrap_or(1),
            Gen::Slice(e, lo, hi) => int_edges(*lo, *hi).len().max(e.period()),
        }
    }

    /// The positions of this parameter in the edge phase: a struct one per field (nested structs
    /// flattened), anything else one.
    pub fn positions(&self) -> usize {
        match self {
            Gen::Struct { fields, .. } => fields.iter().map(|(_, g)| g.positions()).sum(),
            _ => 1,
        }
    }

    /// The edge-phase value from flattened position `*pos` on (advanced past this value): on the
    /// anti-diagonal (`anti`) an odd position takes edge `period - 1 - k mod period` of its own
    /// generator, every other position edge `k`. Inside an array, tuple, or slice a struct is one
    /// value of [`Gen::edge`].
    fn edge_at(&self, k: usize, anti: bool, pos: &mut usize) -> Value {
        if let Gen::Struct { fields, .. } = self {
            return Value::List(fields.iter().map(|(_, g)| g.edge_at(k, anti, pos)).collect());
        }
        let j = *pos;
        *pos += 1;
        let per = self.period();
        self.edge(if anti && j % 2 == 1 { per - 1 - k % per } else { k })
    }

    /// Whether the generator has no value (an empty int range or length range).
    pub fn is_empty(&self) -> bool {
        match self {
            Gen::Int { lo, hi, .. } => lo > hi,
            Gen::Array(e, n) => *n > 0 && e.is_empty(),
            Gen::Tuple(gs) => gs.iter().any(Gen::is_empty),
            Gen::Struct { fields, .. } => fields.iter().any(|(_, g)| g.is_empty()),
            Gen::Slice(e, lo, hi) => lo > hi || (*lo > 0 && e.is_empty()),
            _ => false,
        }
    }

    /// The `k`-th edge value (cyclic): element `j` of an array or slice is the element
    /// generator's edge `k + j`, every tuple or struct field its own edge `k`, a slice's length the
    /// `k`-th edge of its length range.
    pub fn edge(&self, k: usize) -> Value {
        match self {
            Gen::Int { lo, hi, .. } => {
                let e = int_edges(*lo, *hi);
                Value::Int(e[k % e.len()])
            }
            Gen::Bool => Value::Bool(k % 2 == 1),
            Gen::Char => Value::Char(CHAR_EDGES[k % CHAR_EDGES.len()]),
            Gen::F32 => Value::F32(F32_EDGES[k % F32_EDGES.len()]),
            Gen::F64 => Value::F64(F64_EDGES[k % F64_EDGES.len()]),
            Gen::Array(e, n) => Value::List((0..*n as usize).map(|j| e.edge(k + j)).collect()),
            Gen::Tuple(gs) => Value::List(gs.iter().map(|g| g.edge(k)).collect()),
            Gen::Struct { fields, .. } => Value::List(fields.iter().map(|(_, g)| g.edge(k)).collect()),
            Gen::Slice(e, lo, hi) => {
                let lens = int_edges(*lo, *hi);
                let n = lens[k % lens.len()] as usize;
                Value::List((0..n).map(|j| e.edge(k + j)).collect())
            }
        }
    }

    /// A sampled value: an int is `lo + next() mod (hi - lo + 1)`, a `bool` the low bit, a
    /// `char` `r = next() mod 0x10F800` (plus 0x800 from 0xD800 on, skipping the surrogates), a
    /// float the low bits of `next()`; arrays, tuples, and struct fields in order; a slice its
    /// length (as an int of its length range), then its elements.
    pub fn sample(&self, r: &mut SplitMix64) -> Value {
        match self {
            Gen::Int { lo, hi, .. } => Value::Int(sample_int(*lo, *hi, r)),
            Gen::Bool => Value::Bool(r.next() & 1 == 1),
            Gen::Char => {
                let c = (r.next() % 0x10_F800) as u32;
                Value::Char(if c >= 0xD800 { c + 0x800 } else { c })
            }
            Gen::F32 => Value::F32(r.next() as u32),
            Gen::F64 => Value::F64(r.next()),
            Gen::Array(e, n) => Value::List((0..*n).map(|_| e.sample(r)).collect()),
            Gen::Tuple(gs) => Value::List(gs.iter().map(|g| g.sample(r)).collect()),
            Gen::Struct { fields, .. } => Value::List(fields.iter().map(|(_, g)| g.sample(r)).collect()),
            Gen::Slice(e, lo, hi) => {
                let n = sample_int(*lo, *hi, r);
                Value::List((0..n).map(|_| e.sample(r)).collect())
            }
        }
    }

    /// The scalars of the largest value.
    pub fn scalars(&self) -> u64 {
        match self {
            Gen::Array(e, n) => n.saturating_mul(e.scalars()),
            Gen::Tuple(gs) => gs.iter().fold(0u64, |a, g| a.saturating_add(g.scalars())),
            Gen::Struct { fields, .. } => fields.iter().fold(0u64, |a, (_, g)| a.saturating_add(g.scalars())),
            Gen::Slice(e, _, hi) => u64::try_from(*hi).unwrap_or(u64::MAX).saturating_mul(e.scalars()),
            _ => 1,
        }
    }
}

fn sample_int(lo: i128, hi: i128, r: &mut SplitMix64) -> i128 {
    let span = (hi - lo + 1) as u128;
    lo + (u128::from(r.next()) % span) as i128
}

impl Domain {
    /// Whether no input exists (some parameter's range is empty).
    pub fn is_empty(&self) -> bool {
        self.params.iter().any(|p| p.values.is_empty())
    }

    /// One input as `name = literal` per parameter, comma-separated.
    pub fn show(&self, input: &[Value]) -> String {
        let parts: Vec<String> = self
            .params
            .iter()
            .zip(input)
            .map(|(p, v)| format!("{} = {}", p.name, literal(&p.values, v)))
            .collect();
        parts.join(", ")
    }
}

/// Every input, in order: the edge phase, then [`SAMPLES`] sampled inputs from one
/// [`SplitMix64`] at [`SEED`] (parameters in order); an input equal to an earlier one is dropped.
/// Empty for an empty domain. The edge phase sees the parameters as positions, a struct's fields
/// flattened into one position each ([`Gen::positions`]). With `n` the largest period, it is the
/// diagonal (input `k` < `n` takes edge `k` at every position), then, with two or more
/// positions, the anti-diagonal (input `k` < `n` takes edge `k` at even positions and edge
/// `period - 1 - k mod period` at odd ones), so low edges meet high ones.
pub fn inputs(d: &Domain) -> Vec<Vec<Value>> {
    if d.is_empty() || d.params.is_empty() {
        return Vec::new();
    }
    let mut out: Vec<Vec<Value>> = Vec::new();
    let mut seen: HashSet<Vec<Value>> = HashSet::new();
    let n = d.params.iter().map(|p| p.values.period()).max().unwrap_or(1);
    let positions: usize = d.params.iter().map(|p| p.values.positions()).sum();
    let phases = if positions > 1 { 2 } else { 1 };
    for phase in 0..phases {
        for k in 0..n {
            let mut pos = 0usize;
            let i: Vec<Value> = d.params.iter().map(|p| p.values.edge_at(k, phase == 1, &mut pos)).collect();
            if seen.insert(i.clone()) {
                out.push(i);
            }
        }
    }
    let mut r = SplitMix64(SEED);
    for _ in 0..SAMPLES {
        let i: Vec<Value> = d.params.iter().map(|p| p.values.sample(&mut r)).collect();
        if seen.insert(i.clone()) {
            out.push(i);
        }
    }
    out
}

/// `v` as a Stave expression of `g`'s type: suffixed int and float literals, `'c'` with escapes,
/// `[..]`, `(..)`, `&[..]`, `S.{ f: .. }`; a float with no literal (inf, NaN) as `mem.f32_from_bits(0x..)`
/// (`std.mem`), which also keeps a NaN's payload.
pub fn literal(g: &Gen, v: &Value) -> String {
    match (g, v) {
        (Gen::Int { ty, .. }, Value::Int(i)) => format!("{i}{ty}"),
        (Gen::Bool, Value::Bool(b)) => b.to_string(),
        (Gen::Char, Value::Char(c)) => char_literal(*c),
        (Gen::F32, Value::F32(b)) => {
            let x = f32::from_bits(*b);
            if x.is_finite() { format!("{x:?}f32") } else { format!("mem.f32_from_bits(0x{b:08x})") }
        }
        (Gen::F64, Value::F64(b)) => {
            let x = f64::from_bits(*b);
            if x.is_finite() { format!("{x:?}f64") } else { format!("mem.f64_from_bits(0x{b:016x})") }
        }
        (Gen::Array(e, _), Value::List(vs)) => format!("[{}]", list(e, vs)),
        (Gen::Slice(e, ..), Value::List(vs)) => format!("&[{}]", list(e, vs)),
        (Gen::Tuple(gs), Value::List(vs)) => {
            let parts: Vec<String> = gs.iter().zip(vs).map(|(g, v)| literal(g, v)).collect();
            if parts.len() == 1 { format!("({},)", parts[0]) } else { format!("({})", parts.join(", ")) }
        }
        (Gen::Struct { path, fields }, Value::List(vs)) => {
            let parts: Vec<String> =
                fields.iter().zip(vs).map(|((f, g), v)| format!("{f}: {}", literal(g, v))).collect();
            if parts.is_empty() {
                format!("{path}.{{ }}")
            } else {
                format!("{path}.{{ {} }}", parts.join(", "))
            }
        }
        _ => "?".to_string(),
    }
}

fn list(e: &Gen, vs: &[Value]) -> String {
    let parts: Vec<String> = vs.iter().map(|v| literal(e, v)).collect();
    parts.join(", ")
}

fn char_literal(c: u32) -> String {
    let body = match c {
        0x27 => "\\'".to_string(),
        0x5C => "\\\\".to_string(),
        0x0A => "\\n".to_string(),
        0x09 => "\\t".to_string(),
        0x0D => "\\r".to_string(),
        0 => "\\0".to_string(),
        0x20..=0x7E => char::from_u32(c).map_or(String::new(), |x| x.to_string()),
        _ => format!("\\u{{{c:x}}}"),
    };
    format!("'{body}'")
}

/// The name (literal suffix) of an int type.
fn int_name(signed: bool, bits: IntBits) -> &'static str {
    match (signed, bits) {
        (true, IntBits::B8) => "i8",
        (true, IntBits::B16) => "i16",
        (true, IntBits::B32) => "i32",
        (true, IntBits::B64) => "i64",
        (true, IntBits::Size) => "isize",
        (false, IntBits::B8) => "u8",
        (false, IntBits::B16) => "u16",
        (false, IntBits::B32) => "u32",
        (false, IntBits::B64) => "u64",
        (false, IntBits::Size) => "usize",
    }
}

/// What a range conjunct constrains: an int parameter's value, an int field of a struct parameter
/// (`path`: field indices, outermost first), or a slice parameter's length.
#[derive(Clone)]
struct Subject {
    param: usize,
    path: Vec<usize>,
    len: bool,
}

/// Why the runner cannot generate a type.
enum NoGen {
    /// The type itself (E0214 without a case).
    Type,
    /// The struct field at this path (`inner.label`), of this type (as the test's module writes
    /// it), is not generatable.
    Field(String, String),
    /// `name` (a field path, or a struct's name) is private to module `module`.
    Private { name: String, module: String, field: bool },
}

impl NoGen {
    /// This reason seen from the struct field `f` that holds it (`Type` is resolved by the
    /// caller, which knows the field's type).
    fn under(self, f: &str) -> NoGen {
        match self {
            NoGen::Type => NoGen::Type,
            NoGen::Field(p, t) => NoGen::Field(format!("{f}.{p}"), t),
            NoGen::Private { name, module, field: true } => {
                NoGen::Private { name: format!("{f}.{name}"), module, field: true }
            }
            p @ NoGen::Private { .. } => p,
        }
    }
}

/// ADT names as a module writes them: its own items bare, another module's through the import
/// that binds it (`geo.Point`), else by the module's full path.
struct ModuleNames<'a> {
    prog: &'a crate::Program,
    module: crate::ModuleId,
    inner: crate::sigs::NameCx<'a>,
}

impl Names for ModuleNames<'_> {
    fn adt_name(&self, d: crate::DeclId) -> String {
        let Some(x) = self.prog.decl(d) else { return "_".to_string() };
        let name = self.prog.strings.get(x.name);
        if x.module == self.module {
            return name.to_string();
        }
        let via = self
            .prog
            .module(self.module)
            .and_then(|me| me.imports.iter().find(|i| i.target == x.module))
            .map(|i| self.prog.strings.get(i.bound).to_string());
        match via.or_else(|| self.prog.module(x.module).map(|m| m.path_text.clone())) {
            Some(m) => format!("{m}.{name}"),
            None => name.to_string(),
        }
    }

    fn param_name(&self, i: u32) -> String {
        self.inner.param_name(i)
    }
}

impl Ck<'_> {
    /// For a `test fn` with parameters (fn node `n`, after its clauses are checked): E0214 for
    /// what the runner cannot generate, else the domain in [`Ck::domain`].
    pub(crate) fn test_domain(&mut self, sig: &FnSig, n: u32) {
        let icx = self.icx;
        if !self.prog.decl(self.decl).is_some_and(|d| d.flags.has(DeclFlags::TEST)) {
            return;
        }
        let Some(ps) = icx.child(n, NodeKind::Params) else { return };
        let nodes: Vec<u32> = icx.kids_vec(ps).iter().copied().collect();
        if nodes.is_empty() {
            return;
        }
        let mut ok = true;
        if sig.ret != ids::UNIT && sig.ret != ids::ERROR && !self.tys.has_error(sig.ret) {
            let at = sig.ret_span.unwrap_or_else(|| icx.tok_span(icx.a(n)));
            self.report(mk("E0214", Some("ret"), &[], at, &[]));
            ok = false;
        }
        let mut params: Vec<Param> = Vec::with_capacity(nodes.len());
        for (i, &p) in nodes.iter().enumerate() {
            let t = sig.params.get(i).copied().unwrap_or(ids::ERROR);
            let at = sig.param_spans.get(i).copied().unwrap_or_else(|| icx.tok_span(icx.a(p)));
            if t == ids::ERROR || self.tys.has_error(t) {
                ok = false;
                continue;
            }
            // the size first, on the type (memoised): a generator holds one node per scalar
            if crate::consteval::cells(self, t).is_some_and(|c| c > MAX_SCALARS) {
                let tn = self.show(t);
                self.report(mk("E0214", Some("size"), &[("type", &tn)], at, &[]));
                ok = false;
                continue;
            }
            match self.param_gen(t, true, 0) {
                Ok(values) => params.push(Param { name: icx.tok_str(icx.a(p)).to_string(), values }),
                Err(why) => {
                    let tn = self.show(t);
                    let d = match &why {
                        NoGen::Type => mk("E0214", None, &[("type", &tn)], at, &[]),
                        NoGen::Field(f, ft) => {
                            mk("E0214", Some("field"), &[("type", &tn), ("name", f), ("inner", ft)], at, &[])
                        }
                        NoGen::Private { name, module, .. } => mk(
                            "E0214",
                            Some("private"),
                            &[("type", &tn), ("name", name), ("module", module)],
                            at,
                            &[],
                        ),
                    };
                    self.report(d);
                    ok = false;
                }
            }
        }
        if !ok {
            return;
        }
        let clauses: Vec<u32> = self
            .contracts
            .as_ref()
            .map(|c| c.requires.iter().map(|cl| cl.expr).collect())
            .unwrap_or_default();
        let mut conj = Vec::new();
        for e in clauses {
            self.conjuncts(e, &mut conj);
        }
        let mut filters = 0u32;
        for e in conj {
            if !self.range_conjunct(e, &nodes, &mut params) {
                filters += 1;
            }
        }
        for (i, p) in params.iter().enumerate() {
            let at = sig.param_spans.get(i).copied().unwrap_or_else(|| icx.tok_span(icx.a(nodes[i])));
            if let Gen::Slice(_, _, hi) = p.values
                && hi == i128::MAX
            {
                self.report(mk("E0214", Some("unbounded"), &[("name", &p.name)], at, &[]));
                ok = false;
            } else if p.values.scalars() > MAX_SCALARS {
                let tn = self.show(sig.params[i]);
                self.report(mk("E0214", Some("size"), &[("type", &tn)], at, &[]));
                ok = false;
            }
        }
        if ok {
            self.domain = Some(Box::new(Domain { params, filters }));
        }
    }

    /// The generator of type `t` (`top`: a parameter's own type, where `&[T]` is allowed; `depth`:
    /// the nesting so far, bounded so a pathological type cannot exhaust the stack).
    fn param_gen(&mut self, t: TypeId, top: bool, depth: u32) -> Result<Gen, NoGen> {
        if depth > 64 {
            return Err(NoGen::Type);
        }
        match self.tys.kind(t).clone() {
            TypeKind::Int { signed, bits } => {
                let (lo, hi) =
                    crate::literals::int_range_kind(&TypeKind::Int { signed, bits }).ok_or(NoGen::Type)?;
                Ok(Gen::Int { ty: int_name(signed, bits), lo, hi })
            }
            TypeKind::Bool => Ok(Gen::Bool),
            TypeKind::Char => Ok(Gen::Char),
            TypeKind::Float(FloatBits::F32) => Ok(Gen::F32),
            TypeKind::Float(FloatBits::F64) => Ok(Gen::F64),
            TypeKind::Array { elem, len: ConstArg::Value(n) } => {
                Ok(Gen::Array(Box::new(self.param_gen(elem, false, depth + 1)?), n))
            }
            TypeKind::Tuple(ts) => {
                let mut gs = Vec::with_capacity(ts.len());
                for &x in ts.iter() {
                    gs.push(self.param_gen(x, false, depth + 1)?);
                }
                Ok(Gen::Tuple(gs))
            }
            TypeKind::Adt { def, args } => self.struct_gen(def, &args, depth),
            TypeKind::Ref { mutable: false, inner } if top => match self.tys.kind(inner).clone() {
                TypeKind::Slice { elem, align: 0 } => {
                    Ok(Gen::Slice(Box::new(self.param_gen(elem, false, depth + 1)?), 0, i128::MAX))
                }
                _ => Err(NoGen::Type),
            },
            _ => Err(NoGen::Type),
        }
    }

    /// The generator of struct `def` with arguments `args`: its fields, each generatable and
    /// visible from the test's module, as is the struct itself. Enums and views are not. A struct
    /// of another package that is not generatable is just `Type`: its insides (`vec.Vec`'s
    /// private pointer) are not the test author's to change, so the report names the field that
    /// holds it, with its type.
    fn struct_gen(&mut self, def: crate::DeclId, args: &[GArg], depth: u32) -> Result<Gen, NoGen> {
        let (prog, sigs) = (self.prog, self.sigs);
        let adt = sigs.adt(def).ok_or(NoGen::Type)?;
        let v = adt.variants.first().ok_or(NoGen::Type)?;
        if adt.is_enum || adt.is_view {
            return Err(NoGen::Type);
        }
        let me = self.icx.res_module();
        let module_text = |m| prog.module(m).map_or(String::new(), |x| x.path_text.clone());
        let d = prog.decl(def).ok_or(NoGen::Type)?;
        let package = |m| prog.module(m).map(|x| x.package);
        let foreign = package(d.module) != package(me);
        let fail = |why: NoGen| Err(if foreign { NoGen::Type } else { why });
        if !d.is_pub() && d.module != me {
            let name = prog.strings.get(d.name).to_string();
            return fail(NoGen::Private { name, module: module_text(d.module), field: false });
        }
        let names = ModuleNames { prog, module: me, inner: self.names() };
        let mut fields = Vec::with_capacity(v.fields.len());
        for f in &v.fields {
            let fname = prog.strings.get(f.name).to_string();
            if let Some(fd) = prog.decl(f.decl)
                && !fd.is_pub()
                && fd.module != me
            {
                return fail(NoGen::Private { name: fname, module: module_text(fd.module), field: true });
            }
            let ft = self.tys.subst(f.ty, args, None);
            match self.param_gen(ft, false, depth + 1) {
                Ok(g) => fields.push((fname, g)),
                Err(NoGen::Type) => {
                    let t = crate::types::display(&self.tys, &names, ft);
                    return fail(NoGen::Field(fname, t));
                }
                Err(why) => return fail(why.under(&fname)),
            }
        }
        let mut path = names.adt_name(def);
        if !args.is_empty() {
            let parts: Vec<String> = args
                .iter()
                .map(|a| match a {
                    GArg::Type(x) => crate::types::display(&self.tys, &names, *x),
                    GArg::Const(ConstArg::Value(n)) => n.to_string(),
                    GArg::Const(_) => "_".to_string(),
                })
                .collect();
            path = format!("{path}::<{}>", parts.join(", "));
        }
        Ok(Gen::Struct { path, fields })
    }

    /// The conjuncts of clause expression `e`: split at `&&`, through parentheses.
    fn conjuncts(&self, e: u32, out: &mut Vec<u32>) {
        let icx = self.icx;
        match icx.kind(e) {
            Some(NodeKind::AndExpr) => {
                for &k in icx.kids_vec(e).iter() {
                    self.conjuncts(k, out);
                }
            }
            _ => match self.paren_inner(e) {
                Some(x) => self.conjuncts(x, out),
                None => out.push(e),
            },
        }
    }

    /// The expression inside `( e )` when node `n` is just that.
    fn paren_inner(&self, n: u32) -> Option<u32> {
        let icx = self.icx;
        if icx.kind(n) != Some(NodeKind::Primary) || !icx.is(icx.a(n), Term::LParen) {
            return None;
        }
        let c = icx.only_child(n)?;
        (icx.is_expr(c) && icx.end_tok(c) + 1 == icx.close(icx.a(n))).then_some(c)
    }

    /// Narrows a parameter's range by conjunct `e` if it is a range comparison; `false`: a
    /// filter.
    fn range_conjunct(&mut self, e: u32, nodes: &[u32], params: &mut [Param]) -> bool {
        let icx = self.icx;
        if icx.kind(e) != Some(NodeKind::CmpExpr) {
            return false;
        }
        let kids = icx.kids_vec(e);
        let (Some(&l), Some(&opn), Some(&r)) = (kids.first(), kids.get(1), kids.get(2)) else { return false };
        let op = icx.tk(icx.a(opn));
        let (s, k, flip) = if let Some(s) = self.subject(l, nodes, params)
            && let Some(k) = self.constant(r)
        {
            (s, k, false)
        } else if let Some(s) = self.subject(r, nodes, params)
            && let Some(k) = self.constant(l)
        {
            (s, k, true)
        } else {
            return false;
        };
        let (lt, le, gt, ge) = (Term::Lt as u8, Term::LtEq as u8, Term::Gt as u8, Term::GtEq as u8);
        let op = match (flip, op) {
            (true, o) if o == lt => gt,
            (true, o) if o == le => ge,
            (true, o) if o == gt => lt,
            (true, o) if o == ge => le,
            (_, o) => o,
        };
        let (lo, hi) = if op == lt {
            (i128::MIN, k - 1)
        } else if op == le {
            (i128::MIN, k)
        } else if op == gt {
            (k + 1, i128::MAX)
        } else if op == ge {
            (k, i128::MAX)
        } else if op == Term::EqEq as u8 {
            (k, k)
        } else {
            return false;
        };
        let mut g = &mut params[s.param].values;
        for &i in &s.path {
            match g {
                Gen::Struct { fields, .. } => g = &mut fields[i].1,
                _ => return false,
            }
        }
        match g {
            Gen::Int { lo: a, hi: b, .. } if !s.len => {
                *a = (*a).max(lo);
                *b = (*b).min(hi);
            }
            Gen::Slice(_, a, b) if s.len => {
                *a = (*a).max(lo);
                *b = (*b).min(hi);
            }
            _ => return false,
        }
        true
    }

    /// Node `n` as a range subject: an int parameter, an int field `p.a.b` of a struct parameter,
    /// or `p.len()` of a slice parameter.
    fn subject(&self, n: u32, nodes: &[u32], params: &[Param]) -> Option<Subject> {
        let icx = self.icx;
        if let Some(x) = self.paren_inner(n) {
            return self.subject(x, nodes, params);
        }
        let a = icx.a(n);
        let param_at = |t: u32| -> Option<usize> {
            let u = icx.res.uses.iter().find(|u| u.token == t)?;
            let Res::Local(b) = u.res else { return None };
            nodes.iter().position(|&p| p == b.node)
        };
        match icx.kind(n)? {
            NodeKind::Primary if icx.is(a, Term::Ident) && icx.end_tok(n) == a => {
                let i = param_at(a)?;
                matches!(params[i].values, Gen::Int { .. }).then_some(Subject {
                    param: i,
                    path: Vec::new(),
                    len: false,
                })
            }
            // `p.a.b`: with `p` a struct parameter, each step after it is a field of its struct.
            NodeKind::Primary | NodeKind::Postfix
                if icx.is(a, Term::Ident)
                    && icx.end_tok(n) > a
                    && (a..=icx.end_tok(n)).all(|t| {
                        if (t - a).is_multiple_of(2) { icx.is(t, Term::Ident) } else { icx.is(t, Term::Dot) }
                    }) =>
            {
                let i = param_at(a)?;
                let mut g = &params[i].values;
                let mut path = Vec::new();
                for t in (a + 2..=icx.end_tok(n)).step_by(2) {
                    let Gen::Struct { fields, .. } = g else { return None };
                    let f = fields.iter().position(|(name, _)| name == icx.tok_str(t))?;
                    path.push(f);
                    g = &fields[f].1;
                }
                matches!(g, Gen::Int { .. }).then_some(Subject { param: i, path, len: false })
            }
            NodeKind::Postfix => {
                let is_len = icx.is(a, Term::Ident)
                    && icx.is(a + 1, Term::Dot)
                    && icx.is(a + 2, Term::Ident)
                    && icx.tok_str(a + 2) == "len"
                    && icx.is(a + 3, Term::LParen)
                    && icx.is(a + 4, Term::RParen)
                    && icx.end_tok(n) == a + 4;
                let builtin = self.targets.iter().any(|(t, x)| {
                    (*t == a + 2 || *t == a + 3) && matches!(x, Target::Builtin { name: "len", .. })
                });
                if !is_len || !builtin {
                    return None;
                }
                let i = param_at(a)?;
                matches!(params[i].values, Gen::Slice(..)).then_some(Subject {
                    param: i,
                    path: Vec::new(),
                    len: true,
                })
            }
            _ => None,
        }
    }

    /// Node `n` as an int constant: a literal, a negated constant, or an int `const` item.
    fn constant(&mut self, n: u32) -> Option<i128> {
        let icx = self.icx;
        if let Some(x) = self.paren_inner(n) {
            return self.constant(x);
        }
        let a = icx.a(n);
        match icx.kind(n)? {
            NodeKind::Primary if icx.is(a, Term::IntLit) && icx.end_tok(n) == a => {
                let (v, _) = crate::literals::parse_int(icx.tok_str(a))?;
                i128::try_from(v).ok()
            }
            NodeKind::Unary if icx.is(a, Term::Minus) => {
                let c = icx.only_child(n)?;
                self.constant(c)?.checked_neg()
            }
            NodeKind::Primary | NodeKind::Postfix => {
                let end = icx.end_tok(n);
                if !(a..=end).all(|t| {
                    if (t - a).is_multiple_of(2) { icx.is(t, Term::Ident) } else { icx.is(t, Term::Dot) }
                }) {
                    return None;
                }
                let decl = self.targets.iter().find_map(|(t, x)| match x {
                    Target::Global { decl } if *t == end => Some(*decl),
                    _ => None,
                })?;
                match crate::consteval::CEnv::const_value(self, decl)? {
                    (_, crate::consteval::Val::Int(i)) => Some(i),
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int(ty: &'static str, lo: i128, hi: i128) -> Gen {
        Gen::Int { ty, lo, hi }
    }

    #[test]
    fn splitmix_reference_values() {
        // The first outputs of SplitMix64 from state 0 (the reference implementation's).
        let mut r = SplitMix64(0);
        assert_eq!(r.next(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(r.next(), 0x6E78_9E6A_A1B9_65F4);
        assert_eq!(r.next(), 0x06C4_5D18_8009_454F);
    }

    #[test]
    fn edges_of_int_ranges() {
        assert_eq!(
            int_edges(0, 255),
            [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 254, 255]
        );
        assert_eq!(int_edges(-3, 3), [-3, -2, -1, 0, 1, 2, 3]);
        assert_eq!(int_edges(10, 12), [10, 11, 12]);
        assert_eq!(int_edges(5, 5), [5]);
        assert!(int_edges(6, 5).is_empty());
        let e = int_edges(i64::MIN as i128, i64::MAX as i128);
        assert_eq!((e[0], e[e.len() - 1]), (i64::MIN as i128, i64::MAX as i128));
        let u = int_edges(0, u64::MAX as i128);
        assert_eq!(u[u.len() - 1], u64::MAX as i128);
    }

    #[test]
    fn inputs_are_deterministic_and_distinct() {
        let d = Domain {
            params: vec![
                Param {
                    name: "xs".into(),
                    values: Gen::Slice(Box::new(int("u32", 0, u32::MAX as i128)), 0, 17),
                },
                Param { name: "k".into(), values: int("u32", 0, 9) },
            ],
            filters: 0,
        };
        let a = inputs(&d);
        assert_eq!(a, inputs(&d));
        let set: HashSet<&Vec<Value>> = a.iter().collect();
        assert_eq!(set.len(), a.len());
        for i in &a {
            let (Value::List(xs), Value::Int(k)) = (&i[0], &i[1]) else { panic!("{i:?}") };
            assert!(xs.len() <= 17 && (0..=9).contains(k));
        }
        // Edge phase first: lengths 0 and 17 and k 0 and 9 all occur.
        let lens: HashSet<usize> =
            a.iter().map(|i| if let Value::List(xs) = &i[0] { xs.len() } else { 99 }).collect();
        assert!(lens.contains(&0) && lens.contains(&17));
    }

    #[test]
    fn low_edges_meet_high_ones() {
        let d = Domain {
            params: vec![
                Param { name: "a".into(), values: int("u8", 0, 255) },
                Param { name: "b".into(), values: int("u8", 0, 255) },
            ],
            filters: 0,
        };
        let a = inputs(&d);
        let n = int_edges(0, 255).len();
        assert_eq!(a[0], [Value::Int(0), Value::Int(0)]);
        assert_eq!(a[n], [Value::Int(0), Value::Int(255)]);
        // The middle of the anti-diagonal repeats the diagonal and is dropped.
        assert_eq!(a[2 * n - 2], [Value::Int(255), Value::Int(0)]);
    }

    #[test]
    fn small_domains_are_not_padded() {
        let d = Domain { params: vec![Param { name: "b".into(), values: Gen::Bool }], filters: 0 };
        assert_eq!(inputs(&d), [vec![Value::Bool(false)], vec![Value::Bool(true)]]);
        let empty = Domain { params: vec![Param { name: "x".into(), values: int("u8", 3, 2) }], filters: 0 };
        assert!(inputs(&empty).is_empty());
    }

    #[test]
    fn literals() {
        assert_eq!(literal(&int("i8", -128, 127), &Value::Int(-128)), "-128i8");
        assert_eq!(literal(&Gen::Char, &Value::Char(0x27)), "'\\''");
        assert_eq!(literal(&Gen::Char, &Value::Char(0x10FFFF)), "'\\u{10ffff}'");
        assert_eq!(literal(&Gen::F32, &Value::F32(0x3F80_0000)), "1.0f32");
        assert_eq!(literal(&Gen::F32, &Value::F32(0x8000_0000)), "-0.0f32");
        assert_eq!(
            literal(&Gen::F64, &Value::F64(0x7FF8_0000_0000_0000)),
            "mem.f64_from_bits(0x7ff8000000000000)"
        );
        let t = Gen::Tuple(vec![Gen::Bool]);
        assert_eq!(literal(&t, &Value::List(vec![Value::Bool(true)])), "(true,)");
        let s = Gen::Slice(Box::new(int("u8", 0, 255)), 0, 4);
        assert_eq!(literal(&s, &Value::List(vec![Value::Int(1), Value::Int(2)])), "&[1u8, 2u8]");
        let a = Gen::Array(Box::new(Gen::Bool), 0);
        assert_eq!(literal(&a, &Value::List(Vec::new())), "[]");
        let p = Gen::Struct {
            path: "geo.Pair::<u8>".into(),
            fields: vec![("a".into(), int("u8", 0, 255)), ("b".into(), Gen::Bool)],
        };
        assert_eq!(
            literal(&p, &Value::List(vec![Value::Int(7), Value::Bool(false)])),
            "geo.Pair::<u8>.{ a: 7u8, b: false }"
        );
        let e = Gen::Struct { path: "E".into(), fields: Vec::new() };
        assert_eq!(literal(&e, &Value::List(Vec::new())), "E.{ }");
    }

    #[test]
    fn struct_fields_are_edge_positions() {
        // One struct parameter of two fields meets low and high edges as two parameters do.
        let s = Gen::Struct {
            path: "S".into(),
            fields: vec![("a".into(), int("u8", 0, 255)), ("b".into(), int("u8", 0, 255))],
        };
        let d = Domain { params: vec![Param { name: "s".into(), values: s.clone() }], filters: 0 };
        let a = inputs(&d);
        let n = int_edges(0, 255).len();
        assert_eq!(a[0], [Value::List(vec![Value::Int(0), Value::Int(0)])]);
        assert_eq!(a[n], [Value::List(vec![Value::Int(0), Value::Int(255)])]);
        // Nested: a struct field is split into its fields; positions run on into the next parameter.
        let outer = Gen::Struct { path: "O".into(), fields: vec![("s".into(), s), ("c".into(), Gen::Bool)] };
        assert_eq!(outer.positions(), 3);
        let d = Domain {
            params: vec![
                Param { name: "o".into(), values: outer },
                Param { name: "k".into(), values: int("u8", 0, 255) },
            ],
            filters: 0,
        };
        let a = inputs(&d);
        // Anti-diagonal, k = 0: positions 1 (`s.b`) and 3 (`k`) take their last edge.
        let first_anti = [
            Value::List(vec![Value::List(vec![Value::Int(0), Value::Int(255)]), Value::Bool(false)]),
            Value::Int(255),
        ];
        assert_eq!(a[n], first_anti);
        // An array of structs is one position: its struct elements are not split.
        let arr = Gen::Array(Box::new(Gen::Struct { path: "S".into(), fields: Vec::new() }), 2);
        assert_eq!(arr.positions(), 1);
    }
}
