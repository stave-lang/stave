//! Built-ins: methods of the built-in types (`mul_wide`, `mul_hi`, int bit counts,
//! raw-pointer `read`/`write`, memory hints (prefetches, `write_stream`, scalar `store_stream`),
//! `len`, `bool` and mask `select`, SIMD vector and mask operations), the type-relative fns of
//! vector and mask types (`f32x4.splat`, `u8x16.load`), and the `extern "stave-intrinsic"` table
//! of contracts/intrinsics.md (E0805; E0704 when called on another arch). `size_of` / `align_of`
//! / `fence_stream` are typed by the body checker.
//!
//! `mul_wide` exists on `i32`, `i64`, `u32` and `u64` only, `mul_hi` on the fixed-width ints:
//! on any other type the lookup fails and the checker reports E0208.

use stave_diag::Diagnostic;
use stave_syntax::NodeKind;

use crate::Program;
use crate::decls::{DeclFlags, DeclKind};
use crate::diag::mk;
use crate::ids::DeclId;
use crate::itemcx::{Brackets, ItemCx};
use crate::package::PackageKind;
use crate::prim::Lanes;
use crate::sigs::{Abi, Sigs};
use crate::types::{self, ConstArg, GArg, Interner, TypeId, TypeKind, ids};

/// A built-in method: its name and how it takes the receiver.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BuiltinMethod {
    pub name: &'static str,
    /// The receiver is taken by `&` (`len`) instead of by value.
    pub by_ref: bool,
    /// Calls are unsafe operations (`p.read()`, `p.write(v)`).
    pub is_unsafe: bool,
}

const INT_METHODS: &[&str] = &["mul_wide", "mul_hi", "popcount", "clz", "ctz"];
const PTR_HINTS: &[&str] = &[
    "prefetch_read_l1",
    "prefetch_read_l2",
    "prefetch_read_l3",
    "prefetch_read_nta",
    "prefetch_write_l1",
    "prefetch_write_l2",
    "prefetch_write_l3",
    "prefetch_write_nta",
];
const VEC_METHODS: &[&str] = &[
    "sqrt",
    "abs",
    "copysign",
    "floor",
    "ceil",
    "round",
    "trunc",
    "fma",
    "min",
    "max",
    "min_num",
    "max_num",
    "min_x86",
    "max_x86",
    "abs_wrap",
    "andnot",
    "popcount",
    "clz",
    "reduce_add",
    "reduce_add_wrap",
    "reduce_min",
    "reduce_max",
    "reduce_and",
    "reduce_or",
    "reduce_add_ordered",
    "scan_add",
    "scan_add_wrap",
    "bitcast",
    "split",
    "widen_lo",
    "widen_hi",
    "narrow_wrap",
    "narrow_sat",
    "permute",
    "lookup",
    "compress",
    "expand",
    "store",
    "store_aligned",
    "store_masked",
    "scatter",
    "checked_add",
    "checked_sub",
    "checked_mul",
    "mul_hi",
    "div_const",
    "div_lanes",
    "store_stream",
];
const MASK_METHODS: &[&str] = &["select", "to_bits", "to_vec", "rewiden"];

/// A SIMD lane type: the fixed-width ints and the floats (streaming stores take only these).
fn is_lane<I: Interner + ?Sized>(cx: &I, t: TypeId) -> bool {
    match cx.kind(t) {
        TypeKind::Int { bits, .. } => *bits != types::IntBits::Size,
        TypeKind::Float(_) => true,
        _ => false,
    }
}

fn intern_name(list: &[&'static str], name: &str) -> Option<&'static str> {
    list.iter().copied().find(|&n| n == name)
}

/// The built-in method `name` of exactly type `t` (no auto-deref), if any.
pub fn builtin_method<I: Interner + ?Sized>(cx: &I, t: TypeId, name: &str) -> Option<BuiltinMethod> {
    match cx.kind(t) {
        TypeKind::Bool => {
            (name == "select").then_some(BuiltinMethod { name: "select", by_ref: false, is_unsafe: false })
        }
        TypeKind::Int { .. } | TypeKind::Float(_) if name == "store_stream" => {
            is_lane(cx, t).then_some(BuiltinMethod { name: "store_stream", by_ref: false, is_unsafe: false })
        }
        TypeKind::Int { bits, .. } => {
            let n = intern_name(INT_METHODS, name)?;
            if n == "mul_wide" && !matches!(bits, types::IntBits::B32 | types::IntBits::B64) {
                return None;
            }
            if n == "mul_hi" && *bits == types::IntBits::Size {
                return None;
            }
            Some(BuiltinMethod { name: n, by_ref: false, is_unsafe: false })
        }
        TypeKind::Ptr { mutable, inner } => {
            if name == "read" {
                return Some(BuiltinMethod { name: "read", by_ref: false, is_unsafe: true });
            }
            if name == "write" && *mutable {
                return Some(BuiltinMethod { name: "write", by_ref: false, is_unsafe: true });
            }
            if name == "write_stream" {
                let streamable = is_lane(cx, *inner) || matches!(cx.kind(*inner), TypeKind::Vector { .. });
                return (*mutable && streamable).then_some(BuiltinMethod {
                    name: "write_stream",
                    by_ref: false,
                    is_unsafe: true,
                });
            }
            let n = intern_name(PTR_HINTS, name)?;
            Some(BuiltinMethod { name: n, by_ref: false, is_unsafe: false })
        }
        TypeKind::Slice { .. } => {
            if name == "len" {
                return Some(BuiltinMethod { name: "len", by_ref: true, is_unsafe: false });
            }
            if name == "align_to" {
                return Some(BuiltinMethod { name: "align_to", by_ref: true, is_unsafe: false });
            }
            let n = intern_name(PTR_HINTS, name)?;
            Some(BuiltinMethod { name: n, by_ref: true, is_unsafe: false })
        }
        TypeKind::Array { .. } | TypeKind::Str => {
            if name == "len" {
                return Some(BuiltinMethod { name: "len", by_ref: true, is_unsafe: false });
            }
            None
        }
        TypeKind::Vector { .. } => {
            let n = intern_name(VEC_METHODS, name)?;
            Some(BuiltinMethod { name: n, by_ref: false, is_unsafe: false })
        }
        TypeKind::Mask { .. } => {
            let n = intern_name(MASK_METHODS, name)?;
            Some(BuiltinMethod { name: n, by_ref: false, is_unsafe: false })
        }
        _ => None,
    }
}

/// Expected argument shape of a built-in call.
#[derive(Clone, Copy, Debug)]
pub enum ArgTy {
    Is(TypeId),
    /// Any vector type (checked by its use).
    AnyVec,
    /// An operand of `c.select(a, b)`: typed by the body checker (`a` and `b` have one type,
    /// the result's).
    Select,
    /// A mask of exactly this type; another mask is E0605.
    Mask(TypeId),
    /// The index vector of a gather / scatter: an unsigned-lane vector with the lane count of this
    /// type (same-width unsigned lanes, the expected type of a mismatch). A sizeless index takes
    /// exactly this type, as sizeless lane counts differ by lane width.
    Index(TypeId),
    /// `&[T align A]` (`&mut` when `mutable`) with A ≥ `bytes`; a smaller alignment is E0606.
    Aligned {
        elem: TypeId,
        mutable: bool,
        bytes: u32,
    },
}

/// The built-in methods that take a turbofish; on any other built-in method or type-relative
/// built-in (`f32x4.splat`) one is E0201 case `type-args`.
pub const TAKES_GENERIC_ARGS: [&str; 5] = ["bitcast", "to_vec", "rewiden", "permute", "align_to"];

/// The signature of built-in method `m` on receiver type `t` with turbofish args `gargs`:
/// (argument types after the receiver, return type). `None`: wrong turbofish shape.
pub fn builtin_method_sig<I: Interner + ?Sized>(
    cx: &mut I,
    lang_option: DeclId,
    m: &BuiltinMethod,
    t: TypeId,
    gargs: &[GArg],
) -> Option<(Vec<ArgTy>, TypeId)> {
    let (us, unit) = (cx.fixed_ty(ids::USIZE), cx.fixed_ty(ids::UNIT));
    let k = cx.kind(t).clone();
    let opt = |cx: &mut I, x: TypeId| -> TypeId {
        if lang_option.is_none() { cx.error_ty() } else { cx.mk_adt(lang_option, vec![GArg::Type(x)]) }
    };
    Some(match (&k, m.name) {
        (TypeKind::Bool, "select") => (vec![ArgTy::Select, ArgTy::Select], cx.error_ty()),
        (TypeKind::Int { .. }, "mul_wide") => {
            let tup = cx.mk_tuple(vec![t, t]);
            (vec![ArgTy::Is(t)], tup)
        }
        (TypeKind::Int { .. }, "mul_hi") => (vec![ArgTy::Is(t)], t),
        (TypeKind::Int { .. } | TypeKind::Float(_), "store_stream") => {
            let s = cx.intern(TypeKind::Slice { elem: t, align: 0 });
            let rs = cx.mk_ref(true, s);
            (vec![ArgTy::Is(rs), ArgTy::Is(us)], unit)
        }
        (TypeKind::Int { .. }, _) => (vec![], t),
        (TypeKind::Ptr { inner, .. }, "read") => (vec![], *inner),
        (TypeKind::Ptr { inner, .. }, "write" | "write_stream") => (vec![ArgTy::Is(*inner)], unit),
        (TypeKind::Ptr { .. }, _) => (vec![], unit),
        (TypeKind::Slice { .. } | TypeKind::Array { .. } | TypeKind::Str, "len") => (vec![], us),
        (TypeKind::Slice { elem, .. }, "align_to") => {
            let a = match gargs {
                [GArg::Const(ConstArg::Value(a))] => *a,
                _ => return None,
            };
            let s = cx.intern(TypeKind::Slice { elem: *elem, align: a.min(1 << 20) as u32 });
            let r = cx.mk_ref(false, s);
            (vec![], opt(cx, r))
        }
        (TypeKind::Slice { .. }, _) => (vec![ArgTy::Is(us)], unit),
        (TypeKind::Vector { lane, n }, name) => vector_method(cx, t, *lane, *n, name, gargs, lang_option)?,
        (TypeKind::Mask { width, n }, name) => mask_method(cx, t, *width, *n, name, gargs)?,
        _ => return None,
    })
}

fn mask_of<I: Interner + ?Sized>(cx: &mut I, lane: TypeId, n: Lanes) -> TypeId {
    let w = types::lane_bits(cx.kind(lane)) as u8;
    cx.intern(TypeKind::Mask { width: w, n })
}

/// The lane type `lane` widened to twice its bits (`widen_lo`), if there is one.
fn wide_lane<I: Interner + ?Sized>(cx: &mut I, lane: TypeId) -> Option<TypeId> {
    use types::{FloatBits as F, IntBits as B};
    Some(match cx.kind(lane).clone() {
        TypeKind::Int { signed, bits } => {
            let bits = match bits {
                B::B8 => B::B16,
                B::B16 => B::B32,
                B::B32 => B::B64,
                _ => return None,
            };
            cx.intern(TypeKind::Int { signed, bits })
        }
        TypeKind::Float(F::F16 | F::B16) => cx.intern(TypeKind::Float(F::F32)),
        TypeKind::Float(F::F32) => cx.intern(TypeKind::Float(F::F64)),
        _ => return None,
    })
}

/// The int lane type of half the bits (`narrow_wrap`), if there is one.
fn narrow_lane<I: Interner + ?Sized>(cx: &mut I, lane: TypeId) -> Option<TypeId> {
    use types::IntBits as B;
    match cx.kind(lane).clone() {
        TypeKind::Int { signed, bits } => {
            let bits = match bits {
                B::B16 => B::B8,
                B::B32 => B::B16,
                B::B64 => B::B32,
                _ => return None,
            };
            Some(cx.intern(TypeKind::Int { signed, bits }))
        }
        _ => None,
    }
}

/// The same-width index vector of a gather / scatter on `lane x n` ([`ArgTy::Index`]).
fn index_vec<I: Interner + ?Sized>(cx: &mut I, lane: TypeId, n: Lanes) -> TypeId {
    use types::IntBits as B;
    let bits = match types::lane_bits(cx.kind(lane)) {
        8 => B::B8,
        16 => B::B16,
        32 => B::B32,
        _ => B::B64,
    };
    let u = cx.intern(TypeKind::Int { signed: false, bits });
    cx.intern(TypeKind::Vector { lane: u, n })
}

/// Bytes of `lane x n` (0 for a sizeless vector).
pub fn vec_bytes<I: Interner + ?Sized>(cx: &I, lane: TypeId, n: Lanes) -> u32 {
    match n {
        Lanes::Fixed(k) => types::lane_bits(cx.kind(lane)) / 8 * k as u32,
        Lanes::Native => 0,
    }
}

fn vector_method<I: Interner + ?Sized>(
    cx: &mut I,
    t: TypeId,
    lane: TypeId,
    n: Lanes,
    name: &str,
    gargs: &[GArg],
    lang_option: DeclId,
) -> Option<(Vec<ArgTy>, TypeId)> {
    let (us, unit) = (cx.fixed_ty(ids::USIZE), cx.fixed_ty(ids::UNIT));
    let slice = cx.intern(TypeKind::Slice { elem: lane, align: 0 });
    let rms = cx.mk_ref(true, slice);
    let mask = mask_of(cx, lane, n);
    let float = matches!(cx.kind(lane), TypeKind::Float(_));
    let signed = matches!(cx.kind(lane), TypeKind::Int { signed: true, .. });
    let half = |cx: &mut I, lane: TypeId| match n {
        Lanes::Fixed(k) if k >= 4 => Some(cx.intern(TypeKind::Vector { lane, n: Lanes::Fixed(k / 2) })),
        _ => None,
    };
    Some(match name {
        "sqrt" | "abs" | "floor" | "ceil" | "round" | "trunc" if float => (vec![], t),
        "abs_wrap" if signed => (vec![], t),
        "popcount" | "clz" if !float => (vec![], t),
        "scan_add" if float => (vec![], t),
        "scan_add_wrap" if !float => (vec![], t),
        "copysign" | "min_num" | "max_num" | "min_x86" | "max_x86" if float => (vec![ArgTy::Is(t)], t),
        "andnot" | "div_lanes" | "mul_hi" if !float => (vec![ArgTy::Is(t)], t),
        "min" | "max" => (vec![ArgTy::Is(t)], t),
        "fma" if float => (vec![ArgTy::Is(t), ArgTy::Is(t)], t),
        "reduce_add" if float => (vec![], lane),
        "reduce_add_wrap" if !float => (vec![], lane),
        "reduce_min" | "reduce_max" => (vec![], lane),
        "reduce_and" | "reduce_or" if !float => (vec![], lane),
        "reduce_add_ordered" if float => (vec![ArgTy::Is(lane)], lane),
        "checked_add" | "checked_sub" | "checked_mul" if !float => {
            let o = if lang_option.is_none() {
                cx.error_ty()
            } else {
                cx.mk_adt(lang_option, vec![GArg::Type(t)])
            };
            (vec![ArgTy::Is(t)], o)
        }
        // the divisor is a const-expr of the lane type (`Ck::div_const_divisor`)
        "div_const" if !float => (vec![ArgTy::Is(lane)], t),
        "store" => (vec![ArgTy::Is(rms), ArgTy::Is(us)], unit),
        "store_aligned" | "store_stream" => {
            let bytes = vec_bytes(cx, lane, n);
            (vec![ArgTy::Aligned { elem: lane, mutable: true, bytes }, ArgTy::Is(us)], unit)
        }
        "store_masked" => (vec![ArgTy::Is(rms), ArgTy::Is(us), ArgTy::Mask(mask)], unit),
        "scatter" => {
            let idx = index_vec(cx, lane, n);
            (vec![ArgTy::Is(rms), ArgTy::Index(idx)], unit)
        }
        "lookup" if !float => (vec![ArgTy::Is(t)], t),
        "compress" | "expand" => (vec![ArgTy::Mask(mask)], t),
        "split" => {
            let h = half(cx, lane)?;
            let tup = cx.mk_tuple(vec![h, h]);
            (vec![], tup)
        }
        "bitcast" => match gargs {
            [GArg::Type(u)] => match cx.kind(*u).clone() {
                TypeKind::Vector { lane: ul, n: un } if vec_bytes(cx, ul, un) == vec_bytes(cx, lane, n) => {
                    (vec![], *u)
                }
                _ => return None,
            },
            _ => return None,
        },
        "permute" => (vec![], t),
        "widen_lo" | "widen_hi" => {
            let w = wide_lane(cx, lane)?;
            (vec![], half(cx, w)?)
        }
        "narrow_wrap" | "narrow_sat" => {
            let l = narrow_lane(cx, lane)?;
            let Lanes::Fixed(k) = n else { return None };
            if k > 32 {
                return None;
            }
            let r = cx.intern(TypeKind::Vector { lane: l, n: Lanes::Fixed(k * 2) });
            (vec![ArgTy::Is(t)], r)
        }
        _ => return None,
    })
}

fn mask_method<I: Interner + ?Sized>(
    cx: &mut I,
    t: TypeId,
    width: u8,
    n: Lanes,
    name: &str,
    gargs: &[GArg],
) -> Option<(Vec<ArgTy>, TypeId)> {
    Some(match name {
        "select" => (vec![ArgTy::AnyVec, ArgTy::AnyVec], cx.error_ty()),
        "to_bits" => {
            let bits = match n {
                Lanes::Fixed(k) => k as u32,
                Lanes::Native => 64,
            };
            let u = match bits {
                0..=8 => cx.fixed_ty(ids::U8),
                9..=16 => cx.fixed_ty(ids::U16),
                17..=32 => cx.fixed_ty(ids::U32),
                _ => cx.fixed_ty(ids::U64),
            };
            (vec![], u)
        }
        "to_vec" => match gargs {
            [GArg::Type(l)]
                if matches!(cx.kind(*l), TypeKind::Int { .. })
                    && types::lane_bits(cx.kind(*l)) == width as u32 =>
            {
                let v = cx.intern(TypeKind::Vector { lane: *l, n });
                (vec![], v)
            }
            _ => return None,
        },
        "rewiden" => match gargs {
            [GArg::Const(ConstArg::Value(w))] if matches!(*w, 8 | 16 | 32 | 64) => {
                let m = cx.intern(TypeKind::Mask { width: *w as u8, n });
                (vec![], m)
            }
            _ => return None,
        },
        _ => {
            let _ = t;
            return None;
        }
    })
}

/// A type-relative built-in fn of a vector or mask type (`f32x4.splat(x)`): (params, ret).
pub fn builtin_assoc<I: Interner + ?Sized>(
    cx: &mut I,
    t: TypeId,
    name: &str,
) -> Option<(Vec<ArgTy>, TypeId)> {
    let us = cx.fixed_ty(ids::USIZE);
    match cx.kind(t).clone() {
        TypeKind::Vector { lane, n } => {
            let slice = cx.intern(TypeKind::Slice { elem: lane, align: 0 });
            let rs = cx.mk_ref(false, slice);
            let mask = mask_of(cx, lane, n);
            Some(match name {
                "splat" => (vec![ArgTy::Is(lane)], t),
                "load" => (vec![ArgTy::Is(rs), ArgTy::Is(us)], t),
                "load_aligned" => {
                    let bytes = vec_bytes(cx, lane, n);
                    (vec![ArgTy::Aligned { elem: lane, mutable: false, bytes }, ArgTy::Is(us)], t)
                }
                "load_masked" => (vec![ArgTy::Is(rs), ArgTy::Is(us), ArgTy::Mask(mask)], t),
                "load_first_fault" => (vec![ArgTy::Is(rs), ArgTy::Is(us)], t),
                "gather" => {
                    let idx = index_vec(cx, lane, n);
                    (vec![ArgTy::Is(rs), ArgTy::Index(idx)], t)
                }
                "join" => {
                    let Lanes::Fixed(k) = n else { return None };
                    if k < 4 {
                        return None;
                    }
                    let h = cx.intern(TypeKind::Vector { lane, n: Lanes::Fixed(k / 2) });
                    (vec![ArgTy::Is(h), ArgTy::Is(h)], t)
                }
                _ => return None,
            })
        }
        TypeKind::Mask { .. } => Some(match name {
            "first_n" => (vec![ArgTy::Is(us)], t),
            "all" | "none" => (vec![], t),
            _ => return None,
        }),
        _ => None,
    }
}

impl crate::typeck::Ck<'_> {
    /// The divisor of `v.div_const(k)` (`k` typed as the lane type `lane` by the call) as the
    /// bits of a `u64`, two's complement for a signed lane. `k` must be a const-expr (E0607)
    /// other than 0 (E0607 case `zero`); `vec` names the vector in the message.
    pub(crate) fn div_const_divisor(&mut self, k: u32, lane: TypeId, vec: TypeId) -> Option<u64> {
        let (n, had_error) = (self.diags.len(), self.has_error);
        let env = self.env();
        let ev = crate::consteval::eval_item_const(self, &env, k, lane);
        let v = ev.value;
        // "not a const-expr" gives way to E0607; an error inside a const-expr (E0203) stays
        let mut kept: Vec<Diagnostic> = self.diags.drain(n..).filter(|d| d.code() != "E0904").collect();
        let mut failed = !kept.is_empty();
        // a deferred error of a const-expr (a literal outside the lane type, a shift amount) is
        // reported once: by the checker where it typed the divisor, else here
        if let Some(d) = ev.deferred.filter(|d| d.code() != "E0904") {
            failed = true;
            if !self.diags.iter().any(|x| x.code() == d.code() && x.primary() == d.primary()) {
                kept.push(d);
            }
        }
        self.diags.extend(kept);
        self.has_error = had_error || failed;
        let tn = self.show(vec);
        let span = self.icx.node_span(k);
        match v {
            Some(crate::consteval::Val::Int(0)) => {
                self.report(mk("E0607", Some("zero"), &[("type", &tn)], span, &[]));
                None
            }
            Some(crate::consteval::Val::Int(d)) => Some(d as u64),
            _ if failed => None,
            _ => {
                self.report(mk("E0607", None, &[("type", &tn)], span, &[]));
                None
            }
        }
    }
}

/// Arch of an intrinsic (contracts/intrinsics.md).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IArch {
    X86,
    A64,
    Both,
    Ptx,
}

/// The table entry of intrinsic `name`: (arch, params, ret); `None`: not in the table. PTX
/// intrinsics have no checked signature (`params` empty, ret `ERROR`). The types are well-known
/// ids of a fresh table ([`ids`]); map them with [`Interner::fixed_ty`] on a canonical table.
pub fn intrinsic(name: &str) -> Option<(IArch, Vec<TypeId>, IRet)> {
    let us = ids::USIZE;
    if let Some(k) = name.strip_prefix("syscall").and_then(|s| s.parse::<usize>().ok()) {
        return (k <= 6).then(|| (IArch::Both, vec![us; k + 1], IRet::T(ids::ISIZE)));
    }
    if let Some(k) = name.strip_prefix("ccall").and_then(|s| s.parse::<usize>().ok()) {
        return (k <= 12).then(|| (IArch::Both, vec![us; k + 1], IRet::T(us)));
    }
    Some(match name {
        "cpuid" => (IArch::X86, vec![ids::U32, ids::U32], IRet::Cpuid),
        "xgetbv" => (IArch::X86, vec![ids::U32], IRet::T(ids::U64)),
        "thread_start_addr"
        | "main_addr"
        | "signal_handler_addr"
        | "site_table"
        | "pc_table"
        | "stack_table"
        | "thread_block" => (IArch::Both, vec![], IRet::T(us)),
        "set_thread_block" => (IArch::A64, vec![us], IRet::T(ids::UNIT)),
        "call_on_stack" => (IArch::Both, vec![us, us, us], IRet::T(us)),
        "cpu_init" => (IArch::Both, vec![], IRet::T(ids::UNIT)),
        "cpu_baseline" => (IArch::Both, vec![], IRet::T(ids::U64)),
        "thread_clone" => (IArch::Both, vec![us; 7], IRet::T(ids::ISIZE)),
        "atomic_load_u32" => (IArch::Both, vec![us], IRet::T(ids::U32)),
        "atomic_load_u64" => (IArch::Both, vec![us], IRet::T(ids::U64)),
        "atomic_xchg_u32" | "atomic_add_u32" => (IArch::Both, vec![us, ids::U32], IRet::T(ids::U32)),
        "atomic_xchg_u64" | "atomic_add_u64" => (IArch::Both, vec![us, ids::U64], IRet::T(ids::U64)),
        "atomic_cas_u32" => (IArch::Both, vec![us, ids::U32, ids::U32], IRet::T(ids::U32)),
        "atomic_cas_u64" => (IArch::Both, vec![us, ids::U64, ids::U64], IRet::T(ids::U64)),
        "copy_bytes" => (IArch::Both, vec![], IRet::CopyBytes),
        _ => {
            if is_ptx_intrinsic(name) {
                (IArch::Ptx, vec![], IRet::Any)
            } else {
                return None;
            }
        }
    })
}

/// An intrinsic's return type shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IRet {
    T(TypeId),
    /// `(u32, u32, u32, u32)`.
    Cpuid,
    /// Not checked (PTX set).
    Any,
    /// The whole signature `(dst: *mut u8, src: *const u8, n: usize)` with a unit result:
    /// pointer types have no well-known id ([`copy_bytes_ok`]).
    CopyBytes,
}

/// Whether `params` and `ret` are the signature of `copy_bytes`.
pub fn copy_bytes_ok<I: Interner + ?Sized>(cx: &I, params: &[TypeId], ret: TypeId) -> bool {
    let u8t = cx.fixed_ty(ids::U8);
    let ptr = |t: TypeId, m: bool| matches!(cx.kind(t), TypeKind::Ptr { mutable, inner } if *mutable == m && *inner == u8t);
    matches!(params, [d, s, n] if ptr(*d, true) && ptr(*s, false) && *n == cx.fixed_ty(ids::USIZE))
        && ret == cx.fixed_ty(ids::UNIT)
}

fn is_ptx_intrinsic(name: &str) -> bool {
    const FIXED: &[&str] = &[
        "fence_sc_gpu",
        "fence_sc_sys",
        "fence_acqrel_gpu",
        "nanosleep",
        "globaltimer",
        "clock64",
        "tid_x",
        "tid_y",
        "tid_z",
        "ntid_x",
        "ntid_y",
        "ntid_z",
        "ctaid_x",
        "ctaid_y",
        "ctaid_z",
        "nctaid_x",
        "nctaid_y",
        "nctaid_z",
        "laneid",
        "smid",
        "activemask",
        "match_any_u32",
        "shfl_idx_u32",
        "vote_ballot",
        "bar_sync",
        "bar_warp_sync",
        "ld_shared_u32",
        "st_shared_u32",
        "stacksave",
        "thread_exit",
    ];
    if FIXED.contains(&name) {
        return true;
    }
    let pre = ["ld_acquire_", "st_release_", "ld_relaxed_", "st_relaxed_", "atom_"];
    pre.iter().any(|p| name.starts_with(p))
        && (name.ends_with("u32") || name.ends_with("u64") || name.ends_with("_acqrel"))
}

/// E0805: `extern "stave-intrinsic"` outside the shipped packages (once per block, at the ABI
/// string), and names or signatures not in the table (case `unknown`, at the fn name).
pub(crate) fn check_intrinsics(
    prog: &Program,
    sigs: &Sigs,
    tys: &types::TypeTable,
    brackets: &[Brackets],
) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    let mut reported_blocks: Vec<(crate::ids::ItemId, u32)> = Vec::new();
    for (k, d) in prog.decls.iter().enumerate() {
        if d.kind != DeclKind::Fn || !d.flags.has(DeclFlags::EXTERN_IMPORT) {
            continue;
        }
        let Some(sig) = sigs.fn_sig(DeclId(k as u32)) else { continue };
        if sig.abi != Abi::Intrinsic {
            continue;
        }
        let Some(icx) = brackets.get(d.item.idx()).and_then(|b| ItemCx::new(prog, d.item, b)) else {
            continue;
        };
        let pkg = prog.files.get(icx.fid().0 as usize).map(|f| f.package);
        let shipped =
            pkg.and_then(|p| prog.packages.get(p.idx())).is_some_and(|p| p.kind == PackageKind::Builtin);
        if !shipped {
            // the extern block's ABI string
            let block = icx.ast.root().and_then(|r| {
                icx.ast.descendants(r).find(|&x| {
                    icx.kind(x) == Some(NodeKind::ExternItem) && icx.ast.descendants(x).contains(&d.node)
                })
            });
            if let Some(b) = block
                && !reported_blocks.contains(&(d.item, b))
            {
                reported_blocks.push((d.item, b));
                let name = prog.strings.get(d.name);
                out.extend(mk("E0805", None, &[("name", name)], icx.tok_span(icx.a(b) + 1), &[]));
            }
            continue;
        }
        let name = prog.strings.get(d.name);
        let ok = match intrinsic(name) {
            None => false,
            Some((_, _, IRet::Any)) => true,
            Some((_, _, IRet::CopyBytes)) => copy_bytes_ok(tys, &sig.params, sig.ret) && !sig.variadic,
            Some((_, params, ret)) => {
                let ret_ok = match ret {
                    IRet::T(t) => sig.ret == t,
                    IRet::Cpuid => cpuid_ok(tys, sig.ret),
                    IRet::Any | IRet::CopyBytes => true,
                };
                params == sig.params && ret_ok && !sig.variadic
            }
        };
        if !ok {
            out.extend(mk("E0805", Some("unknown"), &[("name", name)], d.name_span, &[]));
        }
    }
    out
}

/// Whether `t` is `(u32, u32, u32, u32)`.
pub fn cpuid_ok<I: Interner + ?Sized>(cx: &I, t: TypeId) -> bool {
    matches!(cx.kind(t), TypeKind::Tuple(ts) if ts.len() == 4 && ts.iter().all(|&x| x == cx.fixed_ty(ids::U32)))
}

/// The static name of a type-relative built-in (for [`crate::typeck::Target::Builtin`]).
pub fn builtin_assoc_name(name: &str) -> &'static str {
    const NAMES: &[&str] = &[
        "splat",
        "load",
        "load_aligned",
        "load_masked",
        "load_first_fault",
        "gather",
        "join",
        "first_n",
        "all",
        "none",
    ];
    NAMES.iter().copied().find(|&n| n == name).unwrap_or("?")
}
