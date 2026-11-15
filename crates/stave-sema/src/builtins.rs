//! Built-ins: methods of the built-in types (`u32`/`u64` `mul_wide`, int bit counts,
//! raw-pointer `read`/`write` and memory hints, `len`, `bool` and mask `select`, SIMD vector and
//! mask operations), the type-relative fns of vector and mask types (`f32x4.splat`,
//! `u8x16.load`), and the `extern "stave-intrinsic"` table of contracts/intrinsics.md (E0805;
//! E0704 when called on another arch). `size_of` / `align_of` are typed by the body checker.
//!
//! `mul_wide` exists on `u32` and `u64` only: on any other type the lookup fails and the
//! checker reports E0208.

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

const INT_METHODS: &[&str] = &["mul_wide", "popcount", "clz", "ctz"];
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
    "abs_wrap",
    "andnot",
    "popcount",
    "clz",
    "reduce_add",
    "reduce_min",
    "reduce_max",
    "reduce_and",
    "reduce_or",
    "reduce_add_ordered",
    "scan_add",
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
    "store_masked",
    "scatter",
    "checked_add",
    "checked_sub",
    "checked_mul",
    "div_const",
    "div_lanes",
    "store_stream",
];
const MASK_METHODS: &[&str] = &["select", "to_bits", "to_vec", "all", "none", "rewiden"];

fn intern_name(list: &[&'static str], name: &str) -> Option<&'static str> {
    list.iter().copied().find(|&n| n == name)
}

/// The built-in method `name` of exactly type `t` (no auto-deref), if any.
pub fn builtin_method<I: Interner + ?Sized>(cx: &I, t: TypeId, name: &str) -> Option<BuiltinMethod> {
    match cx.kind(t) {
        TypeKind::Bool => {
            (name == "select").then_some(BuiltinMethod { name: "select", by_ref: false, is_unsafe: false })
        }
        TypeKind::Int { signed, bits } => {
            let n = intern_name(INT_METHODS, name)?;
            if n == "mul_wide" && (*signed || !matches!(bits, types::IntBits::B32 | types::IntBits::B64)) {
                return None;
            }
            Some(BuiltinMethod { name: n, by_ref: false, is_unsafe: false })
        }
        TypeKind::Ptr { mutable, .. } => {
            if name == "read" {
                return Some(BuiltinMethod { name: "read", by_ref: false, is_unsafe: true });
            }
            if name == "write" && *mutable {
                return Some(BuiltinMethod { name: "write", by_ref: false, is_unsafe: true });
            }
            if name == "write_stream" && *mutable {
                return Some(BuiltinMethod { name: "write_stream", by_ref: false, is_unsafe: true });
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

/// Expected argument shape of a built-in call: a fixed type, or "the same as argument k".
#[derive(Clone, Copy, Debug)]
pub enum ArgTy {
    Is(TypeId),
    /// Any vector type (checked by its use).
    AnyVec,
    /// An operand of `c.select(a, b)`: typed by the body checker (`a` and `b` have one type,
    /// the result's).
    Select,
}

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
    let rs = cx.mk_ref(false, slice);
    let rms = cx.mk_ref(true, slice);
    let mask = mask_of(cx, lane, n);
    Some(match name {
        "sqrt" | "abs" | "floor" | "ceil" | "round" | "trunc" | "abs_wrap" | "popcount" | "clz"
        | "scan_add" => (vec![], t),
        "copysign" | "min" | "max" | "min_num" | "max_num" | "min_x86" | "andnot" | "div_lanes" => {
            (vec![ArgTy::Is(t)], t)
        }
        "fma" => (vec![ArgTy::Is(t), ArgTy::Is(t)], t),
        "reduce_add" | "reduce_min" | "reduce_max" | "reduce_and" | "reduce_or" => (vec![], lane),
        "reduce_add_ordered" => (vec![ArgTy::Is(lane)], lane),
        "checked_add" | "checked_sub" | "checked_mul" => {
            let o = if lang_option.is_none() {
                cx.error_ty()
            } else {
                cx.mk_adt(lang_option, vec![GArg::Type(t)])
            };
            (vec![ArgTy::Is(t)], o)
        }
        "div_const" => (vec![], t),
        "store" | "store_stream" => (vec![ArgTy::Is(rms), ArgTy::Is(us)], unit),
        "store_masked" => (vec![ArgTy::Is(rms), ArgTy::Is(us), ArgTy::Is(mask)], unit),
        "scatter" => (vec![ArgTy::Is(rms), ArgTy::AnyVec], unit),
        "lookup" | "compress" | "expand" => (vec![ArgTy::AnyVec], t),
        "split" => {
            let Lanes::Fixed(k) = n else { return None };
            if k < 4 {
                return None;
            }
            let h = cx.intern(TypeKind::Vector { lane, n: Lanes::Fixed(k / 2) });
            let tup = cx.mk_tuple(vec![h, h]);
            (vec![], tup)
        }
        "bitcast" => match gargs {
            [GArg::Type(u)] if matches!(cx.kind(*u), TypeKind::Vector { .. }) => (vec![], *u),
            _ => return None,
        },
        "permute" => (vec![], t),
        "widen_lo" | "widen_hi" | "narrow_wrap" | "narrow_sat" => {
            let _ = rs;
            (vec![ArgTy::AnyVec], t)
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
        "select" => (vec![ArgTy::Select, ArgTy::Select], cx.error_ty()),
        "all" | "none" => (vec![], cx.fixed_ty(ids::BOOL)),
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
            [GArg::Type(l)] => {
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
            let _ = (t, width);
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
                "load_aligned" => (vec![ArgTy::AnyVec, ArgTy::Is(us)], t),
                "load_masked" => (vec![ArgTy::Is(rs), ArgTy::Is(us), ArgTy::Is(mask)], t),
                "load_first_fault" => (vec![ArgTy::Is(rs), ArgTy::Is(us)], t),
                "gather" => (vec![ArgTy::Is(rs), ArgTy::AnyVec], t),
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
            _ => return None,
        }),
        _ => None,
    }
}

/// Arch of an intrinsic (contracts/intrinsics.md).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IArch {
    X86,
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
        "thread_start_addr" | "main_addr" | "signal_handler_addr" | "site_table" | "pc_table" => {
            (IArch::Both, vec![], IRet::T(us))
        }
        "call_on_stack" => (IArch::Both, vec![us, us, us], IRet::T(us)),
        "thread_clone" => (IArch::Both, vec![us; 7], IRet::T(ids::ISIZE)),
        "atomic_load_u32" => (IArch::Both, vec![us], IRet::T(ids::U32)),
        "atomic_load_u64" => (IArch::Both, vec![us], IRet::T(ids::U64)),
        "atomic_xchg_u32" | "atomic_add_u32" => (IArch::Both, vec![us, ids::U32], IRet::T(ids::U32)),
        "atomic_xchg_u64" | "atomic_add_u64" => (IArch::Both, vec![us, ids::U64], IRet::T(ids::U64)),
        "atomic_cas_u32" => (IArch::Both, vec![us, ids::U32, ids::U32], IRet::T(ids::U32)),
        "atomic_cas_u64" => (IArch::Both, vec![us, ids::U64, ids::U64], IRet::T(ids::U64)),
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
            Some((_, params, ret)) => {
                let ret_ok = match ret {
                    IRet::T(t) => sig.ret == t,
                    IRet::Cpuid => cpuid_ok(tys, sig.ret),
                    IRet::Any => true,
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
    const NAMES: &[&str] =
        &["splat", "load", "load_aligned", "load_masked", "load_first_fault", "gather", "join", "first_n"];
    NAMES.iter().copied().find(|&n| n == name).unwrap_or("?")
}
