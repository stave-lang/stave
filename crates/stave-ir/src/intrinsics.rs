//! The intrinsic table of contracts/intrinsics.md: every name an `extern "stave-intrinsic"` block
//! may declare, the arch it is built for, and its signature. This is the only copy: the front
//! end checks each declaration against it (E0805, E0704) and the verifier checks every
//! `intrinsic.<name>` instruction against it, so a name or a signature the table lacks is
//! rejected by both.
//!
//! [`STATIC_ADDR`] is no declaration: lowering alone emits it, and its result type depends on the
//! module ([`crate::Static::ty`]).

/// The name of the lowering-only intrinsic `static_addr <index>`: a pointer to static `index`.
pub const STATIC_ADDR: &str = "static_addr";

/// The arch an intrinsic is built for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IArch {
    X86,
    A64,
    Both,
    Ptx,
}

/// A parameter or result type of an intrinsic signature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ITy {
    Usize,
    Isize,
    U32,
    U64,
    /// `()`: as a result, the instruction has none.
    Unit,
    /// `(u32, u32, u32, u32)`.
    U32x4,
    /// `*mut u8`.
    MutPtrU8,
    /// `*const u8`.
    ConstPtrU8,
    /// `*const usize` (no intrinsic has it; the start entry of [`crate::known::abi_signature`]).
    ConstPtrUsize,
}

/// One table entry. `sig` is `None` for the PTX set, whose signatures are not checked and which
/// no target with a back end builds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Intrinsic {
    pub arch: IArch,
    pub sig: Option<(Vec<ITy>, ITy)>,
}

/// The table entry of intrinsic `name`; `None`: no such intrinsic.
pub fn lookup(name: &str) -> Option<Intrinsic> {
    use ITy::*;
    let entry = |arch, params: Vec<ITy>, ret| Some(Intrinsic { arch, sig: Some((params, ret)) });
    // `syscall3`, never `syscall03` or `syscall+3`: one spelling per name.
    let numbered = |prefix: &str, max: usize| {
        let d = name.strip_prefix(prefix)?;
        d.parse::<usize>().ok().filter(|&k| k <= max && k.to_string() == d)
    };
    if let Some(k) = numbered("syscall", 6) {
        return entry(IArch::Both, vec![Usize; k + 1], Isize);
    }
    if let Some(k) = numbered("ccall", 12) {
        return entry(IArch::Both, vec![Usize; k + 1], Usize);
    }
    match name {
        "cpuid" => entry(IArch::X86, vec![U32, U32], U32x4),
        "xgetbv" => entry(IArch::X86, vec![U32], U64),
        "thread_start_addr"
        | "main_addr"
        | "signal_handler_addr"
        | "site_table"
        | "pc_table"
        | "stack_table"
        | "thread_block" => entry(IArch::Both, vec![], Usize),
        "set_thread_block" => entry(IArch::A64, vec![Usize], Unit),
        "call_on_stack" => entry(IArch::Both, vec![Usize, Usize, Usize], Usize),
        "cpu_init" => entry(IArch::Both, vec![], Unit),
        "cpu_baseline" => entry(IArch::Both, vec![], U64),
        "thread_clone" => entry(IArch::Both, vec![Usize; 7], Isize),
        "atomic_load_u32" => entry(IArch::Both, vec![Usize], U32),
        "atomic_load_u64" => entry(IArch::Both, vec![Usize], U64),
        "atomic_xchg_u32" | "atomic_add_u32" => entry(IArch::Both, vec![Usize, U32], U32),
        "atomic_xchg_u64" | "atomic_add_u64" => entry(IArch::Both, vec![Usize, U64], U64),
        "atomic_cas_u32" => entry(IArch::Both, vec![Usize, U32, U32], U32),
        "atomic_cas_u64" => entry(IArch::Both, vec![Usize, U64, U64], U64),
        "copy_bytes" => entry(IArch::Both, vec![MutPtrU8, ConstPtrU8, Usize], Unit),
        _ if is_ptx(name) => Some(Intrinsic { arch: IArch::Ptx, sig: None }),
        _ => None,
    }
}

fn is_ptx(name: &str) -> bool {
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
