//! The intrinsic table of contracts/intrinsics.md: every name an `extern "stave-intrinsic"` block
//! may declare, the arch it is built for, and its signature. This is the only copy: [`kind`] is
//! the one parse of a name, the front end checks each declaration against the table (E0805,
//! E0704), the verifier checks every `intrinsic.<name>` instruction against it, and every other
//! stage that tells intrinsics apart matches on [`Kind`]. A name or a signature the table lacks
//! is rejected by all of them.
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

/// The operation of a CPU atomic intrinsic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AtomicOp {
    Load,
    Xchg,
    Add,
    Cas,
}

impl AtomicOp {
    fn name(self) -> &'static str {
        match self {
            AtomicOp::Load => "load",
            AtomicOp::Xchg => "xchg",
            AtomicOp::Add => "add",
            AtomicOp::Cas => "cas",
        }
    }
}

/// The family of an intrinsic name. [`kind`] returns `None` for a name the table lacks, so no
/// stage sees a spelling the verifier would reject.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Kind {
    /// `syscall<k>`, `k` in `0..=6`: `k + 1` operands.
    Syscall(u8),
    /// `ccall<k>`, `k` in `0..=12`: `k + 1` operands.
    Ccall(u8),
    Cpuid,
    Xgetbv,
    ThreadStartAddr,
    MainAddr,
    SignalHandlerAddr,
    SiteTable,
    PcTable,
    StackTable,
    ThreadBlock,
    SetThreadBlock,
    CallOnStack,
    CpuInit,
    CpuBaseline,
    ThreadClone,
    /// `atomic_<op>_u32` / `atomic_<op>_u64`; `wide` is `u64`.
    Atomic {
        op: AtomicOp,
        wide: bool,
    },
    CopyBytes,
    /// [`STATIC_ADDR`]: no declaration, lowering alone emits it.
    StaticAddr,
    /// A name of the PTX set: no checked signature, and no back end builds it.
    Ptx,
}

/// The intrinsic named `name`; `None`: not in the table. One spelling per name: no leading zero,
/// sign, or extra digit in a number.
pub fn kind(name: &str) -> Option<Kind> {
    let numbered = |prefix: &str, max: u8| {
        let d = name.strip_prefix(prefix)?;
        let plain = !d.is_empty()
            && d.len() <= 2
            && d.bytes().all(|c| c.is_ascii_digit())
            && (d == "0" || !d.starts_with('0'));
        if plain { d.parse::<u8>().ok().filter(|&k| k <= max) } else { None }
    };
    if let Some(k) = numbered("syscall", 6) {
        return Some(Kind::Syscall(k));
    }
    if let Some(k) = numbered("ccall", 12) {
        return Some(Kind::Ccall(k));
    }
    if let Some(rest) = name.strip_prefix("atomic_") {
        let (op, wide) = match rest.rsplit_once('_')? {
            (op, "u32") => (op, false),
            (op, "u64") => (op, true),
            _ => return None,
        };
        let op = match op {
            "load" => AtomicOp::Load,
            "xchg" => AtomicOp::Xchg,
            "add" => AtomicOp::Add,
            "cas" => AtomicOp::Cas,
            _ => return None,
        };
        return Some(Kind::Atomic { op, wide });
    }
    Some(match name {
        "cpuid" => Kind::Cpuid,
        "xgetbv" => Kind::Xgetbv,
        "thread_start_addr" => Kind::ThreadStartAddr,
        "main_addr" => Kind::MainAddr,
        "signal_handler_addr" => Kind::SignalHandlerAddr,
        "site_table" => Kind::SiteTable,
        "pc_table" => Kind::PcTable,
        "stack_table" => Kind::StackTable,
        "thread_block" => Kind::ThreadBlock,
        "set_thread_block" => Kind::SetThreadBlock,
        "call_on_stack" => Kind::CallOnStack,
        "cpu_init" => Kind::CpuInit,
        "cpu_baseline" => Kind::CpuBaseline,
        "thread_clone" => Kind::ThreadClone,
        "copy_bytes" => Kind::CopyBytes,
        STATIC_ADDR => Kind::StaticAddr,
        _ if is_ptx(name) => Kind::Ptx,
        _ => return None,
    })
}

impl Kind {
    /// The arch the intrinsic is built for. [`Kind::StaticAddr`] is built for every one.
    pub fn arch(self) -> IArch {
        match self {
            Kind::Cpuid | Kind::Xgetbv => IArch::X86,
            Kind::SetThreadBlock => IArch::A64,
            Kind::Ptx => IArch::Ptx,
            Kind::Syscall(_)
            | Kind::Ccall(_)
            | Kind::ThreadStartAddr
            | Kind::MainAddr
            | Kind::SignalHandlerAddr
            | Kind::SiteTable
            | Kind::PcTable
            | Kind::StackTable
            | Kind::ThreadBlock
            | Kind::CallOnStack
            | Kind::CpuInit
            | Kind::CpuBaseline
            | Kind::ThreadClone
            | Kind::Atomic { .. }
            | Kind::CopyBytes
            | Kind::StaticAddr => IArch::Both,
        }
    }

    /// The declared signature (params, result); `None` for the PTX set (unchecked) and for
    /// [`Kind::StaticAddr`], whose result type depends on the module.
    pub fn sig(self) -> Option<(Vec<ITy>, ITy)> {
        use ITy::*;
        Some(match self {
            Kind::Syscall(k) => (vec![Usize; usize::from(k) + 1], Isize),
            Kind::Ccall(k) => (vec![Usize; usize::from(k) + 1], Usize),
            Kind::Cpuid => (vec![U32, U32], U32x4),
            Kind::Xgetbv => (vec![U32], U64),
            Kind::ThreadStartAddr
            | Kind::MainAddr
            | Kind::SignalHandlerAddr
            | Kind::SiteTable
            | Kind::PcTable
            | Kind::StackTable
            | Kind::ThreadBlock => (vec![], Usize),
            Kind::SetThreadBlock => (vec![Usize], Unit),
            Kind::CallOnStack => (vec![Usize, Usize, Usize], Usize),
            Kind::CpuInit => (vec![], Unit),
            Kind::CpuBaseline => (vec![], U64),
            Kind::ThreadClone => (vec![Usize; 7], Isize),
            Kind::Atomic { op, wide } => {
                let w = if wide { U64 } else { U32 };
                match op {
                    AtomicOp::Load => (vec![Usize], w),
                    AtomicOp::Xchg | AtomicOp::Add => (vec![Usize, w], w),
                    AtomicOp::Cas => (vec![Usize, w, w], w),
                }
            }
            Kind::CopyBytes => (vec![MutPtrU8, ConstPtrU8, Usize], Unit),
            Kind::StaticAddr | Kind::Ptx => return None,
        })
    }
}

/// The name of the intrinsic; [`Kind::Ptx`] has none of its own.
impl std::fmt::Display for Kind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Kind::Syscall(k) => write!(f, "syscall{k}"),
            Kind::Ccall(k) => write!(f, "ccall{k}"),
            Kind::Cpuid => f.write_str("cpuid"),
            Kind::Xgetbv => f.write_str("xgetbv"),
            Kind::ThreadStartAddr => f.write_str("thread_start_addr"),
            Kind::MainAddr => f.write_str("main_addr"),
            Kind::SignalHandlerAddr => f.write_str("signal_handler_addr"),
            Kind::SiteTable => f.write_str("site_table"),
            Kind::PcTable => f.write_str("pc_table"),
            Kind::StackTable => f.write_str("stack_table"),
            Kind::ThreadBlock => f.write_str("thread_block"),
            Kind::SetThreadBlock => f.write_str("set_thread_block"),
            Kind::CallOnStack => f.write_str("call_on_stack"),
            Kind::CpuInit => f.write_str("cpu_init"),
            Kind::CpuBaseline => f.write_str("cpu_baseline"),
            Kind::ThreadClone => f.write_str("thread_clone"),
            Kind::Atomic { op, wide } => {
                write!(f, "atomic_{}_{}", op.name(), if wide { "u64" } else { "u32" })
            }
            Kind::CopyBytes => f.write_str("copy_bytes"),
            Kind::StaticAddr => f.write_str(STATIC_ADDR),
            Kind::Ptx => f.write_str("(PTX intrinsic)"),
        }
    }
}

/// One table entry: what a declaration of an intrinsic is checked against. `sig` is `None` for
/// the PTX set, whose signatures are not checked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Intrinsic {
    pub kind: Kind,
    pub arch: IArch,
    pub sig: Option<(Vec<ITy>, ITy)>,
}

/// The table entry of the declarable intrinsic `name`; `None`: not declarable ([`STATIC_ADDR`]
/// included).
pub fn lookup(name: &str) -> Option<Intrinsic> {
    let kind = kind(name).filter(|&k| k != Kind::StaticAddr)?;
    Some(Intrinsic { kind, arch: kind.arch(), sig: kind.sig() })
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
