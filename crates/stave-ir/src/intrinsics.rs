//! The intrinsic table of contracts/intrinsics.md: every name an `extern "stave-intrinsic"` block
//! may declare, the arch it is built for, and its signature. Its one source is
//! `crates/stave-ir/tables/intrinsics.toml` (`tools/gen-tables` writes the table this file includes
//! and stage1's): [`kind`] is the one parse of a name, the front end checks each declaration against the table (E0805,
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

impl IArch {
    /// Whether an intrinsic of this arch is built for a program of `arch`. The PTX set has no
    /// lowering and no checked signature, so it is built for none.
    pub fn builds_for(self, arch: stave_target::Arch) -> bool {
        use stave_target::Arch;
        match self {
            IArch::Both => true,
            IArch::X86 => arch == Arch::X86_64,
            IArch::A64 => arch == Arch::Aarch64,
            IArch::Ptx => false,
        }
    }
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

include!("generated/intrinsic_table.rs");

/// A row of the generated table: a name, its kind, arch, and signature (`None`: unchecked).
struct Entry {
    name: &'static str,
    kind: Kind,
    arch: IArch,
    sig: Option<(&'static [ITy], ITy)>,
}

/// The table row of `k` (none for [`Kind::Ptx`]).
fn entry(k: Kind) -> Option<&'static Entry> {
    TABLE.iter().find(|e| e.kind == k)
}

/// The intrinsic named `name`; `None`: not in the table. One spelling per name: no leading zero,
/// sign, or extra digit in a number.
pub fn kind(name: &str) -> Option<Kind> {
    match TABLE.binary_search_by(|e| e.name.as_bytes().cmp(name.as_bytes())) {
        Ok(i) => Some(TABLE[i].kind),
        Err(_) if is_ptx(name) => Some(Kind::Ptx),
        Err(_) => None,
    }
}

impl Kind {
    /// The arch the intrinsic is built for. [`Kind::StaticAddr`] is built for every one.
    pub fn arch(self) -> IArch {
        entry(self).map_or(IArch::Ptx, |e| e.arch)
    }

    /// The declared signature (params, result); `None` for the PTX set (unchecked) and for
    /// [`Kind::StaticAddr`], whose result type depends on the module.
    pub fn sig(self) -> Option<(Vec<ITy>, ITy)> {
        entry(self).and_then(|e| e.sig).map(|(ps, r)| (ps.to_vec(), r))
    }
}

/// The name of the intrinsic; [`Kind::Ptx`] has none of its own.
impl std::fmt::Display for Kind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match entry(*self) {
            Some(e) => f.write_str(e.name),
            None => f.write_str("(PTX intrinsic)"),
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
    PTX_NAMES.contains(&name)
        || (PTX_PREFIXES.iter().any(|p| name.starts_with(p))
            && PTX_SUFFIXES.iter().any(|x| name.ends_with(x)))
}
