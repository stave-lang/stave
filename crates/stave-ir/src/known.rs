//! Compiler-known items: the items of the built-in packages that stages treat specially
//! (contracts/intrinsics.md "Paths fixed by the compiler", the lang types and traits). This table
//! is the only place their paths are written. The front end resolves each entry once, by package
//! identity (an item of a module of a built-in package), and stamps the functions it lowers with
//! their entry ([`crate::Func::known`]); later stages ask for an entry, never for a path. A user
//! item spelled like one is an ordinary item everywhere.

use crate::intrinsics::ITy;

macro_rules! known_items {
    ($($v:ident = $module:literal $name:literal $(. $method:literal)?;)*) => {
        /// A compiler-known item.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub enum Known {
            $($v,)*
        }

        impl Known {
            /// Every entry, in declaration order.
            pub const ALL: &'static [Known] = &[$(Known::$v,)*];

            /// The module of the item in its built-in package, segments joined by `.`.
            pub fn module(self) -> &'static str {
                match self {
                    $(Known::$v => $module,)*
                }
            }

            /// The item's name in that module.
            pub fn name(self) -> &'static str {
                match self {
                    $(Known::$v => $name,)*
                }
            }

            /// An inherent method of that item, when the entry is the method.
            pub fn method(self) -> Option<&'static str> {
                match self {
                    $(Known::$v => known_items!(@method $($method)?),)*
                }
            }
        }
    };
    (@method) => {
        None
    };
    (@method $m:literal) => {
        Some($m)
    };
}

known_items! {
    Option = "std.core" "Option";
    Result = "std.core" "Result";
    Copy = "std.core" "Copy";
    Send = "std.core" "Send";
    Sync = "std.core" "Sync";
    Flat = "std.core" "Flat";
    Frozen = "std.core" "Frozen";
    Drop = "std.core" "Drop";
    Display = "std.core" "Display";
    Write = "std.core" "Write";
    OpAdd = "std.core" "Add";
    OpWrapAdd = "std.core" "WrapAdd";
    OpSatAdd = "std.core" "SatAdd";
    OpSub = "std.core" "Sub";
    OpWrapSub = "std.core" "WrapSub";
    OpSatSub = "std.core" "SatSub";
    OpMul = "std.core" "Mul";
    OpWrapMul = "std.core" "WrapMul";
    OpSatMul = "std.core" "SatMul";
    OpDiv = "std.core" "Div";
    OpRem = "std.core" "Rem";
    OpShl = "std.core" "Shl";
    OpWrapShl = "std.core" "WrapShl";
    OpShr = "std.core" "Shr";
    OpBitAnd = "std.core" "BitAnd";
    OpBitOr = "std.core" "BitOr";
    OpBitXor = "std.core" "BitXor";
    OpNeg = "std.core" "Neg";
    OpNot = "std.core" "Not";
    String = "std.string" "String";
    StringNew = "std.string" "String"."new";
    StringPushStr = "std.string" "String"."push_str";
    StringAsStr = "std.string" "String"."as_str";
    UnsafeCell = "std.cell" "UnsafeCell";
    UnsafeCellNew = "std.cell" "UnsafeCell"."new";
    Soa = "std.soa" "Soa";
    AtomicU32New = "std.atomic" "U32"."new";
    AtomicU64New = "std.atomic" "U64"."new";
    AtomicUsizeNew = "std.atomic" "Usize"."new";
    AtomicBoolNew = "std.atomic" "Bool"."new";
    Mutex = "std.sync" "Mutex";
    MutexNew = "std.sync" "Mutex"."new";
    MutexLock = "std.sync" "Mutex"."lock";
    MutexGuard = "std.sync" "MutexGuard";
    Once = "std.sync" "Once";
    OnceNew = "std.sync" "Once"."new";
    OnceGetOrInit = "std.sync" "Once"."get_or_init";
    LockPair = "std.sync" "lock_pair";
    MemForget = "std.mem" "forget";
    CopyRaw = "std.slice" "copy_raw";
    ThreadJoin = "std.thread" "join";
    ParFor = "std.thread" "par_for";
    ParForMut = "std.thread" "par_for_mut";
    StartEntry = "runtime.start" "entry";
    Trap = "runtime.panic" "trap";
    TrapMsg = "runtime.panic" "trap_msg";
    TrapErr = "runtime.panic" "trap_err";
    Abort = "runtime.panic" "abort";
    OnSignal = "runtime.panic" "on_signal";
    PanicBegin = "runtime.panic" "begin";
    RequiresFailed = "runtime.panic" "requires_failed";
    EnsuresFailed = "runtime.panic" "ensures_failed";
    SetUserSite = "runtime.panic" "set_user_site";
    TestInput = "runtime.panic" "test_input";
    TestSkip = "runtime.panic" "test_skip";
    ErrtracePush = "runtime.errtrace" "push";
    ErrtraceRestart = "runtime.errtrace" "restart";
    ErrtraceSave = "runtime.errtrace" "save";
    ErrtraceRestore = "runtime.errtrace" "restore";
    ThreadEntry = "runtime.thread" "entry";
    ThreadStart = "runtime.thread" "start";
    RtThreadJoin = "runtime.thread" "join";
    ScopeSpawn = "runtime.thread" "scope_spawn";
    ScopeJoin = "runtime.thread" "scope_join";
    RunQueued = "runtime.thread" "run_queued";
    CpuFeatures = "runtime.cpu" "features";
    CpuWord = "runtime.cpu" "WORD";
    CpuInit = "runtime.cpu" "init";
    CpuTrace = "runtime.cpu" "trace";
    Alloc = "runtime.alloc" "alloc";
    TryAlloc = "runtime.alloc" "try_alloc";
    Free = "runtime.alloc" "free";
    Realloc = "runtime.alloc" "realloc";
    TryRealloc = "runtime.alloc" "try_realloc";
    AtomicLock = "runtime.atomic" "lock";
    AtomicTryLock = "runtime.atomic" "try_lock";
    AtomicUnlock = "runtime.atomic" "unlock";
    Futex = "runtime.os" "futex";
    OsRead = "runtime.os" "read";
    OsWrite = "runtime.os" "write";
    OsPread64 = "runtime.os" "pread64";
    OsPwrite64 = "runtime.os" "pwrite64";
    OsOpenat = "runtime.os" "openat";
    OsFlock = "runtime.os" "flock";
}

/// The operator traits of `std.core`, in the order of the front end's operator table.
pub const OP_TRAITS: [Known; 19] = [
    Known::OpAdd,
    Known::OpWrapAdd,
    Known::OpSatAdd,
    Known::OpSub,
    Known::OpWrapSub,
    Known::OpSatSub,
    Known::OpMul,
    Known::OpWrapMul,
    Known::OpSatMul,
    Known::OpDiv,
    Known::OpRem,
    Known::OpShl,
    Known::OpWrapShl,
    Known::OpShr,
    Known::OpBitAnd,
    Known::OpBitOr,
    Known::OpBitXor,
    Known::OpNeg,
    Known::OpNot,
];

impl Known {
    /// The item's path (`runtime.panic.trap`, `std.string.String.new`): text for messages and
    /// dumps, never a key.
    pub fn path(self) -> String {
        match self.method() {
            Some(m) => format!("{}.{}.{m}", self.module(), self.name()),
            None => format!("{}.{}", self.module(), self.name()),
        }
    }
}

/// Whether the known item is a panic report entry, which runs on the alternate signal stack
/// ([`crate::Func::alt_stack`]): every trap and every std panic enters the report through one of
/// these. Any other item runs on the stack it is called on.
pub fn reports_on_alt_stack(k: Known) -> bool {
    matches!(
        k,
        Known::Trap
            | Known::TrapMsg
            | Known::TrapErr
            | Known::Abort
            | Known::RequiresFailed
            | Known::EnsuresFailed
    )
}

/// The signature (params, result) of a known item that code generation or the interpreter calls
/// or enters with an argument list of its own (the start stub, the trap and signal reports, the
/// thread stubs, `cpu_init`, `copy_bytes`, the checked allocator); `None` for the others, which
/// are only ever called by IR calls. A result of `()` is no result; `-> never` is none either.
pub fn abi_signature(k: Known) -> Option<(&'static [ITy], ITy)> {
    use ITy::*;
    Some(match k {
        Known::StartEntry => (&[ConstPtrUsize], Unit),
        Known::Trap => (&[U32, U32], Unit),
        Known::OnSignal => (&[Usize, Usize, Usize], Unit),
        Known::ThreadEntry => (&[Usize], Usize),
        Known::CpuInit => (&[], Unit),
        Known::CopyRaw => (&[MutPtrU8, ConstPtrU8, Usize], Unit),
        Known::TryAlloc => (&[Usize, Usize], MutPtrU8),
        Known::Free => (&[MutPtrU8, Usize, Usize], Unit),
        Known::TryRealloc => (&[MutPtrU8, Usize, Usize, Usize], MutPtrU8),
        // Called by IR calls only.
        Known::Option
        | Known::Result
        | Known::Copy
        | Known::Send
        | Known::Sync
        | Known::Flat
        | Known::Frozen
        | Known::Drop
        | Known::Display
        | Known::Write
        | Known::OpAdd
        | Known::OpWrapAdd
        | Known::OpSatAdd
        | Known::OpSub
        | Known::OpWrapSub
        | Known::OpSatSub
        | Known::OpMul
        | Known::OpWrapMul
        | Known::OpSatMul
        | Known::OpDiv
        | Known::OpRem
        | Known::OpShl
        | Known::OpWrapShl
        | Known::OpShr
        | Known::OpBitAnd
        | Known::OpBitOr
        | Known::OpBitXor
        | Known::OpNeg
        | Known::OpNot
        | Known::String
        | Known::StringNew
        | Known::StringPushStr
        | Known::StringAsStr
        | Known::UnsafeCell
        | Known::UnsafeCellNew
        | Known::Soa
        | Known::AtomicU32New
        | Known::AtomicU64New
        | Known::AtomicUsizeNew
        | Known::AtomicBoolNew
        | Known::Mutex
        | Known::MutexNew
        | Known::MutexLock
        | Known::MutexGuard
        | Known::Once
        | Known::OnceNew
        | Known::OnceGetOrInit
        | Known::LockPair
        | Known::MemForget
        | Known::ThreadJoin
        | Known::ParFor
        | Known::ParForMut
        | Known::TrapMsg
        | Known::TrapErr
        | Known::Abort
        | Known::PanicBegin
        | Known::RequiresFailed
        | Known::EnsuresFailed
        | Known::SetUserSite
        | Known::TestInput
        | Known::TestSkip
        | Known::ErrtracePush
        | Known::ErrtraceRestart
        | Known::ErrtraceSave
        | Known::ErrtraceRestore
        | Known::ThreadStart
        | Known::RtThreadJoin
        | Known::ScopeSpawn
        | Known::ScopeJoin
        | Known::RunQueued
        | Known::CpuFeatures
        | Known::CpuWord
        | Known::CpuTrace
        | Known::Alloc
        | Known::Realloc
        | Known::AtomicLock
        | Known::AtomicTryLock
        | Known::AtomicUnlock
        | Known::Futex
        | Known::OsRead
        | Known::OsWrite
        | Known::OsPread64
        | Known::OsPwrite64
        | Known::OsOpenat
        | Known::OsFlock => return None,
    })
}

/// Whether `f` is the fn a failed `requires` check calls.
pub fn is_requires_failed(f: &crate::Func) -> bool {
    f.known == Some(Known::RequiresFailed)
}

/// Whether `f` is the fn a failed `requires` or `ensures` check calls.
pub fn is_contract_failed(f: &crate::Func) -> bool {
    matches!(f.known, Some(Known::RequiresFailed | Known::EnsuresFailed))
}

impl std::fmt::Display for Known {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.path())
    }
}
