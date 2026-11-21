//! Fixed per-arch code the writer emits itself (contracts/intrinsics.md "Program entry").

use crate::{DataObj, FuncCode, Machine, Reloc, RelocKind, SymbolId};

/// Name of the entry stub.
pub const START: &str = "_start";

/// The symbol of the runtime entry (`runtime.start.entry(sp: *const usize)`) when its path is its
/// symbol. The backends hand the stub the entry's symbol id; this name only labels images built
/// without a front end.
pub const RUNTIME_ENTRY: &str = "runtime.start.entry";

/// Name of the stack table (contracts/intrinsics.md "Runtime start and traps").
pub const STACK_TABLE: &str = "__stave_stack_table";

/// `__stave_stack_table`: `u64 default` (the `[run] stack` bytes; 0: the runtime's own default),
/// `u64 main` (bytes of main's stack; 0: no bound, the default applies), `u64 count`, then
/// `count` rows of 16 bytes `{code: u64, bytes: u64}`: the code address a fn value holds (an
/// [`RelocKind::Abs64`] to the function, the jump-table stub where there is one) and the bytes of
/// the stack of a thread started with it (0: no bound).
pub fn stack_table(default: u64, main: u64, rows: &[(SymbolId, u64)]) -> DataObj {
    let mut bytes = Vec::with_capacity(24 + 16 * rows.len());
    let mut relocs = Vec::with_capacity(rows.len());
    bytes.extend_from_slice(&default.to_le_bytes());
    bytes.extend_from_slice(&main.to_le_bytes());
    bytes.extend_from_slice(&(rows.len() as u64).to_le_bytes());
    for &(sym, n) in rows {
        relocs.push(Reloc { offset: bytes.len() as u32, kind: RelocKind::Abs64, sym, addend: 0 });
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&n.to_le_bytes());
    }
    DataObj { bytes, align: 8, relocs }
}

/// `_start`: pass the initial stack pointer (pointing at `argc`) to `entry` in the first argument
/// register, with the frame pointer and link register cleared; trap if `entry` returns.
///
/// x86-64: `xor ebp,ebp; mov rdi,rsp; and rsp,-16; call entry; ud2`.
/// AArch64: `mov x29,#0; mov x30,#0; mov x0,sp; bl entry; brk #0`.
pub fn start(m: Machine, entry: SymbolId) -> FuncCode {
    match m {
        Machine::X86_64 => FuncCode {
            code: vec![
                0x31, 0xed, // xor ebp, ebp
                0x48, 0x89, 0xe7, // mov rdi, rsp
                0x48, 0x83, 0xe4, 0xf0, // and rsp, -16
                0xe8, 0, 0, 0, 0, // call entry
                0x0f, 0x0b, // ud2
            ],
            relocs: vec![Reloc { offset: 10, kind: RelocKind::X86Pc32, sym: entry, addend: -4 }],
            lines: Vec::new(),
        },
        Machine::Aarch64 => {
            let words: [u32; 5] = [
                0xd280_001d, // mov x29, #0
                0xd280_001e, // mov x30, #0
                0x9100_03e0, // mov x0, sp
                0x9400_0000, // bl entry
                0xd420_0000, // brk #0
            ];
            FuncCode {
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
                relocs: vec![Reloc { offset: 12, kind: RelocKind::A64Branch26, sym: entry, addend: 0 }],
                lines: Vec::new(),
            }
        }
    }
}

/// Size of one jump-table stub.
pub fn jump_stub_size(m: Machine) -> u32 {
    match m {
        Machine::X86_64 => 8,
        Machine::Aarch64 => 4,
    }
}

/// A jump-table stub with its branch field unresolved: x86-64 `jmp rel32` plus three `int3`;
/// AArch64 `b imm26`. Returns the stub bytes and the relocation (offset within the stub).
pub(crate) fn jump_stub(m: Machine) -> (&'static [u8], u32, RelocKind, i64) {
    match m {
        Machine::X86_64 => (&[0xe9, 0, 0, 0, 0, 0xcc, 0xcc, 0xcc], 1, RelocKind::X86Pc32, -4),
        Machine::Aarch64 => (&[0, 0, 0, 0x14], 0, RelocKind::A64Branch26, 0),
    }
}
