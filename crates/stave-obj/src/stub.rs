//! Fixed per-arch code the writer emits itself (contracts/intrinsics.md "Program entry").

use crate::{FuncCode, Machine, Reloc, RelocKind, SymbolId};

/// Name of the entry stub.
pub const START: &str = "_start";

/// Name of the runtime entry the stub calls (`runtime.start.entry(sp: *const usize)`).
pub const RUNTIME_ENTRY: &str = "runtime.start.entry";

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
