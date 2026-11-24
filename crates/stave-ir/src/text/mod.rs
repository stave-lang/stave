//! The IR text (contracts/ir-text.ebnf, contracts/checked-ir.md): the one printer of a module,
//! and the data the text holds besides the module.
//!
//! - [`print`]: [`print_module`] prints a module as the gate form of the text, or as the emit
//!   form when it is given the stack section.
//! - [`print_checked`]: the text of a bare module, for callers that have no target, mode, or
//!   entry point (tests, debugging).

mod print;

use stave_target::Triple;

pub use print::{print_checked, print_module};

/// The contract version of the text; the header's number.
pub const VERSION: u32 = 4;

/// The build mode of the header.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mode {
    Debug,
    Release,
}

impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Mode::Debug => "debug",
            Mode::Release => "release",
        }
    }
}

/// The IR level of the header (contracts/checked-ir.md "Levels and forms").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Level {
    /// After lowering and the verifier, before the removal of proven checks (`--dump=lir`).
    Lowered,
    /// After the removal of the checks that have a witness (`--dump=ir`).
    Checked,
    /// After the release passes (`--dump=opt`).
    Opt,
}

impl Level {
    pub fn name(self) -> &'static str {
        match self {
            Level::Lowered => "lowered",
            Level::Checked => "checked",
            Level::Opt => "opt",
        }
    }
}

/// The header line: what the lowering ran for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub target: Triple,
    pub mode: Mode,
    pub level: Level,
}

/// The kind of a call-graph edge of the stack section.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EdgeKind {
    Call,
    Indirect,
    Spawn,
}

impl EdgeKind {
    pub fn name(self) -> &'static str {
        match self {
            EdgeKind::Call => "call",
            EdgeKind::Indirect => "indirect",
            EdgeKind::Spawn => "spawn",
        }
    }
}

/// A call-graph edge of the stack section.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StackEdge {
    pub from: u32,
    pub to: u32,
    pub kind: EdgeKind,
}

/// The stack section (emit form only): what the stack bound reads besides the module.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StackInput {
    /// `runstack`: the root package's `[run] stack` in bytes.
    pub run_stack: Option<u64>,
    /// `truncated`: the call graph is incomplete.
    pub truncated: bool,
    /// Per call-graph node: whether it is a `rec fn`.
    pub nodes: Vec<bool>,
    /// Ascending by (from, to), one per pair.
    pub edges: Vec<StackEdge>,
    /// Per fn (by [`crate::FuncId`]): the nodes it stands for, ascending.
    pub nodes_of_func: Vec<Vec<u32>>,
}
