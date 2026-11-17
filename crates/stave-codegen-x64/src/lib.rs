//! x86-64 code generation: the instruction encoder ([`enc`]) and the debug backend ([`debug`]).

pub mod debug;
pub mod enc;

pub use debug::{Emitted, Error, Options, SiteRow, compile_func, compile_module, emit_module, link};
