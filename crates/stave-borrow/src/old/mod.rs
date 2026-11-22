//! The dense dataflow: one full state per block, cloned per statement. It finds what
//! [`crate::loans`] and [`crate::check`] find, at a cost that grows with blocks times locals times
//! loans. `STAVE_BORROW_DATAFLOW=old` selects it; `both` runs it beside the sparse one and
//! fails a body whose diagnostics differ ([`crate::Dataflow`]).

pub mod check;
pub mod loans;
