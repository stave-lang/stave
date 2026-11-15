//! Debug backend tests: differential runs against the stave-ir interpreter (`harness`), the
//! hand-written programs (`cases`), native-only intrinsics (`native`), random programs
//! (`randgen`, `props`), and the codegen speed measurement (`bench`).

mod bench;
mod cases;
mod harness;
mod native;
mod props;
mod randgen;
mod regress;
mod stress;
