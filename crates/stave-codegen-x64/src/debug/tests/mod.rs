//! Backend tests: differential runs against the stave-ir interpreter (`harness`, each program as
//! a debug and as a release build), the hand-written programs (`cases`), native-only intrinsics
//! (`native`), random programs (`randgen`, `props`), the layout of cold code (`cold`), vector and
//! mask ops per op under every feature mode (`simd`), per-function feature sets (`features`), and
//! the codegen speed measurement (`bench`).

mod bench;
mod cases;
mod cold;
mod features;
mod harness;
mod kmode;
mod native;
mod props;
mod randgen;
mod regress;
mod simd;
mod stress;
