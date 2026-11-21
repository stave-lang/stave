#![allow(dead_code)]

pub mod forge;

use stave_ir::interp::{self, Config, Outcome, Val};
use stave_ir::{BufferHost, FuncBuilder, FuncId, InstanceId, Module, SyscallAbi, TypeId};

pub fn inst(n: &[u32]) -> InstanceId {
    InstanceId(n.to_vec())
}

pub fn host() -> BufferHost {
    BufferHost::new(SyscallAbi::X86_64)
}

/// Runs `f` with the default config; panics (test failure) on an interpreter error.
pub fn run(m: &Module, f: FuncId, args: Vec<Val>) -> Outcome {
    let (r, _) = interp::run(m, f, args, host(), Config::default());
    match r {
        Ok(o) => o,
        Err(e) => panic!("interpreter error: {e:?}"),
    }
}

/// Runs and also returns the host (stdout / stderr buffers).
pub fn run_host(m: &Module, f: FuncId, args: Vec<Val>) -> (Outcome, BufferHost) {
    let (r, h) = interp::run(m, f, args, host(), Config::default());
    match (r, h) {
        (Ok(o), Some(h)) => (o, h),
        (r, _) => panic!("interpreter error: {r:?}"),
    }
}

/// Declares and defines the function of compiler-known item `k` (its path, stamped with `k`, as
/// the front end lowers it).
pub fn define_known(
    m: &mut Module,
    id: &[u32],
    k: stave_ir::Known,
    params: &[TypeId],
    ret: Option<TypeId>,
    body: impl FnOnce(&mut FuncBuilder, &mut Module),
) -> FuncId {
    let f = define(m, id, &k.path(), params, ret, body);
    m.funcs[f.index()].known = Some(k);
    f
}

/// Declares and defines a function in one step.
pub fn define(
    m: &mut Module,
    id: &[u32],
    path: &str,
    params: &[TypeId],
    ret: Option<TypeId>,
    body: impl FnOnce(&mut FuncBuilder, &mut Module),
) -> FuncId {
    let f = m.declare(inst(id), path, params.to_vec(), ret);
    let mut b = FuncBuilder::new(params);
    body(&mut b, m);
    let body = b.finish().expect("builder");
    m.define(f, body);
    f
}

pub fn ret_val(o: Outcome) -> Val {
    match o {
        Outcome::Return(Some(v)) => v,
        o => panic!("expected a returned value, got {o:?}"),
    }
}
