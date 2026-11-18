//! `test fn`s with parameters in the stave-ir reference interpreter: for each `test fn` of
//! `tests/conformance/run/test_fn_params`, lowered for x86-64 and for AArch64, the generated
//! caller is valid checked IR and runs in the interpreter over every input `stave test` passes,
//! with one input marker per input and skip markers exactly where a `requires` filter rejects.

use std::path::{Path, PathBuf};

use stave_driver::exec;
use stave_driver::harness::{self, INPUT_MARKER, SKIP_MARKER};
use stave_driver::interp::{self, Limits};
use stave_driver::pipeline::{self, Goal, Options};
use stave_driver::shipped::Library;
use stave_sema::testgen::{self, Value};
use stave_target::Triple;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn inputs_run_in_the_interpreter() {
    let pkg = repo().join("tests/conformance/run/test_fn_params");
    for target in ["x86_64-linux", "aarch64-linux"] {
        let target = Triple::parse(target).unwrap();
        let mut o = Options::new(Goal::Check).threads(Some(4));
        o.target = Some(target);
        o.tests = true;
        o.lower = true;
        let out = pipeline::run(&pkg, &Library::in_dir(&repo()), &o).unwrap();
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        let tc = out.typeck.as_ref().unwrap();
        let mut m = out.module.clone().unwrap();
        let tests = exec::test_fns(&out.program, &m);
        assert_eq!(tests.len(), 6, "{target}");
        for (k, (name, f, decl)) in tests.into_iter().enumerate() {
            let d = tc.body(decl).and_then(|b| b.test_domain.as_deref()).unwrap_or_else(|| panic!("{name}"));
            let all = testgen::inputs(d);
            assert!(!all.is_empty(), "{name}");
            let picked: Vec<&[Value]> = all.iter().map(Vec::as_slice).collect();
            let caller = harness::add_caller(&mut m, f, d, &picked, k).unwrap();
            if let Err(es) = stave_ir::verify_module(&m) {
                panic!("{name} ({target}): {es:?}");
            }
            let path = m.func(caller).unwrap().path.clone();
            let r = interp::run(&m, &path, target, &[], &[], Limits::default()).unwrap();
            let err = String::from_utf8_lossy(&r.ran.stderr);
            assert_eq!(r.ran.exit, 0, "{name} ({target}): {err}");
            assert_eq!(harness::count(&r.ran.stdout, INPUT_MARKER), all.len(), "{name} ({target})");
            let skipped = harness::count(&r.ran.stdout, SKIP_MARKER);
            if name == "main.filtered" {
                let equal = all.iter().filter(|i| i[0] == i[1]).count();
                assert!(equal > 0);
                assert_eq!(skipped, equal, "{target}");
            } else {
                assert_eq!(skipped, 0, "{name} ({target})");
            }
        }
    }
}
