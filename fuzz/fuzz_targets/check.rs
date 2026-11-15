//! Fuzz the semantic front end: parse + resolve + type check + borrow check of one
//! file against the pinned stand-in std of the dump goldens, then, for an input
//! without error diagnostics, mono + lowering, the IR verifier, the bounds audit, and the
//! checked-IR printer. No panic on any bytes; every diagnostic locates and renders; a program
//! without errors lowers to IR that verifies and passes the bounds audit (or names a construct
//! stage0 does not support yet).
#![no_main]

use std::collections::BTreeMap;
use std::path::PathBuf;

use stave_diag::{emit_human, emit_json};
use stave_sema::{Config, Fs, Mode, Program};
use libfuzzer_sys::fuzz_target;

const STD: [(&str, &[u8]); 2] = [
    ("core.stv", include_bytes!("../../tests/conformance/dumps/_lib/std/core.stv")),
    ("io.stv", include_bytes!("../../tests/conformance/dumps/_lib/std/io.stv")),
];

fuzz_target!(|data: &[u8]| {
    let mut m = BTreeMap::new();
    m.insert(PathBuf::from("/fuzz/main.stv"), data.to_vec());
    for (p, t) in STD {
        m.insert(PathBuf::from("/std").join(p), t.to_vec());
    }
    let mut cfg = Config::new("/fuzz/main.stv").with_std("/std");
    cfg.fs = Fs::Mem(m);
    let prog = Program::load(&cfg);
    let tc = stave_sema::check(&prog);
    // borrow check (bodies without type errors only), with the per-access lines
    let bc = stave_borrow::check(&prog, &tc, stave_borrow::Options { dump: true });
    assert_eq!(stave_borrow::diagnostic_build_failures(), 0, "borrow diagnostics build from templates");
    let mut diags = tc.all_diagnostics(&prog);
    diags.extend(bc.diagnostics);
    let diags = stave_diag::canonical_order(diags, &prog.source_map);
    emit_json(&diags, &prog.source_map).expect("spans locate");
    emit_human(&diags, &prog.source_map).expect("spans locate");
    if diags.iter().any(|d| d.severity() == stave_diag::Severity::Error) {
        return;
    }
    match stave_sema::lower(&prog, &tc, Mode::Debug) {
        Ok(module) => {
            stave_ir::verify_module(&module).expect("lowered IR verifies");
            stave_ir::bounds::audit(&module).expect("every element access is checked or proven");
            let text = stave_ir::print_checked(&module);
            assert!(text.is_empty() || text.ends_with('\n'));
        }
        Err(es) => {
            // Only a construct stage0 does not support yet is acceptable for a valid program.
            for e in &es {
                assert_eq!(e.kind, stave_sema::LowerErrorKind::Unsupported, "lowering: {e:?}");
            }
        }
    }
});
