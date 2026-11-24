//! Mutants of IR texts with stage0's verdict on each, for the stage1 IR gate
//! (`tools/ci-step/stage1-ir-gate.sh`), which loads the same files with stage1 and compares.
//! Ignored: the gate runs it.
//!
//! `STAVE_IR_MUTANTS_LIST`: a file of input paths, one per line; `STAVE_IR_MUTANTS_OUT`: the output
//! directory; `STAVE_IR_MUTANTS`: how many (default 2000). Mutant k is `OUT/k.ir`, made from input
//! `k % inputs` by the reader fuzz test's mutator with a fixed seed. `OUT/verdicts.txt` holds one
//! line `k VERDICT` per mutant: `accept`, or the loading step of its first fault (`syntax`,
//! `structure`, `verify`, `audit`).

mod common;

use std::path::PathBuf;

use common::mutate::{Rng, mutate};
use stave_ir::text::{ErrorKind, read};

#[test]
#[ignore]
fn write_mutants_and_verdicts() {
    let (Ok(list), Ok(out)) = (std::env::var("STAVE_IR_MUTANTS_LIST"), std::env::var("STAVE_IR_MUTANTS_OUT"))
    else {
        panic!("set STAVE_IR_MUTANTS_LIST and STAVE_IR_MUTANTS_OUT");
    };
    let n: usize = std::env::var("STAVE_IR_MUTANTS").ok().and_then(|s| s.parse().ok()).unwrap_or(2000);
    let inputs: Vec<Vec<u8>> = std::fs::read_to_string(&list)
        .expect("read the input list")
        .lines()
        .filter(|l| !l.is_empty())
        .map(|p| std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}")))
        .collect();
    assert!(!inputs.is_empty(), "{list}: no inputs");
    let out = PathBuf::from(out);
    std::fs::create_dir_all(&out).unwrap();
    let mut r = Rng(0x9e37_79b9_7f4a_7c15);
    let mut verdicts = String::new();
    for k in 0..n {
        let m = mutate(&mut r, &inputs[k % inputs.len()]);
        std::fs::write(out.join(format!("{k}.ir")), &m).unwrap();
        let v = match read(&m) {
            Ok(_) => "accept",
            Err(es) => match es.first().map(|e| e.kind) {
                Some(ErrorKind::Syntax) => "syntax",
                Some(ErrorKind::Structure) => "structure",
                Some(ErrorKind::Verify) => "verify",
                Some(ErrorKind::Audit) => "audit",
                None => "accept",
            },
        };
        verdicts.push_str(&format!("{k} {v}\n"));
    }
    std::fs::write(out.join("verdicts.txt"), verdicts).unwrap();
}
