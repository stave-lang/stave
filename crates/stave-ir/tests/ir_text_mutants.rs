//! Mutants of IR texts with stage0's verdict on each, for the stage1 IR gate
//! (`tools/ci-step/stage1-ir-gate.sh`), which loads the same files with stage1 and compares.
//! Ignored: the gate runs it.
//!
//! `STAVE_IR_MUTANTS_LIST`: a file of input paths, one per line; `STAVE_IR_MUTANTS_OUT`: the output
//! directory; `STAVE_IR_MUTANTS`: how many (default 2000). Mutant k is `OUT/k.ir`, made from input
//! `k % inputs` with a fixed seed: an even k by the reader fuzz test's mutator, an odd k by
//! [`retype`], which aims at the verifier. `OUT/verdicts.txt` holds one
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
        let base = &inputs[k % inputs.len()];
        let m = if k % 2 == 0 { mutate(&mut r, base) } else { retype(&mut r, base) };
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

/// A mutant for the verifier: in one instruction or terminator line, one type id `tN` becomes
/// another type of the file or one value `%N` another value number near it, so the text still
/// lexes and mostly keeps its structure while its typing changes.
fn retype(r: &mut Rng, base: &[u8]) -> Vec<u8> {
    let text = String::from_utf8_lossy(base).into_owned();
    let mut lines: Vec<String> = text.split('\n').map(str::to_string).collect();
    let ntypes = lines.iter().filter(|l| l.starts_with("type t")).count().max(1);
    let body: Vec<usize> = (0..lines.len()).filter(|&i| lines[i].starts_with("  ")).collect();
    if body.is_empty() {
        return mutate(r, base);
    }
    let i = body[r.below(body.len())];
    let mut toks: Vec<String> = lines[i].split(' ').map(str::to_string).collect();
    let slots: Vec<usize> = (0..toks.len())
        .filter(|&j| {
            let t = toks[j].trim_end_matches([',', ')']);
            let num = |p: char| t.strip_prefix(p).is_some_and(|d| !d.is_empty() && d.bytes().all(|c| c.is_ascii_digit()));
            num('t') || num('%')
        })
        .collect();
    if slots.is_empty() {
        return mutate(r, base);
    }
    let j = slots[r.below(slots.len())];
    let tok = toks[j].clone();
    let core = tok.trim_end_matches([',', ')']);
    let tail = &tok[core.len()..];
    let new = if core.starts_with('t') {
        format!("t{}", r.below(ntypes))
    } else {
        let v: usize = core[1..].parse().unwrap_or(0);
        format!("%{}", (v + r.below(5)).saturating_sub(2))
    };
    toks[j] = format!("{new}{tail}");
    lines[i] = toks.join(" ");
    lines.join("\n").into_bytes()
}
