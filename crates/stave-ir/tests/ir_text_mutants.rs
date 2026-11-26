//! Mutants of IR texts with stage0's verdict on each, for the stage1 IR gate
//! (`tools/ci-step/stage1-ir-gate.sh`), which loads the same files with stage1 and compares.
//! Ignored: the gate runs it.
//!
//! `STAVE_IR_MUTANTS_LIST`: a file of input paths, one per line; `STAVE_IR_MUTANTS_OUT`: the output
//! directory; `STAVE_IR_MUTANTS`: how many (default 2000). Mutant k is `OUT/k.ir`, made from input
//! `k % inputs` with a fixed seed: an even k by the reader fuzz test's mutator, a k of 1 mod 4 by
//! [`retype`], which aims at the verifier, a k of 3 mod 4 by [`remut`], which aims at its
//! mutability rule. `OUT/verdicts.txt` holds one
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
        let m = match k % 4 {
            1 => retype(&mut r, base),
            3 => remut(&mut r, base),
            _ => mutate(&mut r, base),
        };
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
            let num = |p: char| {
                t.strip_prefix(p).is_some_and(|d| !d.is_empty() && d.bytes().all(|c| c.is_ascii_digit()))
            };
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

/// A pointer type line of a text: its id, whether it is a raw pointer, whether it is mutable,
/// and its pointee's token.
struct PtrLine {
    id: usize,
    raw: bool,
    mutable: bool,
    inner: String,
}

fn type_id(tok: &str) -> Option<usize> {
    tok.trim_end_matches([',', ')']).strip_prefix('t').and_then(|d| d.parse().ok())
}

/// A mutant for the mutability rule: in one instruction line, a pointer type `tN` becomes the
/// type of the file that differs from it only in mutability (`ref` and `ref mut`, `ptr const` and
/// `ptr mut`), or the pointer of a `store` becomes a value of its fn whose type is the shared form
/// of the pointer's type. Without either, a [`retype`] mutant.
fn remut(r: &mut Rng, base: &[u8]) -> Vec<u8> {
    let text = String::from_utf8_lossy(base).into_owned();
    let mut lines: Vec<String> = text.split('\n').map(str::to_string).collect();
    let mut ptrs: Vec<PtrLine> = Vec::new();
    for l in &lines {
        let toks: Vec<&str> = l.split(' ').collect();
        let (t, raw, mutable, inner) = match toks.as_slice() {
            ["type", t, "ref", "mut", p] => (t, false, true, p),
            ["type", t, "ref", p] => (t, false, false, p),
            ["type", t, "ptr", "mut", p] => (t, true, true, p),
            ["type", t, "ptr", "const", p] => (t, true, false, p),
            _ => continue,
        };
        if let Some(id) = type_id(t) {
            ptrs.push(PtrLine { id, raw, mutable, inner: inner.to_string() });
        }
    }
    // the type differing from `t` only in mutability (`shared`: only from a mutable `t`)
    let twin = |t: usize, shared: bool| {
        let p = ptrs.iter().find(|x| x.id == t)?;
        if shared && !p.mutable {
            return None;
        }
        ptrs.iter().find(|x| x.raw == p.raw && x.mutable != p.mutable && x.inner == p.inner).map(|x| x.id)
    };
    let body: Vec<usize> = (0..lines.len()).filter(|&i| lines[i].starts_with("  ")).collect();
    if body.is_empty() || ptrs.is_empty() {
        return retype(r, base);
    }
    for _ in 0..32 {
        let i = body[r.below(body.len())];
        let mut toks: Vec<String> = lines[i].split(' ').map(str::to_string).collect();
        if r.below(2) == 0 && toks.get(2).map(String::as_str) == Some("store") && toks.len() >= 6 {
            // `  store () %p %v`: the values of the fn up to here, typed by their definitions
            let Some(head) = (0..i).rev().find(|&j| lines[j].starts_with("fn ")) else { continue };
            let mut typed: Vec<(String, usize)> = Vec::new();
            for l in &lines[head..i] {
                let ts: Vec<&str> = l.trim_start().split(' ').collect();
                if let [v, "=", _, t, ..] = ts.as_slice()
                    && let Some(t) = type_id(t)
                {
                    typed.push((v.to_string(), t));
                }
                if l.starts_with('b') {
                    for part in l.split('%').skip(1) {
                        if let Some((v, t)) = part.split_once(": ")
                            && let Some(t) = t.split([',', ')', ' ']).next().and_then(type_id)
                        {
                            typed.push((format!("%{v}"), t));
                        }
                    }
                }
            }
            let Some(&(_, pt)) = typed.iter().find(|(v, _)| *v == toks[4]) else { continue };
            let Some(st) = twin(pt, true) else { continue };
            let shared: Vec<&String> = typed.iter().filter(|(_, t)| *t == st).map(|(v, _)| v).collect();
            if shared.is_empty() {
                continue;
            }
            toks[4] = shared[r.below(shared.len())].clone();
        } else {
            let slots: Vec<usize> = (0..toks.len())
                .filter(|&j| type_id(&toks[j]).and_then(|t| twin(t, false)).is_some())
                .collect();
            if slots.is_empty() {
                continue;
            }
            let j = slots[r.below(slots.len())];
            let core = toks[j].trim_end_matches([',', ')']).to_string();
            let tail = toks[j][core.len()..].to_string();
            let Some(t) = type_id(&core).and_then(|t| twin(t, false)) else { continue };
            toks[j] = format!("t{t}{tail}");
        }
        lines[i] = toks.join(" ");
        return lines.join("\n").into_bytes();
    }
    retype(r, base)
}
