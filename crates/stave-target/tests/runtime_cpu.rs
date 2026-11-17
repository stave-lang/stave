//! The runtime's feature-word constants against [`Feature`]: `runtime/cpu.stv` declares
//! `pub const <NAME>: u64 = 1 << <bit>;` for every feature (NAME = the feature name in upper
//! case, bit = `Feature as u8`), and its `CLOSURE` table row i is exactly `FeatureSet::of(i)`;
//! `runtime/{x86_64,aarch64}/cpu.stv` declare the same constants for the features their arch
//! owns and none of another arch's; `std/cpu.stv` names every CPU feature through runtime.cpu.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use stave_target::{Arch, Feature, FeatureSet};

fn repo_file(rel: &str) -> String {
    let p: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

fn const_name(f: Feature) -> String {
    f.name().to_ascii_uppercase()
}

/// `[pub] const NAME: u64 = 1 << K;` declarations whose NAME is a feature's constant name.
fn feature_consts(text: &str) -> BTreeMap<String, u64> {
    let mut out = BTreeMap::new();
    for line in text.lines() {
        let l = line.trim_start();
        let l = l.strip_prefix("pub ").unwrap_or(l);
        let Some(rest) = l.strip_prefix("const ") else { continue };
        let Some((name, rhs)) = rest.split_once(": u64 = ") else { continue };
        if !Feature::ALL.iter().any(|f| const_name(*f) == name) {
            continue;
        }
        let k = rhs.strip_prefix("1 << ").and_then(|s| s.strip_suffix(';'));
        let k: u32 = k.and_then(|k| k.parse().ok()).unwrap_or_else(|| panic!("`{line}`: not `1 << <bit>;`"));
        assert!(out.insert(name.to_string(), 1u64 << k).is_none(), "`{name}` declared twice");
    }
    out
}

#[test]
fn runtime_cpu_constants_equal_feature_bits() {
    let text = repo_file("runtime/cpu.stv");
    let consts = feature_consts(&text);
    assert_eq!(consts.len(), Feature::ALL.len(), "runtime/cpu.stv: one constant per feature");
    for f in Feature::ALL {
        assert_eq!(
            consts.get(&const_name(f)),
            Some(&(1u64 << f as u8)),
            "runtime/cpu.stv: `{}`",
            const_name(f)
        );
    }
    let n = format!("pub const NFEATURES: usize = {};", Feature::ALL.len());
    assert!(text.contains(&n), "runtime/cpu.stv lacks `{n}`");

    // CLOSURE rows: `A | B | ..,` in bit order.
    let start = text.find("const CLOSURE: [u64; ").expect("runtime/cpu.stv: CLOSURE table");
    let body = &text[start..];
    let body = &body[body.find("= [").unwrap() + 3..body.find("];").unwrap()];
    let rows: Vec<u64> = body
        .split(',')
        .map(str::trim)
        .filter(|r| !r.is_empty())
        .map(|r| {
            r.split('|').map(str::trim).fold(0, |acc, name| {
                acc | consts.get(name).unwrap_or_else(|| panic!("CLOSURE: unknown name `{name}`"))
            })
        })
        .collect();
    assert_eq!(rows.len(), Feature::ALL.len(), "CLOSURE: one row per feature");
    for (f, row) in Feature::ALL.into_iter().zip(rows) {
        assert_eq!(row, FeatureSet::of(f).bits(), "CLOSURE row of `{f}`");
    }
}

#[test]
fn arch_probe_constants_equal_feature_bits() {
    for (dir, arch) in [("x86_64", Arch::X86_64), ("aarch64", Arch::Aarch64)] {
        let consts = feature_consts(&repo_file(&format!("runtime/{dir}/cpu.stv")));
        for f in Feature::ALL {
            let got = consts.get(&const_name(f));
            if f.is_owned_by(arch) && f != Feature::Cuda {
                assert_eq!(got, Some(&(1u64 << f as u8)), "runtime/{dir}/cpu.stv: `{}`", const_name(f));
            } else {
                assert_eq!(got, None, "runtime/{dir}/cpu.stv names `{f}`, not a CPU feature of {arch}");
            }
        }
    }
}

#[test]
fn std_cpu_names_every_cpu_feature() {
    let text = repo_file("std/cpu.stv");
    for f in Feature::ALL.into_iter().filter(|f| *f != Feature::Cuda) {
        let c = const_name(f);
        let decl = format!("pub const {c}: u64 = rt.{c};");
        assert!(text.contains(&decl), "std/cpu.stv lacks `{decl}`");
    }
}
