//! Runtime CPU feature detection (`std.cpu.features()` over `runtime.cpu.detect()`) against
//! ground truth, debug and release, as static ELF executables:
//!
//! - x86-64, run natively: the `flags` line of `/proc/cpuinfo` (the kernel clears AVX-class flags
//!   the OS does not enable), closed downward under implication like the runtime's word;
//! - AArch64, under `qemu-aarch64 -cpu <model>`: the features the model has (cortex-a76: no
//!   bf16/sve; neoverse-v1: sve but no sve2; max: all).
//!
//! The program prints `<feature> <0|1>` per CPU feature; features of the other arch must be 0.
//! A leg that cannot run here is skipped with a message, or fails when `STAVE_REQUIRE_LEGS`
//! lists it.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "../../../tests/support/legs.rs"]
mod legs;

use stave_driver::exec::{self, RunSpec, TempDir};
use stave_driver::pipeline::{self, Goal, Options, Output};
use stave_driver::shipped::Library;
use stave_target::{Arch, Feature, FeatureSet, Triple};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn cpu_features() -> impl Iterator<Item = Feature> {
    Feature::ALL.into_iter().filter(|f| *f != Feature::Cuda)
}

/// One `line(..)` call per CPU feature, in bit order.
fn program() -> String {
    let mut s = String::from(
        "import std.cpu;\nimport std.io;\n\n\
         fn line(name: &str, w: u64, bit: u64) {\n    io.print(name);\n    \
         if w & bit != 0 { io.print(\" 1\\n\"); } else { io.print(\" 0\\n\"); }\n}\n\n\
         fn main() -> i32 {\n    let w = cpu.features();\n",
    );
    for f in cpu_features() {
        s.push_str(&format!("    line(\"{}\", w, cpu.{});\n", f.name(), f.name().to_ascii_uppercase()));
    }
    s.push_str("    return 0;\n}\n");
    s
}

/// The expected output for a CPU with the raw feature set `raw` (unclosed): a feature is 1 iff
/// it and everything it implies are in `raw`.
fn expected(raw: &BTreeSet<Feature>) -> String {
    let mut bits = 0u64;
    for f in raw {
        bits |= 1 << *f as u8;
    }
    let mut s = String::new();
    for f in cpu_features() {
        let all = FeatureSet::of(f).bits();
        s.push_str(&format!("{} {}\n", f.name(), u8::from(bits & all == all)));
    }
    s
}

/// `/proc/cpuinfo` flag of each x86-64 feature.
fn cpuinfo_flag(f: Feature) -> &'static str {
    match f {
        Feature::Sse42 => "sse4_2",
        Feature::Avx512vbmi2 => "avx512_vbmi2",
        Feature::Avx512vpopcntdq => "avx512_vpopcntdq",
        Feature::Avx512bitalg => "avx512_bitalg",
        Feature::Avx512fp16 => "avx512_fp16",
        Feature::Avx512bf16 => "avx512_bf16",
        _ => f.name(),
    }
}

fn host_x86_features() -> BTreeSet<Feature> {
    let info = std::fs::read_to_string("/proc/cpuinfo").expect("/proc/cpuinfo");
    let line = info.lines().find(|l| l.starts_with("flags")).expect("/proc/cpuinfo: no flags line");
    let flags: BTreeSet<&str> = line.split_once(':').unwrap().1.split_whitespace().collect();
    cpu_features().filter(|f| f.is_owned_by(Arch::X86_64) && flags.contains(cpuinfo_flag(*f))).collect()
}

/// `qemu-aarch64 -cpu` models, the features they have, and the output lines not compared: qemu
/// versions disagree on whether neoverse-v1 reports bf16.
const QEMU_MODELS: [(&str, &[Feature], &[&str]); 3] = [
    ("cortex-a76", &[Feature::Neon, Feature::Fp16, Feature::Dotprod], &[]),
    (
        "neoverse-v1",
        &[Feature::Neon, Feature::Fp16, Feature::Dotprod, Feature::Bf16, Feature::Sve],
        &["bf16"],
    ),
    (
        "max",
        &[Feature::Neon, Feature::Fp16, Feature::Dotprod, Feature::Bf16, Feature::Sve, Feature::Sve2],
        &[],
    ),
];

/// One run: (target, runner, expected stdout, label, lines not compared).
type Case = (Triple, Option<String>, String, String, &'static [&'static str]);

/// `out` without the lines of the features named in `skip`.
fn without(out: &str, skip: &[&str]) -> String {
    out.lines()
        .filter(|l| !skip.iter().any(|s| l.split(' ').next() == Some(*s)))
        .map(|l| format!("{l}\n"))
        .collect()
}

fn have(tool: &str) -> bool {
    Command::new(tool).arg("--version").output().is_ok_and(|o| o.status.success())
}

fn build(p: &Path, target: Triple, release: bool) -> Output {
    let mut o = Options::new(Goal::Build).threads(Some(4));
    o.target = Some(target);
    o.release = release;
    let out = pipeline::run(p, &Library::in_dir(&repo()), &o).unwrap();
    assert!(!out.has_errors(), "{}: diagnostics {:?}", p.display(), out.diagnostics);
    let mode = if release { "release" } else { "debug" };
    assert!(out.failure.is_none(), "{} ({target}, {mode}): {}", p.display(), out.failure.as_ref().unwrap());
    assert!(out.image.is_some());
    out
}

#[test]
fn detected_features_match_ground_truth() {
    let tmp = TempDir::new("cpuf").unwrap();
    let src = tmp.path().join("cpu_features.stv");
    std::fs::write(&src, program()).unwrap();
    let x86 = Triple::parse("x86_64-linux").unwrap();
    let a64 = Triple::parse("aarch64-linux").unwrap();
    let native_x86 = legs::gate(x86.name(), exec::can_run(x86) && Path::new("/proc/cpuinfo").is_file());
    let qemu = legs::gate(a64.name(), cfg!(target_os = "linux") && have("qemu-aarch64"));

    let mut cases: Vec<Case> = Vec::new();
    if native_x86 {
        cases.push((x86, None, expected(&host_x86_features()), "native".into(), &[]));
    } else {
        eprintln!("note: x86_64-linux binaries cannot run here: build only");
    }
    if qemu {
        for (model, feats, skip) in QEMU_MODELS {
            let want = expected(&feats.iter().copied().collect());
            cases.push((a64, Some(format!("qemu-aarch64 -cpu {model}")), want, model.into(), skip));
        }
    } else {
        eprintln!("note: aarch64-linux binaries cannot run under qemu-aarch64 here: build only");
    }

    let mut failures = Vec::new();
    let mut runs = 0;
    for target in [x86, a64] {
        for release in [false, true] {
            let mode = if release { "release" } else { "debug" };
            let out = build(&src, target, release);
            let bin = tmp.path().join(format!("cpu-{target}-{mode}"));
            out.image.as_ref().unwrap().write_file(&bin).unwrap();
            for (t, runner, want, label, skip) in &cases {
                if *t != target {
                    continue;
                }
                let spec = RunSpec { args: &[], runner: runner.as_deref(), cwd: None, inherit: false };
                let got = exec::run(&bin, &spec).unwrap_or_else(|e| panic!("{}: {e}", bin.display()));
                runs += 1;
                if got.exit != 0
                    || without(&String::from_utf8_lossy(&got.stdout), skip) != without(want, skip)
                {
                    failures.push(format!(
                        "[{target}, {mode}, {label}]: exit {}\n--- stdout\n{}--- want\n{want}--- stderr\n{}",
                        got.exit,
                        String::from_utf8_lossy(&got.stdout),
                        String::from_utf8_lossy(&got.stderr)
                    ));
                }
            }
        }
    }
    eprintln!("{runs} runs");
    assert!(failures.is_empty(), "{} failure(s):\n{}", failures.len(), failures.join("\n"));
}

/// A dispatch that reads no feature word: its first live arm is in the baseline of both arches
/// (`neon` is dropped on x86-64, `sse42` on AArch64).
const BASELINE_DISPATCH: &str = "fn on_neon() -> u32 needs(neon) {\n    return 2;\n}\n\n\
     fn on_v2() -> u32 needs(sse42) {\n    return 1;\n}\n\n\
     fn on_base() -> u32 {\n    return 0;\n}\n\n\
     dispatch fn pick() -> u32 { neon => on_neon, sse42 => on_v2, _ => on_base, }\n\n\
     fn main() -> i32 {\n    return (pick() as% i32) - 1;\n}\n";

/// A program that does not use `std.cpu` and has no dispatch that tests the feature word has no
/// `runtime.cpu` code: detection is not linked in unconditionally, and start code fills no word.
#[test]
fn hello_has_no_cpu_detection() {
    let tmp = TempDir::new("cpuf-none").unwrap();
    let disp = tmp.path().join("baseline_dispatch.stv");
    std::fs::write(&disp, BASELINE_DISPATCH).unwrap();
    let hello = repo().join("tests/conformance/run/hello");
    for target in ["x86_64-linux", "aarch64-linux"].map(|t| Triple::parse(t).unwrap()) {
        for (p, release) in [(&hello, false), (&hello, true), (&disp, false), (&disp, true)] {
            let out = build(p, target, release);
            let m = out.module.as_ref().unwrap();
            let cpu: Vec<&str> = m
                .funcs
                .iter()
                .filter(|f| f.body.is_some() && f.path.starts_with("runtime.cpu."))
                .map(|f| f.path.as_str())
                .collect();
            assert!(cpu.is_empty(), "{} [{target}, release {release}]: has {cpu:?}", p.display());
        }
    }
}
