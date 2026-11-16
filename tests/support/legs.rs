//! Native test legs that must really run.
//!
//! `STAVE_REQUIRE_LEGS=v2,v3,aarch64` lists the legs (x86-64 `v2`, x86-64 `v3`, AArch64 `aarch64`
//! under `qemu-aarch64`) whose native runs may not be skipped: a test that cannot run a listed
//! leg fails and names what is missing. Unset or empty, a leg that cannot run is skipped. An
//! unknown leg name fails every gated test, so a typo cannot switch the check off.
//!
//! Included by path from each test crate that runs native binaries.
#![allow(dead_code)]

const KNOWN: [&str; 3] = ["v2", "v3", "aarch64"];

fn required() -> Vec<String> {
    let var = std::env::var("STAVE_REQUIRE_LEGS").unwrap_or_default();
    let legs: Vec<String> =
        var.split(',').map(str::trim).filter(|s| !s.is_empty()).map(String::from).collect();
    for l in &legs {
        assert!(KNOWN.contains(&l.as_str()), "STAVE_REQUIRE_LEGS: unknown leg `{l}` (known: {KNOWN:?})");
    }
    legs
}

/// `(leg, what the host needs)` of the target named `target`; `None` for targets without a leg.
fn leg_of(target: &str) -> Option<(&'static str, &'static str)> {
    match target {
        "x86_64-linux" | "x86_64-linux-v2" => Some(("v2", "an x86-64 Linux host whose CPU has sse4.2")),
        "x86_64-linux-v3" => Some(("v3", "an x86-64 Linux host whose CPU has avx2, fma, bmi2 and f16c")),
        "aarch64-linux" => Some(("aarch64", "an AArch64 Linux host or `qemu-aarch64` on PATH")),
        _ => None,
    }
}

/// Whether the native leg of `target` runs: `runnable` as probed by the caller. Panics when the
/// leg is required and `runnable` is false.
pub fn gate(target: &str, runnable: bool) -> bool {
    let required = required();
    if let Some((leg, needs)) = leg_of(target) {
        assert!(
            runnable || !required.iter().any(|r| r == leg),
            "STAVE_REQUIRE_LEGS requires leg `{leg}` ({target}) but it cannot run here: needs {needs}"
        );
    }
    runnable
}
