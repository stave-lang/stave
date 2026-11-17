//! Native test legs that must really run.
//!
//! `STAVE_REQUIRE_LEGS=v2,v3,aarch64,v4` lists the legs (x86-64 `v2`, x86-64 `v3`, AArch64
//! `aarch64` under `qemu-aarch64`, x86-64 `v4` natively or under Intel SDE) whose native runs may
//! not be skipped: a test that cannot run a listed leg fails and names what is missing. Unset or
//! empty, a leg that cannot run is skipped. An unknown leg name fails every gated test, so a typo
//! cannot switch the check off.
//!
//! A `v4` binary runs natively when the CPU has the v4 set, else through `$STAVE_SDE -icx --`
//! when `STAVE_SDE` names an executable file (Intel SDE's `sde64`, emulating an Ice Lake CPU
//! and its CPUID); [`v4`] says which.
//!
//! Included by path from each test crate that runs native binaries.
#![allow(dead_code)]

const KNOWN: [&str; 4] = ["v2", "v3", "aarch64", "v4"];

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
        "x86_64-linux-v4" => Some((
            "v4",
            "an x86-64 Linux host whose CPU has avx512f, avx512bw, avx512vl, avx512dq and avx512cd, \
             or `STAVE_SDE` naming an executable Intel SDE `sde64`",
        )),
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

/// How a binary of the `v4` leg runs here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum V4Run {
    /// On the CPU itself.
    Native,
    /// Through this runner command: words split at whitespace, the binary and its arguments
    /// appended.
    Runner(String),
}

/// `"$STAVE_SDE -icx --"` when `STAVE_SDE` names an executable file whose path has no
/// whitespace (the runner command is split at whitespace).
pub fn sde_runner() -> Option<String> {
    let sde =
        std::env::var("STAVE_SDE").ok().filter(|s| !s.is_empty() && !s.contains(char::is_whitespace))?;
    let meta = std::fs::metadata(&sde).ok()?;
    #[cfg(unix)]
    let exec = {
        use std::os::unix::fs::PermissionsExt;
        meta.permissions().mode() & 0o111 != 0
    };
    #[cfg(not(unix))]
    let exec = true;
    (meta.is_file() && exec).then(|| format!("{sde} -icx --"))
}

/// How the `v4` leg (`x86_64-linux-v4`) runs here: natively when `native` (the caller's probe of
/// the CPU, `exec::can_run`), else under [`sde_runner`]; `None` when neither is possible.
/// Panics when `STAVE_REQUIRE_LEGS` lists `v4` and it cannot run.
pub fn v4(native: bool) -> Option<V4Run> {
    let how = if native { Some(V4Run::Native) } else { sde_runner().map(V4Run::Runner) };
    gate("x86_64-linux-v4", how.is_some());
    how
}
