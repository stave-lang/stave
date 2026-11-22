//! Diagnostic construction (only through `stave_diag::Diagnostic::new`, from registry templates).

use stave_diag::{Diagnostic, Span};

/// Builds a located diagnostic. A template failure is a compiler bug; it is counted (see
/// [`failures`]) instead of panicking, so no input can crash a checker worker, and the driver
/// fails the run as an internal error: a diagnostic never vanishes.
pub(crate) fn mk(
    code: &str,
    case: Option<&str>,
    holes: &[(&str, &str)],
    primary: Span,
    roles: &[(&str, Span)],
) -> Option<Diagnostic> {
    match Diagnostic::new(code, case, holes, Some(primary), roles) {
        Ok(d) => Some(d),
        Err(e) => {
            let _ = FIRST_FAILURE.set(format!("{code}: {e}"));
            FAILURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            None
        }
    }
}

static FAILURES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FIRST_FAILURE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// The first diagnostic since process start that could not be built: its code and the error.
pub fn first_failure() -> Option<&'static str> {
    FIRST_FAILURE.get().map(String::as_str)
}

/// How many diagnostics could not be built from their templates since process start (tests
/// assert 0).
pub fn failures() -> u64 {
    FAILURES.load(std::sync::atomic::Ordering::Relaxed)
}
