//! Diagnostic construction (only through `stave_diag::Diagnostic::new`, from registry templates).

use stave_diag::{Diagnostic, Span};

/// Builds a located diagnostic. A template failure is a compiler bug; it is counted (see
/// [`failures`]) instead of panicking, so no input can crash the compiler.
pub(crate) fn mk(
    code: &str,
    case: Option<&str>,
    holes: &[(&str, &str)],
    primary: Span,
    roles: &[(&str, Span)],
) -> Option<Diagnostic> {
    match Diagnostic::new(code, case, holes, Some(primary), roles) {
        Ok(d) => Some(d),
        Err(_) => {
            FAILURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            None
        }
    }
}

static FAILURES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// How many diagnostics could not be built from their templates since process start (tests
/// assert 0).
pub fn failures() -> u64 {
    FAILURES.load(std::sync::atomic::Ordering::Relaxed)
}
