//! The `Diagnostic` value, built only from registry templates.

use crate::span::Span;
use crate::template::{self, RenderError};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Severity {
    Error,
    Warning,
    Note,
}

impl Severity {
    /// The spelling in both output forms.
    pub fn as_str(self) -> &'static str {
        match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Note => "note",
        }
    }
}

/// A related location with its label. The label of a template's `"primary"` label is a related
/// entry at the primary span.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Related {
    pub span: Span,
    pub label: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Edit {
    pub span: Span,
    pub replacement: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fix {
    pub description: String,
    pub edits: Vec<Edit>,
}

/// A diagnostic. Its texts come from the registry; construct it with [`Diagnostic::new`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    code: &'static str,
    severity: Severity,
    primary: Option<Span>,
    related: Vec<Related>,
    message: String,
    fix: Option<Fix>,
}

/// Why a diagnostic could not be built. Each is a compiler bug at the call site.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DiagError {
    Render(RenderError),
    /// The code has a primary location, but none was given.
    MissingPrimary(&'static str),
    /// The code has no location, but a primary span was given.
    UnexpectedPrimary(&'static str),
    /// A role span was given that the chosen shape does not use (or `"primary"` as a role).
    UnknownRole {
        code: &'static str,
        role: String,
    },
    /// An error without a fix (the registry guarantees one; kept for robustness).
    MissingFix(&'static str),
}

impl std::fmt::Display for DiagError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DiagError::Render(e) => e.fmt(f),
            DiagError::MissingPrimary(c) => write!(f, "{c}: primary span required"),
            DiagError::UnexpectedPrimary(c) => write!(f, "{c}: code has no location, primary span given"),
            DiagError::UnknownRole { code, role } => {
                write!(f, "{code}: role {role:?} not used by this shape")
            }
            DiagError::MissingFix(c) => write!(f, "{c}: error without a fix"),
        }
    }
}

impl std::error::Error for DiagError {}

impl From<RenderError> for DiagError {
    fn from(e: RenderError) -> DiagError {
        DiagError::Render(e)
    }
}

impl Diagnostic {
    /// Builds a diagnostic of `code` from its templates.
    /// - `case`: the shape (`None` for the base shape);
    /// - `holes`: `(name, value)` pairs of canonical data;
    /// - `primary`: the primary span, `None` exactly for location-less codes;
    /// - `roles`: related spans by role. A role may be given several times (a chain or a cycle);
    ///   its label is printed once per span, in the order given. Labels whose role is not given
    ///   are left out, as are edits at such a role.
    pub fn new(
        code: &str,
        case: Option<&str>,
        holes: &[(&str, &str)],
        primary: Option<Span>,
        roles: &[(&str, Span)],
    ) -> Result<Diagnostic, DiagError> {
        let r = template::render(code, case, holes)?;
        let entry = r.entry;
        match (entry.located, primary) {
            (true, None) => return Err(DiagError::MissingPrimary(entry.code)),
            (false, Some(_)) => return Err(DiagError::UnexpectedPrimary(entry.code)),
            _ => {}
        }
        for (role, _) in roles {
            let used = *role != "primary"
                && (r.labels.iter().any(|l| l.at == *role)
                    || r.fix.as_ref().is_some_and(|f| f.edits.iter().any(|e| e.at == *role)));
            if !used {
                return Err(DiagError::UnknownRole { code: entry.code, role: role.to_string() });
            }
        }
        let spans_at = |at: &str| -> Vec<Span> {
            if at == "primary" {
                primary.into_iter().collect()
            } else {
                roles.iter().filter(|(r, _)| *r == at).map(|(_, s)| *s).collect()
            }
        };
        let mut related = Vec::new();
        for label in &r.labels {
            for span in spans_at(label.at) {
                related.push(Related { span, label: label.text.clone() });
            }
        }
        let fix = r.fix.map(|f| {
            let mut edits = Vec::new();
            for e in &f.edits {
                for span in spans_at(e.at) {
                    edits.push(Edit { span, replacement: e.replacement.clone() });
                }
            }
            Fix { description: f.description, edits }
        });
        if entry.severity == Severity::Error && fix.is_none() {
            return Err(DiagError::MissingFix(entry.code));
        }
        Ok(Diagnostic {
            code: entry.code,
            severity: entry.severity,
            primary,
            related,
            message: r.message,
            fix,
        })
    }

    pub fn code(&self) -> &'static str {
        self.code
    }

    pub fn severity(&self) -> Severity {
        self.severity
    }

    pub fn primary(&self) -> Option<Span> {
        self.primary
    }

    pub fn related(&self) -> &[Related] {
        &self.related
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn fix(&self) -> Option<&Fix> {
        self.fix.as_ref()
    }
}
