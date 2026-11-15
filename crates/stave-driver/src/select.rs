//! Compiler selection: `--compiler auto|cpu|gpu` and the stage1 probe (contracts/cli.md
//! `--compiler` probe table, cli-stage1.md "Summary line" and "Selection under auto").
//!
//! Probe steps, stopping at the first failure: 1 target supported by stage1 (code-generating
//! commands only; not AArch64 in v1), 1a no C objects linked, 2 stage1 image present, 3 NVIDIA
//! driver and device, 4 compute capability ≥ 8.6, 5 daemon reachable. Under `gpu` a failure is
//! E1303 (step 1), E1304 (1a), or E1302 (2–5); under `auto` it is the reason in the summary line.

use std::fmt;
use std::path::{Path, PathBuf};

use stave_target::{Arch, Triple};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CompilerChoice {
    #[default]
    Auto,
    Cpu,
    Gpu,
}

impl CompilerChoice {
    pub fn parse(s: &str) -> Option<CompilerChoice> {
        match s {
            "auto" => Some(CompilerChoice::Auto),
            "cpu" => Some(CompilerChoice::Cpu),
            "gpu" => Some(CompilerChoice::Gpu),
            _ => None,
        }
    }
}

/// The compiling commands (the ones that print a summary line).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompileCommand {
    Build,
    Run,
    Check,
    Test,
}

impl CompileCommand {
    /// `build`, `run`, `test` generate code; `check` does not (probe step 1 skips it).
    pub fn generates_code(self) -> bool {
        !matches!(self, CompileCommand::Check)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProbeFailure {
    /// Step 1.
    TargetNotSupported(Triple),
    /// Step 1a.
    CLinking,
    /// Step 2.
    ImageMissing,
    /// Step 3.
    NoDriver,
    /// Step 3.
    NoDevice,
    /// Step 4.
    ComputeCapability { major: u32, minor: u32 },
    /// Step 5.
    DaemonUnreachable,
}

impl ProbeFailure {
    /// The reason text of cli.md's table.
    pub fn reason(&self) -> String {
        match self {
            ProbeFailure::TargetNotSupported(t) => format!("target {t} not supported by stage1"),
            ProbeFailure::CLinking => "C linking not supported by stage1".to_string(),
            ProbeFailure::ImageMissing => "stage1 image missing".to_string(),
            ProbeFailure::NoDriver => "no NVIDIA driver".to_string(),
            ProbeFailure::NoDevice => "no device".to_string(),
            ProbeFailure::ComputeCapability { major, minor } => {
                format!("compute capability {major}.{minor} < 8.6")
            }
            ProbeFailure::DaemonUnreachable => "daemon not reachable".to_string(),
        }
    }

    /// Steps 1 and 1a are not "no usable GPU": their reason stands alone in the summary line.
    fn stands_alone(&self) -> bool {
        matches!(self, ProbeFailure::TargetNotSupported(_) | ProbeFailure::CLinking)
    }
}

/// What the probe needs to know.
#[derive(Clone, Debug)]
pub struct ProbeInput<'a> {
    pub command: CompileCommand,
    pub target: Triple,
    /// The package declares C-link inputs (probe step 1a).
    pub links_c: bool,
    /// The running `stave` binary.
    pub exe: Option<&'a Path>,
    /// This binary's build id (16 hex digits, resident-protocol.md); `None` until the build
    /// embeds one.
    pub build_id: Option<&'a str>,
}

/// `<stave install>/lib/stave/stage1-<build id>.img`, where `<stave install>` is the parent of the
/// directory holding the `stave` binary.
pub fn stage1_image_path(exe: &Path, build_id: &str) -> Option<PathBuf> {
    let install = exe.parent()?.parent()?;
    Some(install.join("lib").join("stave").join(format!("stage1-{build_id}.img")))
}

/// A usable stage1 (device name for the summary line).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1 {
    pub device: String,
}

/// Run the probe in cli.md order.
pub fn probe(input: &ProbeInput<'_>) -> Result<Stage1, ProbeFailure> {
    if input.command.generates_code() && input.target.arch() == Arch::Aarch64 {
        return Err(ProbeFailure::TargetNotSupported(input.target));
    }
    if input.links_c {
        return Err(ProbeFailure::CLinking);
    }
    let image = match (input.exe, input.build_id) {
        (Some(exe), Some(id)) => stage1_image_path(exe, id),
        _ => None,
    };
    if !image.is_some_and(|p| p.is_file()) {
        return Err(ProbeFailure::ImageMissing);
    }
    // Steps 3-5 (driver and device, compute capability, daemon) are not probed: this binary
    // cannot drive a stage1 image, so a present image still fails, at the last step.
    Err(ProbeFailure::DaemonUnreachable)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Selection {
    Stage1(Stage1),
    /// `--compiler cpu`.
    Stage0Forced,
    /// `--compiler auto` after a failed probe step.
    Stage0Fallback(ProbeFailure),
}

/// E1302, E1303, or E1304 for `--compiler gpu` (exit 1): the registry code, shape, and hole
/// values. Every text comes from the registry templates (`contracts/diagnostics.toml`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectError {
    pub code: &'static str,
    /// The registry shape (`None`: the base shape).
    pub case: Option<&'static str>,
    pub holes: Vec<(&'static str, String)>,
}

impl SelectError {
    fn from_failure(f: &ProbeFailure) -> SelectError {
        let (code, case, holes) = match f {
            ProbeFailure::TargetNotSupported(t) => ("E1303", None, vec![("target", t.name().to_string())]),
            ProbeFailure::CLinking => ("E1304", None, Vec::new()),
            ProbeFailure::ComputeCapability { major, minor } => {
                ("E1302", Some("cc"), vec![("major", major.to_string()), ("minor", minor.to_string())])
            }
            other => ("E1302", None, vec![("reason", other.reason())]),
        };
        SelectError { code, case, holes }
    }

    fn hole_refs(&self) -> Vec<(&str, &str)> {
        self.holes.iter().map(|(n, v)| (*n, v.as_str())).collect()
    }

    /// The message and fix texts.
    pub fn render(&self) -> Result<stave_diag::Rendered, stave_diag::RenderError> {
        stave_diag::render(self.code, self.case, &self.hole_refs())
    }

    /// The diagnostic. `primary`: E1304 is located (the first C-link input in `stave.toml`);
    /// E1302 and E1303 have no location (`None`).
    pub fn diagnostic(
        &self,
        primary: Option<stave_diag::Span>,
    ) -> Result<stave_diag::Diagnostic, stave_diag::DiagError> {
        stave_diag::Diagnostic::new(self.code, self.case, &self.hole_refs(), primary, &[])
    }
}

impl fmt::Display for SelectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.render() {
            Ok(r) => {
                write!(f, "error[{}]: {}", self.code, r.message)?;
                if let Some(fix) = r.fix {
                    write!(f, "\n  fix: {}", fix.description)?;
                }
                Ok(())
            }
            Err(e) => write!(f, "error[{}]: {e}", self.code),
        }
    }
}

/// Choose the compiler for a compiling command.
pub fn select(choice: CompilerChoice, input: &ProbeInput<'_>) -> Result<Selection, SelectError> {
    match choice {
        CompilerChoice::Cpu => Ok(Selection::Stage0Forced),
        CompilerChoice::Auto => Ok(match probe(input) {
            Ok(s) => Selection::Stage1(s),
            Err(f) => Selection::Stage0Fallback(f),
        }),
        CompilerChoice::Gpu => match probe(input) {
            Ok(s) => Ok(Selection::Stage1(s)),
            Err(f) => Err(SelectError::from_failure(&f)),
        },
    }
}

/// The summary line on stderr. `elapsed_ms` is shown for stage1 only.
pub fn summary_line(sel: &Selection, elapsed_ms: u64) -> String {
    match sel {
        Selection::Stage1(s) => format!("compiled on GPU (stage1, {}) in {elapsed_ms} ms", s.device),
        Selection::Stage0Forced => "compiled on CPU (--compiler cpu)".to_string(),
        Selection::Stage0Fallback(f) if f.stands_alone() => format!("compiled on CPU ({})", f.reason()),
        Selection::Stage0Fallback(f) => format!("compiled on CPU (no usable GPU: {})", f.reason()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(command: CompileCommand, target: &str) -> ProbeInput<'static> {
        ProbeInput {
            command,
            target: Triple::parse(target).unwrap(),
            links_c: false,
            exe: Some(Path::new("/opt/stave/bin/stave")),
            build_id: None,
        }
    }

    #[test]
    fn auto_falls_back_with_image_missing() {
        let s = select(CompilerChoice::Auto, &input(CompileCommand::Build, "x86_64-linux")).unwrap();
        assert_eq!(s, Selection::Stage0Fallback(ProbeFailure::ImageMissing));
        assert_eq!(summary_line(&s, 0), "compiled on CPU (no usable GPU: stage1 image missing)");
    }

    #[test]
    fn probe_order() {
        // Step 1 before 1a before 2; step 1 only for code-generating commands.
        let mut i = input(CompileCommand::Build, "aarch64-linux");
        i.links_c = true;
        let s = select(CompilerChoice::Auto, &i).unwrap();
        assert_eq!(summary_line(&s, 0), "compiled on CPU (target aarch64-linux not supported by stage1)");
        assert_eq!(select(CompilerChoice::Gpu, &i).unwrap_err().code, "E1303");
        i.command = CompileCommand::Check;
        assert_eq!(select(CompilerChoice::Gpu, &i).unwrap_err().code, "E1304");
        let s = select(CompilerChoice::Auto, &i).unwrap();
        assert_eq!(summary_line(&s, 0), "compiled on CPU (C linking not supported by stage1)");
        i.links_c = false;
        assert_eq!(select(CompilerChoice::Gpu, &i).unwrap_err().code, "E1302");
    }

    #[test]
    fn gpu_reports_e1302() {
        let e = select(CompilerChoice::Gpu, &input(CompileCommand::Run, "gpu-sm86")).unwrap_err();
        assert_eq!(e.code, "E1302");
        assert_eq!(e.render().unwrap().message, "`--compiler gpu`: no usable GPU (stage1 image missing)");
        assert!(
            e.to_string()
                .starts_with("error[E1302]: `--compiler gpu`: no usable GPU (stage1 image missing)\n  fix: ")
        );
        let d = e.diagnostic(None).unwrap();
        assert_eq!((d.code(), d.primary()), ("E1302", None));
        let cc = SelectError::from_failure(&ProbeFailure::ComputeCapability { major: 7, minor: 5 });
        assert_eq!(
            cc.render().unwrap().message,
            "`--compiler gpu`: no usable GPU (compute capability 7.5 < 8.6)"
        );
        // Every probe failure renders from the registry.
        for f in [
            ProbeFailure::TargetNotSupported(Triple::parse("aarch64-linux+sve").unwrap()),
            ProbeFailure::CLinking,
            ProbeFailure::ImageMissing,
            ProbeFailure::NoDriver,
            ProbeFailure::NoDevice,
            ProbeFailure::DaemonUnreachable,
        ] {
            let e = SelectError::from_failure(&f);
            let r = e.render().unwrap_or_else(|err| panic!("{f:?}: {err}"));
            assert!(r.fix.is_some(), "{f:?}");
            let located = e.code == "E1304";
            assert_eq!(e.diagnostic(None).is_ok(), !located, "{f:?}");
        }
    }

    #[test]
    fn cpu_and_lines() {
        let s = select(CompilerChoice::Cpu, &input(CompileCommand::Build, "aarch64-linux")).unwrap();
        assert_eq!(summary_line(&s, 0), "compiled on CPU (--compiler cpu)");
        let g = Selection::Stage1(Stage1 { device: "NVIDIA RTX A6000".into() });
        assert_eq!(summary_line(&g, 412), "compiled on GPU (stage1, NVIDIA RTX A6000) in 412 ms");
        assert_eq!(
            stage1_image_path(Path::new("/opt/stave/bin/stave"), "3f2a9c0d17e4b852"),
            Some(PathBuf::from("/opt/stave/lib/stave/stage1-3f2a9c0d17e4b852.img"))
        );
        assert_eq!(stage1_image_path(Path::new("stave"), "x"), None);
        assert_eq!(CompilerChoice::parse("gpu"), Some(CompilerChoice::Gpu));
        assert_eq!(CompilerChoice::parse("GPU"), None);
    }
}
