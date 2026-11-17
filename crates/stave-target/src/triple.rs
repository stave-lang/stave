//! Target triples (contracts/cli.md `--target`) and the target feature baseline.
//!
//! `x86_64-linux[-v2|-v3|-v4]`, `aarch64-linux[+sve|+sve2]`, `gpu-sm86`. `x86_64-linux` is
//! `x86_64-linux-v2`. The default is the host's *baseline* (`-v2` or plain NEON), never
//! auto-upgraded to what the host CPU supports (no silent choice).

use std::fmt;

use crate::features::{Arch, Feature, FeatureError, FeatureGroup, FeatureSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum X86Level {
    V2,
    V3,
    V4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum A64Ext {
    Base,
    Sve,
    Sve2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Triple {
    X86_64(X86Level),
    Aarch64(A64Ext),
    /// Whole program on an NVIDIA sm_86 GPU. Never a default.
    GpuSm86,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TripleError(pub String);

impl fmt::Display for TripleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown target `{}` (known: x86_64-linux[-v2|-v3|-v4], aarch64-linux[+sve|+sve2], gpu-sm86)",
            self.0
        )
    }
}

impl std::error::Error for TripleError {}

impl Triple {
    /// Every target, by canonical name (the names `[target.<T>]` tables may use, plus the
    /// `x86_64-linux` alias handled by [`Triple::parse`]).
    pub const ALL: [Triple; 7] = [
        Triple::X86_64(X86Level::V2),
        Triple::X86_64(X86Level::V3),
        Triple::X86_64(X86Level::V4),
        Triple::Aarch64(A64Ext::Base),
        Triple::Aarch64(A64Ext::Sve),
        Triple::Aarch64(A64Ext::Sve2),
        Triple::GpuSm86,
    ];

    pub fn parse(s: &str) -> Result<Triple, TripleError> {
        if s == "x86_64-linux" {
            return Ok(Triple::X86_64(X86Level::V2));
        }
        Triple::ALL.into_iter().find(|t| t.name() == s).ok_or_else(|| TripleError(s.to_string()))
    }

    /// Canonical name (`x86_64-linux` prints as `x86_64-linux-v2`).
    pub fn name(self) -> &'static str {
        match self {
            Triple::X86_64(X86Level::V2) => "x86_64-linux-v2",
            Triple::X86_64(X86Level::V3) => "x86_64-linux-v3",
            Triple::X86_64(X86Level::V4) => "x86_64-linux-v4",
            Triple::Aarch64(A64Ext::Base) => "aarch64-linux",
            Triple::Aarch64(A64Ext::Sve) => "aarch64-linux+sve",
            Triple::Aarch64(A64Ext::Sve2) => "aarch64-linux+sve2",
            Triple::GpuSm86 => "gpu-sm86",
        }
    }

    pub fn arch(self) -> Arch {
        match self {
            Triple::X86_64(_) => Arch::X86_64,
            Triple::Aarch64(_) => Arch::Aarch64,
            Triple::GpuSm86 => Arch::Gpu,
        }
    }

    /// Features every program for this target may assume (simd-ops.md §Features tiers):
    /// v2 = sse42; v3 and v4 = the feature groups of the same names; AArch64 = neon, `+sve`,
    /// `+sve2`. On `gpu-sm86`, `cuda` is always true. On CPU targets `cuda` is a runtime feature
    /// (a `dispatch` arm), never in the baseline.
    pub fn baseline(self) -> FeatureSet {
        use Feature as F;
        match self {
            Triple::X86_64(X86Level::V2) => FeatureSet::of(F::Sse42),
            Triple::X86_64(X86Level::V3) => FeatureGroup::V3.features(),
            Triple::X86_64(X86Level::V4) => FeatureGroup::V4.features(),
            Triple::Aarch64(A64Ext::Base) => FeatureSet::of(F::Neon),
            Triple::Aarch64(A64Ext::Sve) => FeatureSet::of(F::Sve),
            Triple::Aarch64(A64Ext::Sve2) => FeatureSet::of(F::Sve2),
            Triple::GpuSm86 => FeatureSet::of(F::Cuda),
        }
    }

    /// The lowest target of this triple's arch: `x86_64-linux-v2`, `aarch64-linux`, `gpu-sm86`.
    pub fn arch_base(self) -> Triple {
        match self {
            Triple::X86_64(_) => Triple::X86_64(X86Level::V2),
            Triple::Aarch64(_) => Triple::Aarch64(A64Ext::Base),
            Triple::GpuSm86 => Triple::GpuSm86,
        }
    }

    /// The host's baseline target: `x86_64-linux-v2` or `aarch64-linux`, from the architecture
    /// this compiler was built for. CPU features of the host are never probed for this (cli.md).
    /// `None` on any other architecture.
    pub fn host() -> Option<Triple> {
        if cfg!(target_arch = "x86_64") {
            Some(Triple::X86_64(X86Level::V2))
        } else if cfg!(target_arch = "aarch64") {
            Some(Triple::Aarch64(A64Ext::Base))
        } else {
            None
        }
    }
}

impl fmt::Display for Triple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TargetError {
    Triple(TripleError),
    Feature(FeatureError),
    /// `--features` names a feature another architecture owns.
    Foreign {
        feature: Feature,
        target: Triple,
    },
    /// No `--target` and the host architecture has no Stave target.
    NoHostTarget,
}

impl fmt::Display for TargetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TargetError::Triple(e) => e.fmt(f),
            TargetError::Feature(e) => e.fmt(f),
            TargetError::Foreign { feature, target } => write!(
                f,
                "feature `{feature}` is not available on target {target} (it belongs to {})",
                if feature.is_owned_by(Arch::X86_64) { "x86-64" } else { "AArch64" }
            ),
            TargetError::NoHostTarget => {
                f.write_str("the host architecture has no Stave target: pass --target")
            }
        }
    }
}

impl std::error::Error for TargetError {}

impl From<TripleError> for TargetError {
    fn from(e: TripleError) -> Self {
        TargetError::Triple(e)
    }
}

impl From<FeatureError> for TargetError {
    fn from(e: FeatureError) -> Self {
        TargetError::Feature(e)
    }
}

/// A target triple plus its feature set (baseline ∪ `--features`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Target {
    pub triple: Triple,
    pub features: FeatureSet,
}

impl Target {
    pub fn new(triple: Triple) -> Target {
        Target { triple, features: triple.baseline() }
    }

    /// The default target: the host baseline.
    pub fn host() -> Result<Target, TargetError> {
        Triple::host().map(Target::new).ok_or(TargetError::NoHostTarget)
    }

    /// Add features to the baseline. Every added feature must be owned by the target's
    /// architecture.
    pub fn with_features(self, extra: FeatureSet) -> Result<Target, TargetError> {
        if let Some(&feature) = extra.foreign_to(self.triple.arch()).first() {
            return Err(TargetError::Foreign { feature, target: self.triple });
        }
        Ok(Target { triple: self.triple, features: self.features.union(extra) })
    }

    /// From the command line: `--target` (default: host baseline) and every `--features` value.
    pub fn from_cli(target: Option<&str>, features: &[&str]) -> Result<Target, TargetError> {
        let t = match target {
            Some(s) => Target::new(Triple::parse(s)?),
            None => Target::host()?,
        };
        let mut extra = FeatureSet::EMPTY;
        for list in features {
            for name in list.split(',') {
                let feature: Feature = name.trim().parse()?;
                // Check the named feature (implications never cross architectures), so the error
                // names what the user wrote.
                if !feature.is_owned_by(t.triple.arch()) {
                    return Err(TargetError::Foreign { feature, target: t.triple });
                }
                extra.insert(feature);
            }
        }
        t.with_features(extra)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_names_roundtrip() {
        for t in Triple::ALL {
            assert_eq!(Triple::parse(t.name()), Ok(t));
        }
        assert_eq!(Triple::parse("x86_64-linux"), Ok(Triple::X86_64(X86Level::V2)));
        assert_eq!(Triple::parse("gpu-sm86").map(Triple::arch), Ok(Arch::Gpu));
        for bad in ["", "x86_64-linux-v5", "x86_64-linux-v1", "aarch64-linux+sve3", "aarch64", "gpu-sm80"] {
            assert_eq!(Triple::parse(bad), Err(TripleError(bad.to_string())), "{bad}");
        }
    }

    #[test]
    fn baselines() {
        let names = |t: Triple| t.baseline().to_string();
        assert_eq!(names(Triple::X86_64(X86Level::V2)), "sse42");
        assert_eq!(names(Triple::X86_64(X86Level::V3)), "sse42,avx,avx2,fma,bmi2,f16c");
        assert_eq!(
            names(Triple::X86_64(X86Level::V4)),
            "sse42,avx,avx2,fma,bmi2,f16c,avx512f,avx512bw,avx512vl,avx512dq,avx512cd"
        );
        assert_eq!(names(Triple::Aarch64(A64Ext::Base)), "neon");
        assert_eq!(names(Triple::Aarch64(A64Ext::Sve)), "neon,fp16,sve");
        assert_eq!(names(Triple::Aarch64(A64Ext::Sve2)), "neon,fp16,sve,sve2");
        assert_eq!(names(Triple::GpuSm86), "cuda");
        for t in Triple::ALL {
            assert!(t.baseline().is_owned_by(t.arch()), "{t}");
        }
        // Each tier contains the one below it.
        assert!(Triple::X86_64(X86Level::V2).baseline().is_subset(Triple::X86_64(X86Level::V3).baseline()));
        assert!(Triple::X86_64(X86Level::V3).baseline().is_subset(Triple::X86_64(X86Level::V4).baseline()));
        assert!(!Triple::X86_64(X86Level::V2).baseline().contains(Feature::Cuda));
    }

    #[test]
    fn host_default_is_baseline() {
        let h = Target::host().unwrap();
        assert_eq!(h.features, h.triple.baseline());
        assert!(matches!(h.triple, Triple::X86_64(X86Level::V2) | Triple::Aarch64(A64Ext::Base)));
        assert_eq!(Target::from_cli(None, &[]), Ok(h));
    }

    #[test]
    fn cli_features() {
        let t = Target::from_cli(Some("x86_64-linux"), &["avx2", "fma,bmi2"]).unwrap();
        assert_eq!(t.triple, Triple::X86_64(X86Level::V2));
        assert_eq!(t.features.to_string(), "sse42,avx,avx2,fma,bmi2");
        assert!(matches!(Target::from_cli(Some("x86_64-linux"), &["avx3"]), Err(TargetError::Feature(_))));
        assert!(matches!(Target::from_cli(Some("x86_64-linux-v9"), &[]), Err(TargetError::Triple(_))));
        assert_eq!(
            Target::from_cli(Some("x86_64-linux-v3"), &["sve"]),
            Err(TargetError::Foreign { feature: Feature::Sve, target: Triple::X86_64(X86Level::V3) })
        );
        assert!(matches!(Target::from_cli(Some("gpu-sm86"), &["avx2"]), Err(TargetError::Foreign { .. })));
        assert!(Target::from_cli(Some("aarch64-linux"), &["dotprod,cuda"]).is_ok());
    }
}
