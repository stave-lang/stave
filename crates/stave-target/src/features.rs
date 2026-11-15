//! Target features (contracts/simd-ops.md §Features).
//!
//! A [`FeatureSet`] is a `u64` bitset that is always closed under implication: inserting `avx2`
//! also inserts `avx` and `sse42`. Every feature is owned by one architecture, except `cuda` (a
//! usable NVIDIA device), which is owned by x86-64, AArch64, and the `gpu-sm86` program target.
//!
//! Implications beyond the chains the contract names (`sse42 ⊂ avx ⊂ avx2`,
//! `avx512f ⊂ avx512{bw,vl,dq,cd}`, `sve ⊂ sve2`) follow the hardware (as LLVM models it): `fma` and
//! `f16c` need `avx` (VEX encoding), `avx512f` needs `avx2 fma f16c`, the AVX-512 extras need their
//! base subsets, every AArch64 extra needs `neon`, and `sve` needs `fp16`.

use std::fmt;
use std::str::FromStr;

/// An architecture that owns features. `Gpu` is the whole-program target `gpu-sm86`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Arch {
    X86_64,
    Aarch64,
    Gpu,
}

impl Arch {
    pub fn name(self) -> &'static str {
        match self {
            Arch::X86_64 => "x86-64",
            Arch::Aarch64 => "AArch64",
            Arch::Gpu => "gpu-sm86",
        }
    }

    const fn bit(self) -> u8 {
        1 << self as u8
    }
}

impl fmt::Display for Arch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A set of architectures (the owners of a feature).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ArchSet(u8);

impl ArchSet {
    pub const NONE: ArchSet = ArchSet(0);
    pub const X86_64: ArchSet = ArchSet(Arch::X86_64.bit());
    pub const AARCH64: ArchSet = ArchSet(Arch::Aarch64.bit());
    pub const ALL: ArchSet = ArchSet(Arch::X86_64.bit() | Arch::Aarch64.bit() | Arch::Gpu.bit());

    pub fn contains(self, a: Arch) -> bool {
        self.0 & a.bit() != 0
    }

    pub fn intersect(self, o: ArchSet) -> ArchSet {
        ArchSet(self.0 & o.0)
    }
}

/// A target feature. The discriminant is its bit in [`FeatureSet`]; the order is the contract's
/// listing order, which is also the printing order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Feature {
    Sse42,
    Avx,
    Avx2,
    Fma,
    Bmi2,
    F16c,
    Avx512f,
    Avx512bw,
    Avx512vl,
    Avx512dq,
    Avx512cd,
    Avx512vbmi,
    Avx512vbmi2,
    Avx512vpopcntdq,
    Avx512bitalg,
    Avx512fp16,
    Avx512bf16,
    Neon,
    Fp16,
    Dotprod,
    Bf16,
    Sve,
    Sve2,
    Cuda,
}

const N: usize = 24;

struct Info {
    name: &'static str,
    owners: ArchSet,
    implies: &'static [Feature],
}

const X86: ArchSet = ArchSet::X86_64;
const A64: ArchSet = ArchSet::AARCH64;

use Feature as F;

/// Indexed by `Feature as usize`.
const INFO: [Info; N] = [
    Info { name: "sse42", owners: X86, implies: &[] },
    Info { name: "avx", owners: X86, implies: &[F::Sse42] },
    Info { name: "avx2", owners: X86, implies: &[F::Avx] },
    Info { name: "fma", owners: X86, implies: &[F::Avx] },
    Info { name: "bmi2", owners: X86, implies: &[] },
    Info { name: "f16c", owners: X86, implies: &[F::Avx] },
    Info { name: "avx512f", owners: X86, implies: &[F::Avx2, F::Fma, F::F16c] },
    Info { name: "avx512bw", owners: X86, implies: &[F::Avx512f] },
    Info { name: "avx512vl", owners: X86, implies: &[F::Avx512f] },
    Info { name: "avx512dq", owners: X86, implies: &[F::Avx512f] },
    Info { name: "avx512cd", owners: X86, implies: &[F::Avx512f] },
    Info { name: "avx512vbmi", owners: X86, implies: &[F::Avx512bw] },
    Info { name: "avx512vbmi2", owners: X86, implies: &[F::Avx512bw] },
    Info { name: "avx512vpopcntdq", owners: X86, implies: &[F::Avx512f] },
    Info { name: "avx512bitalg", owners: X86, implies: &[F::Avx512bw] },
    Info { name: "avx512fp16", owners: X86, implies: &[F::Avx512bw, F::Avx512vl, F::Avx512dq] },
    Info { name: "avx512bf16", owners: X86, implies: &[F::Avx512bw] },
    Info { name: "neon", owners: A64, implies: &[] },
    Info { name: "fp16", owners: A64, implies: &[F::Neon] },
    Info { name: "dotprod", owners: A64, implies: &[F::Neon] },
    Info { name: "bf16", owners: A64, implies: &[F::Neon] },
    Info { name: "sve", owners: A64, implies: &[F::Fp16] },
    Info { name: "sve2", owners: A64, implies: &[F::Sve] },
    Info { name: "cuda", owners: ArchSet::ALL, implies: &[] },
];

/// `CLOSURE[f]` = bits of `f` and everything it implies, transitively.
const CLOSURE: [u64; N] = {
    let mut t = [0u64; N];
    let mut i = 0;
    while i < N {
        t[i] = 1 << i;
        i += 1;
    }
    let mut changed = true;
    while changed {
        changed = false;
        let mut f = 0;
        while f < N {
            let imp = INFO[f].implies;
            let mut j = 0;
            while j < imp.len() {
                let merged = t[f] | t[imp[j] as usize];
                if merged != t[f] {
                    t[f] = merged;
                    changed = true;
                }
                j += 1;
            }
            f += 1;
        }
    }
    t
};

const ALL_BITS: u64 = (1u64 << N) - 1;

impl Feature {
    pub const ALL: [Feature; N] = [
        F::Sse42,
        F::Avx,
        F::Avx2,
        F::Fma,
        F::Bmi2,
        F::F16c,
        F::Avx512f,
        F::Avx512bw,
        F::Avx512vl,
        F::Avx512dq,
        F::Avx512cd,
        F::Avx512vbmi,
        F::Avx512vbmi2,
        F::Avx512vpopcntdq,
        F::Avx512bitalg,
        F::Avx512fp16,
        F::Avx512bf16,
        F::Neon,
        F::Fp16,
        F::Dotprod,
        F::Bf16,
        F::Sve,
        F::Sve2,
        F::Cuda,
    ];

    fn info(self) -> &'static Info {
        &INFO[self as usize]
    }

    /// The name used in `needs(..)`, `dispatch` arms, and `--features`.
    pub fn name(self) -> &'static str {
        self.info().name
    }

    /// Architectures that own this feature.
    pub fn owners(self) -> ArchSet {
        self.info().owners
    }

    pub fn is_owned_by(self, a: Arch) -> bool {
        self.owners().contains(a)
    }

    /// Features this one implies directly (see the module doc).
    pub fn implies(self) -> &'static [Feature] {
        self.info().implies
    }

    pub fn from_name(s: &str) -> Option<Feature> {
        Feature::ALL.into_iter().find(|f| f.name() == s)
    }
}

impl fmt::Display for Feature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FeatureError {
    /// A name that is no feature.
    Unknown(String),
    /// An empty element in a comma list (`avx2,,fma`, or an empty list).
    Empty,
}

impl fmt::Display for FeatureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FeatureError::Unknown(s) => {
                write!(f, "unknown feature `{s}` (known: ")?;
                for (i, feat) in Feature::ALL.iter().enumerate() {
                    if i > 0 {
                        f.write_str(" ")?;
                    }
                    f.write_str(feat.name())?;
                }
                f.write_str(")")
            }
            FeatureError::Empty => f.write_str("empty feature name in list"),
        }
    }
}

impl std::error::Error for FeatureError {}

impl FromStr for Feature {
    type Err = FeatureError;
    fn from_str(s: &str) -> Result<Feature, FeatureError> {
        if s.is_empty() {
            return Err(FeatureError::Empty);
        }
        Feature::from_name(s).ok_or_else(|| FeatureError::Unknown(s.to_string()))
    }
}

/// A set of features, closed under implication.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct FeatureSet(u64);

impl FeatureSet {
    pub const EMPTY: FeatureSet = FeatureSet(0);

    fn close(bits: u64) -> u64 {
        let mut out = bits;
        for (i, c) in CLOSURE.iter().enumerate() {
            if bits & (1 << i) != 0 {
                out |= c;
            }
        }
        out
    }

    /// `f` and everything it implies.
    pub fn of(f: Feature) -> FeatureSet {
        FeatureSet(CLOSURE[f as usize])
    }

    pub fn from_features(fs: impl IntoIterator<Item = Feature>) -> FeatureSet {
        fs.into_iter().fold(FeatureSet::EMPTY, |s, f| s.with(f))
    }

    /// The closure of `bits`; `None` if a bit names no feature.
    pub fn from_bits(bits: u64) -> Option<FeatureSet> {
        if bits & !ALL_BITS != 0 { None } else { Some(FeatureSet(Self::close(bits))) }
    }

    pub fn bits(self) -> u64 {
        self.0
    }

    pub fn with(self, f: Feature) -> FeatureSet {
        FeatureSet(self.0 | CLOSURE[f as usize])
    }

    pub fn insert(&mut self, f: Feature) {
        *self = self.with(f);
    }

    /// Union (the union of closed sets is closed).
    pub fn union(self, o: FeatureSet) -> FeatureSet {
        FeatureSet(self.0 | o.0)
    }

    pub fn contains(self, f: Feature) -> bool {
        self.0 & (1 << f as u8) != 0
    }

    /// `self ⊆ o`: the subset rule of E0704 / E0707.
    pub fn is_subset(self, o: FeatureSet) -> bool {
        self.0 & !o.0 == 0
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub fn iter(self) -> impl Iterator<Item = Feature> {
        Feature::ALL.into_iter().filter(move |f| self.contains(*f))
    }

    /// Architectures owning every feature of the set (all of them for the empty set).
    pub fn owners(self) -> ArchSet {
        self.iter().fold(ArchSet::ALL, |a, f| a.intersect(f.owners()))
    }

    /// Every feature is owned by `a`: functions needing this set are compiled for `a`.
    pub fn is_owned_by(self, a: Arch) -> bool {
        self.iter().all(|f| f.is_owned_by(a))
    }

    /// The features not owned by `a`, in bit order.
    pub fn foreign_to(self, a: Arch) -> Vec<Feature> {
        self.iter().filter(|f| !f.is_owned_by(a)).collect()
    }

    /// Parse a `--features` value: comma-separated names (`avx2,fma`).
    pub fn parse_list(s: &str) -> Result<FeatureSet, FeatureError> {
        let mut set = FeatureSet::EMPTY;
        for part in s.split(',') {
            set.insert(part.trim().parse()?);
        }
        Ok(set)
    }
}

impl fmt::Display for FeatureSet {
    /// Names in bit order, comma-separated; the closure is printed in full.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, feat) in self.iter().enumerate() {
            if i > 0 {
                f.write_str(",")?;
            }
            f.write_str(feat.name())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(s: &str) -> FeatureSet {
        FeatureSet::parse_list(s).unwrap()
    }

    #[test]
    fn table_is_consistent() {
        assert_eq!(Feature::ALL.len(), INFO.len());
        for (i, f) in Feature::ALL.iter().enumerate() {
            assert_eq!(*f as usize, i);
            assert_eq!(Feature::from_name(f.name()), Some(*f));
            // An implication never crosses architectures.
            for g in f.implies() {
                assert_eq!(f.owners(), g.owners(), "{f} -> {g}");
            }
        }
    }

    #[test]
    fn implication_chains() {
        assert_eq!(set("avx2").to_string(), "sse42,avx,avx2");
        assert_eq!(set("sse42").to_string(), "sse42");
        assert!(set("avx512bw").contains(Feature::Avx512f));
        assert!(set("avx512bw").contains(Feature::Avx2));
        assert!(set("avx512bw").contains(Feature::Sse42));
        assert!(!set("avx512bw").contains(Feature::Avx512vl));
        assert!(set("avx512fp16").is_subset(set("avx512fp16")));
        assert!(set("avx512bw,avx512vl,avx512dq").is_subset(set("avx512fp16")));
        assert_eq!(set("sve2").to_string(), "neon,fp16,sve,sve2");
        assert_eq!(set("dotprod").to_string(), "neon,dotprod");
        assert_eq!(set("bmi2").to_string(), "bmi2");
        assert_eq!(set("cuda").to_string(), "cuda");
    }

    #[test]
    fn closure_is_idempotent_and_subset_rule() {
        for f in Feature::ALL {
            let s = FeatureSet::of(f);
            assert_eq!(FeatureSet::from_bits(s.bits()), Some(s));
            for g in s.iter() {
                assert!(FeatureSet::of(g).is_subset(s));
            }
        }
        assert!(FeatureSet::EMPTY.is_subset(set("avx")));
        assert!(!set("avx2").is_subset(set("avx")));
        assert_eq!(FeatureSet::from_bits(1 << 40), None);
        assert_eq!(FeatureSet::from_bits(1 << Feature::Avx2 as u8), Some(set("avx2")));
    }

    #[test]
    fn ownership() {
        assert!(set("avx2,fma").is_owned_by(Arch::X86_64));
        assert!(!set("avx2").is_owned_by(Arch::Aarch64));
        assert!(set("sve").is_owned_by(Arch::Aarch64));
        assert_eq!(set("avx2,sve").foreign_to(Arch::X86_64), [Feature::Neon, Feature::Fp16, Feature::Sve]);
        // cuda is owned by both host architectures and by gpu-sm86.
        for a in [Arch::X86_64, Arch::Aarch64, Arch::Gpu] {
            assert!(Feature::Cuda.is_owned_by(a));
        }
        assert!(!set("neon").is_owned_by(Arch::Gpu));
        assert_eq!(set("avx2,cuda").owners(), ArchSet::X86_64);
        assert_eq!(FeatureSet::EMPTY.owners(), ArchSet::ALL);
    }

    #[test]
    fn parse_errors() {
        assert_eq!(FeatureSet::parse_list("avx3"), Err(FeatureError::Unknown("avx3".into())));
        assert_eq!(FeatureSet::parse_list("AVX2"), Err(FeatureError::Unknown("AVX2".into())));
        assert_eq!(FeatureSet::parse_list(""), Err(FeatureError::Empty));
        assert_eq!(FeatureSet::parse_list("avx2,,fma"), Err(FeatureError::Empty));
        assert!(FeatureError::Unknown("x".into()).to_string().starts_with("unknown feature `x`"));
        assert_eq!(set("fma, bmi2"), set("bmi2,fma"));
    }
}
