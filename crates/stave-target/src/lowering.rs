//! SIMD lowering tables: how each portable op lowers on each target tier (class, cost, recipe).
//!
//! The source is the declarative tables `tables/x86.toml` and `tables/a64.toml`; `tools/gen-tables`
//! generates `generated/x86.rs` and `generated/a64.rs` from them. contracts/simd-ops.md defines
//! the classes, and this crate's tests check the tables against its operation and chunk-loop
//! tables.
//!
//! An [`Entry`] is one (op, type class, form). `op` is the IR name (`vec.*`, `mask.*`); chunk-loop
//! constructs are `loop.*`, and memory hints and ops without an IR op are named by their method
//! (`mem.prefetch_read_l1`, `vec.store_stream`). `keys` are the concrete types the entry covers:
//! lane types (`i32`, `f64`), masks (`m16`), casts (`i32>f32`). `form` is the shift amount
//! (`const`, `vector`) or empty. Each tier has one baseline cell and optional cells that need a
//! feature beyond the tier baseline. A recipe is the instruction sequence for one full-width
//! vector of the tier, `; ` separated: `recipe` keeps the table's placeholders (`*` lane size
//! letter, `#` lane bits, `%` float letter, `?` s or u), `recipes` has it expanded per key. The
//! cost is the number of instructions in the recipe.

use crate::features::{Feature, FeatureSet};
use crate::triple::{A64Ext, Triple, X86Level};

#[rustfmt::skip]
#[path = "generated/x86.rs"]
mod x86;

#[rustfmt::skip]
#[path = "generated/a64.rs"]
mod a64;

/// Lowering class (simd-ops.md): native (at most two instructions), multi-instruction, unavailable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Class {
    N,
    M,
    U,
}

impl Class {
    pub fn name(self) -> &'static str {
        match self {
            Class::N => "N",
            Class::M => "M",
            Class::U => "U",
        }
    }
}

/// A target tier: the feature baseline a table column describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Tier {
    V2,
    V3,
    V4,
    Neon,
    Sve,
}

impl Tier {
    pub const ALL: [Tier; 5] = [Tier::V2, Tier::V3, Tier::V4, Tier::Neon, Tier::Sve];

    pub fn name(self) -> &'static str {
        match self {
            Tier::V2 => "v2",
            Tier::V3 => "v3",
            Tier::V4 => "v4",
            Tier::Neon => "neon",
            Tier::Sve => "sve",
        }
    }

    /// The tier of a CPU target (`aarch64-linux+sve2` uses the sve column); `None` for `gpu-sm86`,
    /// whose table is gpu-ops.md.
    pub fn of(t: Triple) -> Option<Tier> {
        Some(match t {
            Triple::X86_64(X86Level::V2) => Tier::V2,
            Triple::X86_64(X86Level::V3) => Tier::V3,
            Triple::X86_64(X86Level::V4) => Tier::V4,
            Triple::Aarch64(A64Ext::Base) => Tier::Neon,
            Triple::Aarch64(A64Ext::Sve | A64Ext::Sve2) => Tier::Sve,
            Triple::GpuSm86 => return None,
        })
    }

    /// The target whose baseline is this tier.
    pub fn triple(self) -> Triple {
        match self {
            Tier::V2 => Triple::X86_64(X86Level::V2),
            Tier::V3 => Triple::X86_64(X86Level::V3),
            Tier::V4 => Triple::X86_64(X86Level::V4),
            Tier::Neon => Triple::Aarch64(A64Ext::Base),
            Tier::Sve => Triple::Aarch64(A64Ext::Sve),
        }
    }

    /// Every entry of this tier's architecture, in table order.
    pub fn entries(self) -> &'static [Entry] {
        match self {
            Tier::V2 | Tier::V3 | Tier::V4 => x86::ENTRIES,
            Tier::Neon | Tier::Sve => a64::ENTRIES,
        }
    }
}

/// One tier's lowering of an entry.
#[derive(Debug)]
pub struct Cell {
    pub tier: Tier,
    /// The feature this cell needs beyond the tier baseline; `None` for the baseline cell.
    pub feature: Option<&'static str>,
    pub class: Class,
    pub cost: u32,
    /// With placeholders; empty for U.
    pub recipe: &'static str,
    /// Expanded, one per key of the entry.
    pub recipes: &'static [&'static str],
    pub note: &'static str,
    pub unverified: bool,
}

#[derive(Debug)]
pub struct Entry {
    pub op: &'static str,
    /// The type class as written in the table (`int8/int16`, `float`, `i8>f32`).
    pub ty: &'static str,
    pub form: &'static str,
    pub keys: &'static [&'static str],
    pub cells: &'static [Cell],
}

impl Entry {
    /// The baseline cell of `tier` (every entry has one per tier of its architecture).
    pub fn baseline(&self, tier: Tier) -> Option<&'static Cell> {
        self.cells.iter().find(|c| c.tier == tier && c.feature.is_none())
    }

    /// The cells of `tier` that need an extra feature.
    pub fn variants(&self, tier: Tier) -> impl Iterator<Item = &'static Cell> + use<> {
        self.cells.iter().filter(move |c| c.tier == tier && c.feature.is_some())
    }
}

/// A lowering for one concrete key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lowering {
    pub class: Class,
    pub cost: u32,
    /// Expanded for the key; empty for U.
    pub recipe: &'static str,
    pub feature: Option<&'static str>,
    pub note: &'static str,
    pub unverified: bool,
}

impl Lowering {
    fn of(c: &'static Cell, i: usize) -> Lowering {
        Lowering {
            class: c.class,
            cost: c.cost,
            recipe: c.recipes.get(i).copied().unwrap_or(""),
            feature: c.feature,
            note: c.note,
            unverified: c.unverified,
        }
    }
}

/// The entry covering (`op`, `key`, `form`) on `tier`'s architecture, and the key's index in it.
pub fn find(op: &str, key: &str, form: &str, tier: Tier) -> Option<(&'static Entry, usize)> {
    tier.entries()
        .iter()
        .filter(|e| e.op == op && e.form == form)
        .find_map(|e| e.keys.iter().position(|k| *k == key).map(|i| (e, i)))
}

/// The baseline lowering of (`op`, `key`, `form`) on `tier`.
pub fn lookup(op: &str, key: &str, form: &str, tier: Tier) -> Option<Lowering> {
    let (e, i) = find(op, key, form, tier)?;
    e.baseline(tier).map(|c| Lowering::of(c, i))
}

/// The lowering with `features` (a superset of the tier baseline): the first cell whose feature is
/// in `features`, else the baseline.
pub fn lookup_with(op: &str, key: &str, form: &str, tier: Tier, features: FeatureSet) -> Option<Lowering> {
    let (e, i) = find(op, key, form, tier)?;
    let has = |f: &str| Feature::from_name(f).is_some_and(|f| features.contains(f));
    e.variants(tier)
        .find(|c| c.feature.is_some_and(has))
        .or_else(|| e.baseline(tier))
        .map(|c| Lowering::of(c, i))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookups() {
        let l = lookup("vec.add", "f64", "", Tier::V3).unwrap();
        assert_eq!((l.class, l.cost, l.recipe), (Class::N, 1, "vaddpd"));
        assert_eq!(lookup("vec.fma", "f32", "", Tier::V2).unwrap().class, Class::U);
        assert_eq!(lookup("vec.gather", "u32", "", Tier::Sve).unwrap().recipe, "ld1w (vector index)");
        assert_eq!(lookup("vec.shl", "i16", "vector", Tier::V4).unwrap().recipe, "vpsllvw");
        assert!(lookup("vec.shl", "i16", "", Tier::V4).is_none());
        assert!(lookup("vec.add", "f32", "", Tier::Neon).is_some());
        let base = Tier::V4.triple().baseline();
        assert_eq!(lookup_with("vec.compress", "i8", "", Tier::V4, base).unwrap().class, Class::M);
        let vbmi2 = base.with(Feature::Avx512vbmi2);
        let l = lookup_with("vec.compress", "i8", "", Tier::V4, vbmi2).unwrap();
        assert_eq!((l.class, l.recipe, l.feature), (Class::N, "vpcompressb {z}", Some("avx512vbmi2")));
    }

    #[test]
    fn tiers() {
        for t in Tier::ALL {
            assert_eq!(Tier::of(t.triple()), Some(t));
        }
        assert_eq!(Tier::of(Triple::Aarch64(A64Ext::Sve2)), Some(Tier::Sve));
        assert_eq!(Tier::of(Triple::GpuSm86), None);
    }

    #[test]
    fn every_entry_has_one_baseline_per_tier_and_known_features() {
        for t in Tier::ALL {
            for e in t.entries() {
                assert!(e.baseline(t).is_some(), "{} {}: no {} cell", e.op, e.ty, t.name());
                for c in e.cells {
                    assert_eq!(c.recipes.len(), e.keys.len(), "{} {}", e.op, e.ty);
                    if let Some(f) = c.feature {
                        let f =
                            Feature::from_name(f).unwrap_or_else(|| panic!("{}: unknown feature {f}", e.op));
                        assert!(f.is_owned_by(t.triple().arch()), "{}: {f} is foreign", e.op);
                        assert!(
                            !t.triple().baseline().contains(f),
                            "{}: {f} is in the {} baseline",
                            e.op,
                            t.name()
                        );
                    }
                }
            }
        }
    }
}
