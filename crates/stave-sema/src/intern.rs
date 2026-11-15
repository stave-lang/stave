//! Interned strings (canonical ids, stage-gates `strings`).
//!
//! The interned set: the text of every `Ident` token of every source file (dropped items
//! included), every identifier in an f-string hole, every package name, and every module path
//! segment. Ids are canonical: the rank of (xxHash32 seed 0 of the bytes, bytes).
//!
//! Interning runs per file in parallel: a file first dedups its identifiers locally (no locks),
//! then interns the unique ones in the shared sharded pool. The raw pool ids depend on thread
//! timing; [`canonicalize`] ranks them and rewrites every per-token array.

use std::sync::Arc;

use rustc_hash::FxHashMap;
use stave_intern::{StringPool, rank_by_unique_key};

use crate::ids::Sym;

/// The canonical string table.
#[derive(Clone, Debug, Default)]
pub struct Strings {
    strs: Vec<Arc<str>>,
    hashes: Vec<u32>,
    ids: FxHashMap<Arc<str>, Sym>,
}

impl Strings {
    /// The string of `s` (`""` for an unknown id).
    pub fn get(&self, s: Sym) -> &str {
        self.strs.get(s.idx()).map_or("", |a| a)
    }

    /// xxHash32 (seed 0) of the bytes.
    pub fn hash(&self, s: Sym) -> u32 {
        self.hashes.get(s.idx()).copied().unwrap_or(0)
    }

    pub fn lookup(&self, text: &str) -> Option<Sym> {
        self.ids.get(text).copied()
    }

    pub fn len(&self) -> usize {
        self.strs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.strs.is_empty()
    }

    /// Every string, by id.
    pub fn iter(&self) -> impl Iterator<Item = (Sym, &str)> {
        self.strs.iter().enumerate().map(|(i, s)| (Sym(i as u32), &**s))
    }
}

/// The intern phase's pool (raw ids).
pub(crate) struct Pool {
    pool: StringPool,
}

impl Pool {
    pub fn new() -> Pool {
        Pool { pool: StringPool::new() }
    }

    /// Raw id of `s`. The pool holds at most `u32::MAX - 1` strings, far above any input with
    /// u32 offsets; on overflow `Sym::NONE` is returned (the name then resolves to nothing).
    pub fn intern(&self, s: &str) -> Sym {
        self.pool.intern_str(s).map(Sym).unwrap_or(Sym::NONE)
    }

    /// Ranks the pool canonically: (table, `old_to_new[raw] = canonical`).
    pub fn canonicalize(self) -> (Strings, Vec<u32>) {
        let keys: Vec<Arc<str>> = self.pool.freeze().keys().to_vec();
        // Keys of an interner are unique and fewer than u32::MAX, so ranking cannot fail; the
        // fallback only keeps this total.
        let Ok(remap) =
            rank_by_unique_key(&keys, |s| (stave_intern::hash::xxh32(s.as_bytes(), 0), s.clone()))
        else {
            return (Strings::default(), Vec::new());
        };
        let old_to_new = remap.old_to_new().to_vec();
        let strs = remap.permute(keys).unwrap_or_default();
        let hashes = strs.iter().map(|s| stave_intern::hash::xxh32(s.as_bytes(), 0)).collect();
        let ids = strs.iter().enumerate().map(|(i, s)| (s.clone(), Sym(i as u32))).collect();
        (Strings { strs, hashes, ids }, old_to_new)
    }
}

/// Rewrites raw ids to canonical ones (`Sym::NONE` stays).
pub(crate) fn remap_syms(old_to_new: &[u32], syms: &mut [Sym]) {
    for s in syms.iter_mut() {
        if s.is_some() {
            *s = old_to_new.get(s.idx()).map_or(Sym::NONE, |&n| Sym(n));
        }
    }
}

/// Small deterministic PRNG (xorshift64*) for the shuffle test knob.
pub(crate) fn shuffle<T>(v: &mut [T], seed: u64) {
    let mut s = seed | 1;
    for i in (1..v.len()).rev() {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        let r = s.wrapping_mul(0x2545_f491_4f6c_dd1d);
        v.swap(i, (r % (i as u64 + 1)) as usize);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_table_ignores_intern_order() {
        let words = ["b", "a", "zz", "main", "x1", "_"];
        let mut tables = Vec::new();
        let mut raws = Vec::new();
        for seed in [1u64, 2, 3, 99] {
            let mut w = words.to_vec();
            shuffle(&mut w, seed);
            let p = Pool::new();
            let raw: Vec<Sym> = w.iter().map(|s| p.intern(s)).collect();
            let (t, old_to_new) = p.canonicalize();
            let mut canon = raw.clone();
            remap_syms(&old_to_new, &mut canon);
            for (k, s) in w.iter().enumerate() {
                assert_eq!(t.get(canon[k]), *s);
            }
            raws.push(w);
            tables.push(t.iter().map(|(i, s)| (i, s.to_string())).collect::<Vec<_>>());
        }
        assert!(raws.windows(2).any(|w| w[0] != w[1]), "the shuffle must change the intern order");
        assert!(tables.windows(2).all(|w| w[0] == w[1]));
        // Order: (xxh32, bytes).
        let t = &tables[0];
        let keys: Vec<(u32, String)> =
            t.iter().map(|(_, s)| (stave_intern::hash::xxh32(s.as_bytes(), 0), s.clone())).collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted);
    }
}
