//! Canonical renumbering.
//!
//! A canonical id is the rank of an entry under a stable sort by its key: identical in stage0
//! and stage1, whatever order the entries were interned in. [`rank_by_key`] computes the ranks as a
//! [`Remap`] (a permutation of `0..n`, so new ids are dense); [`Remap::apply`] rewrites every
//! id-holding array, and [`Remap::permute`] reorders the table itself.
//!
//! Keys must be unique for the result to be canonical: equal keys keep their old relative order
//! (stable sort), which is internal order. [`rank_by_unique_key`] rejects ties instead.
//!
//! Composite keys (types, decls) compare fields by *canonical* id, so rank tables bottom-up: rank
//! strings first, then pass the string remap into the next table's key function.

use std::fmt;

use crate::hash::xxh32;

/// An id type that is a `u32` underneath (typed ids in later crates implement this).
pub trait RawId: Copy {
    fn raw(self) -> u32;
    fn from_raw(raw: u32) -> Self;
}

impl RawId for u32 {
    fn raw(self) -> u32 {
        self
    }
    fn from_raw(raw: u32) -> Self {
        raw
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CanonError {
    /// More than `u32::MAX` entries.
    TooLarge(usize),
    /// Two entries (old ids, ascending) have equal keys.
    Tie(u32, u32),
    /// An id is not below the remap's length (position in the array, id).
    OutOfRange { index: usize, id: u32 },
    /// A table or remap has the wrong length (expected, got).
    LengthMismatch { expected: usize, got: usize },
    /// `from_old_to_new` input is not a permutation of `0..n`.
    NotPermutation,
}

impl fmt::Display for CanonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CanonError::TooLarge(n) => write!(f, "table of {n} entries exceeds u32 ids"),
            CanonError::Tie(a, b) => write!(f, "entries {a} and {b} have equal canonical keys"),
            CanonError::OutOfRange { index, id } => write!(f, "id {id} at position {index} is out of range"),
            CanonError::LengthMismatch { expected, got } => {
                write!(f, "length mismatch: expected {expected}, got {got}")
            }
            CanonError::NotPermutation => f.write_str("not a permutation"),
        }
    }
}

impl std::error::Error for CanonError {}

/// A bijection between old ids and new ids on `0..len`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Remap {
    old_to_new: Vec<u32>,
    new_to_old: Vec<u32>,
}

fn check_len(n: usize) -> Result<(), CanonError> {
    if n > u32::MAX as usize { Err(CanonError::TooLarge(n)) } else { Ok(()) }
}

/// Ranks of `table`'s entries under a stable sort by `key`: entry `i` gets new id
/// `remap.new_of(i)`. Ties keep the old order.
pub fn rank_by_key<T, K: Ord>(table: &[T], key: impl FnMut(&T) -> K) -> Result<Remap, CanonError> {
    let (remap, _) = rank_inner(table, key)?;
    Ok(remap)
}

/// [`rank_by_key`], failing with [`CanonError::Tie`] if two entries have equal keys.
pub fn rank_by_unique_key<T, K: Ord>(table: &[T], key: impl FnMut(&T) -> K) -> Result<Remap, CanonError> {
    let (remap, tie) = rank_inner(table, key)?;
    match tie {
        Some((a, b)) => Err(CanonError::Tie(a.min(b), a.max(b))),
        None => Ok(remap),
    }
}

fn rank_inner<T, K: Ord>(
    table: &[T],
    mut key: impl FnMut(&T) -> K,
) -> Result<(Remap, Option<(u32, u32)>), CanonError> {
    check_len(table.len())?;
    let mut keyed: Vec<(K, u32)> = table.iter().enumerate().map(|(i, t)| (key(t), i as u32)).collect();
    keyed.sort_by(|a, b| a.0.cmp(&b.0)); // stable
    let tie = keyed.windows(2).find(|w| w[0].0 == w[1].0).map(|w| (w[0].1, w[1].1));
    let new_to_old: Vec<u32> = keyed.into_iter().map(|(_, old)| old).collect();
    let mut old_to_new = vec![0u32; new_to_old.len()];
    for (new, &old) in new_to_old.iter().enumerate() {
        if let Some(slot) = old_to_new.get_mut(old as usize) {
            *slot = new as u32;
        }
    }
    Ok((Remap { old_to_new, new_to_old }, tie))
}

/// Canonical sort key of an interned string: `(xxHash32 seed 0, bytes)`.
pub fn string_rank_key(s: &str) -> (u32, &[u8]) {
    (xxh32(s.as_bytes(), 0), s.as_bytes())
}

impl Remap {
    pub fn identity(n: usize) -> Result<Remap, CanonError> {
        check_len(n)?;
        let v: Vec<u32> = (0..n as u32).collect();
        Ok(Remap { old_to_new: v.clone(), new_to_old: v })
    }

    /// From `old_to_new[old] = new`, which must be a permutation of `0..n`.
    pub fn from_old_to_new(old_to_new: Vec<u32>) -> Result<Remap, CanonError> {
        check_len(old_to_new.len())?;
        let mut new_to_old = vec![u32::MAX; old_to_new.len()];
        for (old, &new) in old_to_new.iter().enumerate() {
            match new_to_old.get_mut(new as usize) {
                Some(slot) if *slot == u32::MAX => *slot = old as u32,
                _ => return Err(CanonError::NotPermutation),
            }
        }
        Ok(Remap { old_to_new, new_to_old })
    }

    pub fn len(&self) -> usize {
        self.old_to_new.len()
    }

    pub fn is_empty(&self) -> bool {
        self.old_to_new.is_empty()
    }

    pub fn new_of(&self, old: u32) -> Option<u32> {
        self.old_to_new.get(old as usize).copied()
    }

    pub fn old_of(&self, new: u32) -> Option<u32> {
        self.new_to_old.get(new as usize).copied()
    }

    pub fn old_to_new(&self) -> &[u32] {
        &self.old_to_new
    }

    pub fn new_to_old(&self) -> &[u32] {
        &self.new_to_old
    }

    pub fn inverse(&self) -> Remap {
        Remap { old_to_new: self.new_to_old.clone(), new_to_old: self.old_to_new.clone() }
    }

    /// `self` then `next`: old id `i` maps to `next.new_of(self.new_of(i))`.
    pub fn then(&self, next: &Remap) -> Result<Remap, CanonError> {
        if next.len() != self.len() {
            return Err(CanonError::LengthMismatch { expected: self.len(), got: next.len() });
        }
        let mut ids = self.old_to_new.clone();
        next.apply(&mut ids)?;
        Remap::from_old_to_new(ids)
    }

    /// Rewrite every id in `ids` (old → new). All-or-nothing: on an out-of-range id nothing changes.
    pub fn apply<I: RawId>(&self, ids: &mut [I]) -> Result<(), CanonError> {
        self.apply_inner(ids, None)
    }

    /// [`Remap::apply`] leaving every `sentinel` (e.g. `u32::MAX` for "none") unchanged.
    pub fn apply_except<I: RawId>(&self, ids: &mut [I], sentinel: u32) -> Result<(), CanonError> {
        self.apply_inner(ids, Some(sentinel))
    }

    fn apply_inner<I: RawId>(&self, ids: &mut [I], sentinel: Option<u32>) -> Result<(), CanonError> {
        if let Some((index, id)) = ids
            .iter()
            .map(|i| i.raw())
            .enumerate()
            .find(|&(_, id)| Some(id) != sentinel && id as usize >= self.old_to_new.len())
        {
            return Err(CanonError::OutOfRange { index, id });
        }
        for slot in ids.iter_mut() {
            let id = slot.raw();
            if Some(id) == sentinel {
                continue;
            }
            if let Some(&new) = self.old_to_new.get(id as usize) {
                *slot = I::from_raw(new);
            }
        }
        Ok(())
    }

    /// Reorder a table indexed by old id into new-id order: `out[new] = table[old_of(new)]`.
    pub fn permute<T>(&self, table: Vec<T>) -> Result<Vec<T>, CanonError> {
        if table.len() != self.len() {
            return Err(CanonError::LengthMismatch { expected: self.len(), got: table.len() });
        }
        let mut slots: Vec<Option<T>> = table.into_iter().map(Some).collect();
        let mut out = Vec::with_capacity(slots.len());
        for &old in &self.new_to_old {
            match slots.get_mut(old as usize).and_then(Option::take) {
                Some(v) => out.push(v),
                None => return Err(CanonError::NotPermutation),
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn ranks_are_stable_sort_positions() {
        let t = ["d", "b", "a", "c"];
        let r = rank_by_key(&t, |s| *s).unwrap();
        assert_eq!(r.old_to_new(), &[3, 1, 0, 2]);
        assert_eq!(r.new_to_old(), &[2, 1, 3, 0]);
        assert_eq!(r.permute(t.to_vec()).unwrap(), ["a", "b", "c", "d"]);
    }

    #[test]
    fn ties_keep_old_order_or_are_rejected() {
        let t = [2, 1, 2, 0];
        let r = rank_by_key(&t, |x| *x).unwrap();
        assert_eq!(r.new_to_old(), &[3, 1, 0, 2]);
        assert_eq!(rank_by_unique_key(&t, |x| *x), Err(CanonError::Tie(0, 2)));
    }

    #[test]
    fn apply_rewrites_and_validates() {
        let r = Remap::from_old_to_new(vec![2, 0, 1]).unwrap();
        let mut ids = vec![0u32, 1, 2, 2];
        r.apply(&mut ids).unwrap();
        assert_eq!(ids, [2, 0, 1, 1]);
        let mut bad = vec![0u32, 3];
        assert_eq!(r.apply(&mut bad), Err(CanonError::OutOfRange { index: 1, id: 3 }));
        assert_eq!(bad, [0, 3], "all-or-nothing");
        let mut with_none = vec![u32::MAX, 1];
        r.apply_except(&mut with_none, u32::MAX).unwrap();
        assert_eq!(with_none, [u32::MAX, 0]);
    }

    #[test]
    fn from_old_to_new_rejects_non_permutations() {
        assert_eq!(Remap::from_old_to_new(vec![0, 0]), Err(CanonError::NotPermutation));
        assert_eq!(Remap::from_old_to_new(vec![0, 2]), Err(CanonError::NotPermutation));
        assert!(Remap::from_old_to_new(vec![]).unwrap().is_empty());
    }

    #[test]
    fn string_order_is_hash_then_bytes() {
        let t = ["b", "a", "abc", ""];
        let r = rank_by_unique_key(&t, |s| string_rank_key(s)).unwrap();
        let sorted = r.permute(t.to_vec()).unwrap();
        let mut keys: Vec<(u32, &[u8])> = sorted.iter().map(|s| string_rank_key(s)).collect();
        let before = keys.clone();
        keys.sort();
        assert_eq!(keys, before);
        // xxh32: "" 02cc5d05 < "abc" 32d153ff < "a" 550d7456; "b" sorts by its own hash.
        let pos = |s: &str| sorted.iter().position(|x| *x == s).unwrap();
        assert!(pos("") < pos("abc") && pos("abc") < pos("a"));
    }

    proptest! {
        #[test]
        fn canonical_regardless_of_insertion_order(mut v in proptest::collection::vec(any::<u16>(), 0..200),
                                                   seed in any::<u64>()) {
            v.sort_unstable();
            v.dedup();
            // Two "intern orders" of the same set: original and a seeded shuffle.
            let mut w = v.clone();
            let mut s = seed | 1;
            for i in (1..w.len()).rev() {
                s ^= s << 13; s ^= s >> 7; s ^= s << 17;
                w.swap(i, (s % (i as u64 + 1)) as usize);
            }
            let ra = rank_by_unique_key(&v, |x| *x).unwrap();
            let rb = rank_by_unique_key(&w, |x| *x).unwrap();
            prop_assert_eq!(ra.permute(v.clone()).unwrap(), rb.permute(w.clone()).unwrap());
            // apply then inverse is the identity on every id array.
            let mut ids: Vec<u32> = (0..w.len() as u32).rev().collect();
            let orig = ids.clone();
            rb.apply(&mut ids).unwrap();
            rb.inverse().apply(&mut ids).unwrap();
            prop_assert_eq!(ids, orig);
            let id = Remap::identity(w.len()).unwrap();
            prop_assert_eq!(rb.then(&id).unwrap(), rb.clone());
            prop_assert_eq!(rb.then(&rb.inverse()).unwrap(), id);
        }
    }
}
