//! Sparse origin sets and the table of the distinct ones.
//!
//! An origin set names a few atoms out of all the loans of a body: it is kept as the ascending
//! list of its atoms, so its size and the cost of every operation follow what it holds, not
//! the number of loans.

use std::hash::{Hash, Hasher};

use rustc_hash::{FxHashMap, FxHasher};

/// Atoms stored in place: a set of up to `INLINE` atoms needs no heap allocation.
const INLINE: usize = 4;

/// A set of atoms below a fixed bound, as an ascending list.
///
/// Invariant (equality and hashing rely on it): with at most [`INLINE`] atoms they are in
/// `small[..len]`, the rest of `small` is zero, and `big` is empty; otherwise all atoms are in
/// `big` and `small` is zero.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Set {
    /// Atoms at or past it are never members: inserting one does nothing.
    cap: u32,
    len: u32,
    small: [u32; INLINE],
    big: Vec<u32>,
}

impl Set {
    /// An empty set over `n` atoms. The bound is `n` rounded up to a multiple of 64.
    pub fn new(n: usize) -> Set {
        Set { cap: (n.div_ceil(64) * 64) as u32, len: 0, small: [0; INLINE], big: Vec::new() }
    }

    /// The atoms, ascending.
    #[inline]
    pub fn atoms(&self) -> &[u32] {
        if self.len as usize <= INLINE { &self.small[..self.len as usize] } else { &self.big }
    }

    /// Replaces the atoms by `v` (ascending, distinct, below the bound).
    fn set(&mut self, v: &[u32]) {
        self.len = v.len() as u32;
        self.small = [0; INLINE];
        self.big.clear();
        if v.len() <= INLINE {
            self.small[..v.len()].copy_from_slice(v);
        } else {
            self.big.extend_from_slice(v);
        }
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn clear(&mut self) {
        self.set(&[]);
    }

    #[inline]
    pub fn contains(&self, i: usize) -> bool {
        let a = self.atoms();
        if a.len() <= INLINE { a.iter().any(|&x| x as usize == i) } else { self.find(i).is_ok() }
    }

    /// The position of atom `i`, or where it would be inserted.
    fn find(&self, i: usize) -> Result<usize, usize> {
        self.atoms().binary_search_by(|&x| (x as usize).cmp(&i))
    }

    /// Adds atom `i`; whether the set changed.
    pub fn insert(&mut self, i: usize) -> bool {
        if i >= self.cap as usize {
            return false;
        }
        let Err(at) = self.find(i) else { return false };
        let n = self.len as usize;
        if n < INLINE {
            self.small.copy_within(at..n, at + 1);
            self.small[at] = i as u32;
        } else {
            if n == INLINE {
                self.big.extend_from_slice(&self.small);
                self.small = [0; INLINE];
            }
            self.big.insert(at, i as u32);
        }
        self.len += 1;
        true
    }

    pub fn remove(&mut self, i: usize) {
        let Ok(at) = self.find(i) else { return };
        let n = self.len as usize;
        if n <= INLINE {
            self.small.copy_within(at + 1..n, at);
            self.small[n - 1] = 0;
        } else {
            self.big.remove(at);
            if n - 1 == INLINE {
                self.small.copy_from_slice(&self.big);
                self.big.clear();
            }
        }
        self.len -= 1;
    }

    /// `self |= o`; whether `self` changed.
    pub fn union_with(&mut self, o: &Set) -> bool {
        if o.is_subset(self) {
            return false;
        }
        let (a, b) = (self.atoms(), o.atoms());
        // the merged list: in place while it is short
        let mut short = [0u32; 2 * INLINE];
        let mut long: Vec<u32> = Vec::new();
        let is_short = a.len() + b.len() <= short.len();
        if !is_short {
            long.reserve(a.len() + b.len());
        }
        let (mut i, mut j, mut n) = (0, 0, 0);
        while i < a.len() || j < b.len() {
            let x = if j == b.len() || (i < a.len() && a[i] <= b[j]) {
                if j < b.len() && a[i] == b[j] {
                    j += 1;
                }
                i += 1;
                a[i - 1]
            } else {
                j += 1;
                b[j - 1]
            };
            if x >= self.cap {
                continue;
            }
            if is_short {
                short[n] = x;
            } else {
                long.push(x);
            }
            n += 1;
        }
        let changed = n != a.len();
        if is_short {
            self.set(&short[..n]);
        } else {
            self.set(&long);
        }
        changed
    }

    /// `self &= !o`.
    pub fn subtract(&mut self, o: &Set) {
        if o.is_empty() || !self.atoms().iter().any(|&x| o.contains(x as usize)) {
            return;
        }
        let v: Vec<u32> = self.atoms().iter().copied().filter(|&x| !o.contains(x as usize)).collect();
        self.set(&v);
    }

    pub fn is_subset(&self, o: &Set) -> bool {
        let (a, b) = (self.atoms(), o.atoms());
        if a.len() > b.len() {
            return false;
        }
        let mut j = 0;
        for &x in a {
            while j < b.len() && b[j] < x {
                j += 1;
            }
            if j == b.len() || b[j] != x {
                return false;
            }
            j += 1;
        }
        true
    }

    /// The atoms, ascending.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.atoms().iter().map(|&x| x as usize)
    }
}

/// The distinct sets a body's dataflow states hold, each under an id. A state names a set by its
/// id, so equal sets are stored once and compare by id. Id 0 is the empty set.
pub struct SetTable {
    sets: Vec<Set>,
    /// The hash of a set -> the id of the last set entered with that hash.
    by_hash: FxHashMap<u64, u32>,
    /// Per id: the id of the set entered before it with the same hash (`u32::MAX`: none).
    same_hash: Vec<u32>,
    /// The atoms of the sets looked up or joined so far.
    work: u64,
}

impl SetTable {
    /// A table whose sets are over `n` atoms.
    pub fn new(n: usize) -> SetTable {
        SetTable {
            sets: vec![Set::new(n)],
            by_hash: FxHashMap::default(),
            same_hash: vec![u32::MAX],
            work: 0,
        }
    }

    /// The atoms of the sets looked up or joined so far.
    pub fn work(&self) -> u64 {
        self.work
    }

    #[inline]
    pub fn get(&self, id: u32) -> &Set {
        &self.sets[id as usize]
    }

    /// The id of set `s`, entering it if it is new.
    pub fn id(&mut self, s: &Set) -> u32 {
        if s.is_empty() {
            return 0;
        }
        self.work += s.len() as u64;
        let mut h = FxHasher::default();
        s.atoms().hash(&mut h);
        let h = h.finish();
        let first = self.by_hash.get(&h).copied().unwrap_or(u32::MAX);
        let mut k = first;
        while k != u32::MAX {
            if self.sets[k as usize] == *s {
                return k;
            }
            k = self.same_hash[k as usize];
        }
        let id = self.sets.len() as u32;
        self.sets.push(s.clone());
        self.same_hash.push(first);
        self.by_hash.insert(h, id);
        id
    }

    /// Whether set `a` holds every atom of set `b`.
    pub fn covers(&self, a: u32, b: u32) -> bool {
        a == b || b == 0 || (a != 0 && self.sets[b as usize].is_subset(&self.sets[a as usize]))
    }

    /// The id of the union of sets `a` and `b`.
    pub fn union(&mut self, a: u32, b: u32) -> u32 {
        if a == b || b == 0 {
            return a;
        }
        if a == 0 {
            return b;
        }
        self.work += (self.sets[a as usize].len() + self.sets[b as usize].len()) as u64;
        if self.sets[b as usize].is_subset(&self.sets[a as usize]) {
            return a;
        }
        let mut s = self.sets[a as usize].clone();
        s.union_with(&self.sets[b as usize]);
        self.id(&s)
    }
}

#[cfg(test)]
mod tests {
    use super::{Set, SetTable};
    use crate::bits::Bits;

    /// A small deterministic generator.
    fn next(seed: &mut u64) -> usize {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 33) as usize
    }

    #[test]
    fn agrees_with_the_bitset() {
        let mut seed = 7u64;
        for n in [0usize, 1, 3, 64, 65, 130, 1000] {
            let mut sets = vec![Set::new(n); 4];
            let mut bits = vec![Bits::new(n); 4];
            for step in 0..4000 {
                let (a, b) = (next(&mut seed) % 4, next(&mut seed) % 4);
                // past the bound too: the last word's unused bits, and beyond it
                let i = next(&mut seed) % (n + 70);
                match next(&mut seed) % 8 {
                    0 | 1 => assert_eq!(sets[a].insert(i), bits[a].insert(i), "{n} {step}"),
                    2 => {
                        sets[a].remove(i);
                        bits[a].remove(i);
                    }
                    3 if a != b => {
                        let (s, t) = (sets[b].clone(), bits[b].clone());
                        assert_eq!(sets[a].union_with(&s), bits[a].union_with(&t), "{n} {step}");
                    }
                    4 if a != b => {
                        let (s, t) = (sets[b].clone(), bits[b].clone());
                        sets[a].subtract(&s);
                        bits[a].subtract(&t);
                    }
                    5 if step % 50 == 0 => {
                        sets[a].clear();
                        bits[a].clear();
                    }
                    _ => {}
                }
                assert_eq!(sets[a].contains(i), bits[a].contains(i), "{n} {step}");
                assert_eq!(sets[a].is_subset(&sets[b]), bits[a].is_subset(&bits[b]), "{n} {step}");
                assert_eq!(sets[a].is_empty(), bits[a].is_empty(), "{n} {step}");
                assert_eq!(sets[a].len(), bits[a].iter().count(), "{n} {step}");
                assert!(sets[a].iter().eq(bits[a].iter()), "{n} {step}");
                assert_eq!(sets[a] == sets[b], bits[a] == bits[b], "{n} {step}");
            }
        }
    }

    #[test]
    fn equal_sets_share_an_id() {
        let mut t = SetTable::new(200);
        let mut a = Set::new(200);
        assert_eq!(t.id(&a), 0);
        for i in [5usize, 9, 70, 130, 131, 199] {
            a.insert(i);
        }
        let mut b = Set::new(200);
        for i in [199usize, 131, 130, 70, 9, 5] {
            b.insert(i);
        }
        let (x, y) = (t.id(&a), t.id(&b));
        assert!(x != 0 && x == y);
        b.remove(9);
        let z = t.id(&b);
        assert!(z != x && t.get(z) == &b);
        assert_eq!(t.union(x, z), x);
        assert_eq!(t.union(z, x), x);
        assert_eq!(t.union(0, z), z);
        let mut c = Set::new(200);
        c.insert(1);
        let w = t.id(&c);
        let u = t.union(z, w);
        assert_eq!(t.get(u).iter().collect::<Vec<_>>(), vec![1, 5, 70, 130, 131, 199]);
    }
}
