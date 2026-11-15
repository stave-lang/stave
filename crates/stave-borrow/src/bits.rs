//! Fixed-width bitsets (origin sets, move-site sets, live-local sets).

/// Words stored in place: sets of up to `64 * INLINE` bits need no heap allocation (the sets of
/// most bodies; the dataflow creates one per operand and statement).
const INLINE: usize = 2;

/// A bitset of a fixed number of 64-bit words.
///
/// Invariant (equality and hashing rely on it): with at most [`INLINE`] words they are in
/// `small`, the unused words of `small` are zero, and `big` is empty; otherwise all words are in
/// `big` and `small` is zero.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Bits {
    words: u32,
    small: [u64; INLINE],
    big: Vec<u64>,
}

impl Bits {
    /// An empty set over `n` bits.
    pub fn new(n: usize) -> Bits {
        let words = n.div_ceil(64);
        let big = if words > INLINE { vec![0; words] } else { Vec::new() };
        Bits { words: words as u32, small: [0; INLINE], big }
    }

    #[inline]
    fn w(&self) -> &[u64] {
        if self.big.is_empty() { &self.small[..self.words as usize] } else { &self.big }
    }

    #[inline]
    fn w_mut(&mut self) -> &mut [u64] {
        if self.big.is_empty() { &mut self.small[..self.words as usize] } else { &mut self.big }
    }

    #[inline]
    pub fn insert(&mut self, i: usize) -> bool {
        let (k, b) = (i / 64, 1u64 << (i % 64));
        match self.w_mut().get_mut(k) {
            Some(x) => {
                let old = *x;
                *x |= b;
                old != *x
            }
            None => false,
        }
    }

    #[inline]
    pub fn remove(&mut self, i: usize) {
        if let Some(x) = self.w_mut().get_mut(i / 64) {
            *x &= !(1u64 << (i % 64));
        }
    }

    #[inline]
    pub fn contains(&self, i: usize) -> bool {
        self.w().get(i / 64).is_some_and(|x| x & (1u64 << (i % 64)) != 0)
    }

    /// `self |= o`; whether `self` changed.
    pub fn union_with(&mut self, o: &Bits) -> bool {
        let mut ch = false;
        for (a, b) in self.w_mut().iter_mut().zip(o.w()) {
            let n = *a | *b;
            ch |= n != *a;
            *a = n;
        }
        ch
    }

    /// `self &= !o`.
    pub fn subtract(&mut self, o: &Bits) {
        for (a, b) in self.w_mut().iter_mut().zip(o.w()) {
            *a &= !*b;
        }
    }

    pub fn is_subset(&self, o: &Bits) -> bool {
        self.w().iter().zip(o.w()).all(|(a, b)| a & !b == 0)
    }

    pub fn is_empty(&self) -> bool {
        self.w().iter().all(|&x| x == 0)
    }

    pub fn clear(&mut self) {
        self.w_mut().iter_mut().for_each(|x| *x = 0);
    }

    /// Set bits, ascending.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.w().iter().enumerate().flat_map(|(k, &x)| {
            let mut x = x;
            std::iter::from_fn(move || {
                if x == 0 {
                    return None;
                }
                let t = x.trailing_zeros() as usize;
                x &= x - 1;
                Some(k * 64 + t)
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::Bits;

    #[test]
    fn ops() {
        let mut a = Bits::new(130);
        assert!(a.insert(3) && a.insert(129) && !a.insert(3));
        assert!(!a.insert(500));
        let mut b = Bits::new(130);
        b.insert(64);
        assert!(a.union_with(&b) && !a.union_with(&b));
        assert_eq!(a.iter().collect::<Vec<_>>(), vec![3, 64, 129]);
        assert!(b.is_subset(&a) && !a.is_subset(&b));
        a.subtract(&b);
        assert!(!a.contains(64) && a.contains(129));
        a.remove(3);
        a.remove(129);
        assert!(a.is_empty());
    }

    #[test]
    fn in_place_and_heap_sets_agree() {
        for n in [0usize, 1, 64, 65, 128, 129, 1000] {
            let mut a = Bits::new(n);
            assert!(a.is_empty() && a.iter().next().is_none(), "{n}");
            assert!(!a.insert(64 * n.div_ceil(64)) && a.is_empty(), "{n}: past the last word");
            let want: Vec<usize> = (0..n).filter(|i| i % 7 == 0 || i + 1 == n).collect();
            for &i in &want {
                assert!(a.insert(i));
            }
            assert_eq!(a.iter().collect::<Vec<_>>(), want, "{n}");
            let mut b = Bits::new(n);
            assert!(b.union_with(&a) == !want.is_empty() && b == a, "{n}");
            b.clear();
            assert!(b.is_empty() && b == Bits::new(n) && b.is_subset(&a), "{n}");
            a.subtract(&a.clone());
            assert_eq!(a, Bits::new(n), "{n}");
        }
        assert_ne!(Bits::new(64), Bits::new(128));
    }
}
