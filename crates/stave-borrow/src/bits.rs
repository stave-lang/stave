//! Fixed-width bitsets (origin sets, move-site sets, live-local sets).

/// A bitset of a fixed number of 64-bit words.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Bits {
    w: Vec<u64>,
}

impl Bits {
    /// An empty set over `n` bits.
    pub fn new(n: usize) -> Bits {
        Bits { w: vec![0; n.div_ceil(64)] }
    }

    #[inline]
    pub fn insert(&mut self, i: usize) -> bool {
        let (k, b) = (i / 64, 1u64 << (i % 64));
        match self.w.get_mut(k) {
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
        if let Some(x) = self.w.get_mut(i / 64) {
            *x &= !(1u64 << (i % 64));
        }
    }

    #[inline]
    pub fn contains(&self, i: usize) -> bool {
        self.w.get(i / 64).is_some_and(|x| x & (1u64 << (i % 64)) != 0)
    }

    /// `self |= o`; whether `self` changed.
    pub fn union_with(&mut self, o: &Bits) -> bool {
        let mut ch = false;
        for (a, b) in self.w.iter_mut().zip(&o.w) {
            let n = *a | *b;
            ch |= n != *a;
            *a = n;
        }
        ch
    }

    /// `self &= !o`.
    pub fn subtract(&mut self, o: &Bits) {
        for (a, b) in self.w.iter_mut().zip(&o.w) {
            *a &= !*b;
        }
    }

    pub fn is_subset(&self, o: &Bits) -> bool {
        self.w.iter().zip(&o.w).all(|(a, b)| a & !b == 0)
    }

    pub fn is_empty(&self) -> bool {
        self.w.iter().all(|&x| x == 0)
    }

    pub fn clear(&mut self) {
        self.w.iter_mut().for_each(|x| *x = 0);
    }

    /// Set bits, ascending.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.w.iter().enumerate().flat_map(|(k, &x)| {
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
}
