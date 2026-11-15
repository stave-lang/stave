//! Deterministic random numbers. Only integer operations and IEEE `+ - * /` are used, so the
//! output is byte-identical on every host (no libm calls).

/// SplitMix64.
pub struct Rng(u64);

fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    /// An independent stream for `(seed, a, b)`.
    pub fn derive(seed: u64, a: u64, b: u64) -> Rng {
        Rng(mix64(mix64(seed ^ 0x51_7CC1_B727_220A) ^ mix64(a.wrapping_add(0x2545_F491_4F6C_DD1D)) ^ b))
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix64(self.0)
    }

    /// Uniform in `0..n` (0 when `n == 0`).
    pub fn below(&mut self, n: u32) -> u32 {
        (((self.next_u64() >> 32) * u64::from(n)) >> 32) as u32
    }

    /// Uniform in `lo..=hi` (`lo` when `hi < lo`).
    pub fn range(&mut self, lo: u32, hi: u32) -> u32 {
        if hi <= lo { lo } else { lo + self.below(hi - lo + 1) }
    }

    pub fn chance(&mut self, permille: u32) -> bool {
        self.below(1000) < permille
    }

    /// Index into `w` with probability proportional to the weight (0 when all are zero).
    pub fn weighted(&mut self, w: &[u32]) -> usize {
        let total: u64 = w.iter().map(|&x| u64::from(x)).sum();
        if total == 0 {
            return 0;
        }
        let mut r = ((self.next_u64() >> 1) % total) as i64;
        for (i, &x) in w.iter().enumerate() {
            r -= i64::from(x);
            if r < 0 {
                return i;
            }
        }
        w.len() - 1
    }

    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Lognormal with the given median and sigma; the normal deviate is Irwin-Hall(12) - 6.
    pub fn lognormal(&mut self, median: f64, sigma: f64) -> f64 {
        let mut z = -6.0;
        for _ in 0..12 {
            z += self.unit();
        }
        median * det_exp(sigma * z)
    }
}

/// `exp(x)` from IEEE basic operations only (deterministic on every host).
pub fn det_exp(x: f64) -> f64 {
    const LN2: f64 = std::f64::consts::LN_2;
    let x = x.clamp(-700.0, 700.0);
    let k = (x / LN2).round();
    let r = x - k * LN2;
    let (mut term, mut sum) = (1.0f64, 1.0f64);
    for i in 1..=22 {
        term = term * r / f64::from(i);
        sum += term;
    }
    let e = (1023 + k as i64) as u64;
    sum * f64::from_bits(e << 52)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_is_close() {
        for &x in &[-5.0, -1.0, -0.3, 0.0, 0.5, 1.0, 2.0, 4.5] {
            let (a, b) = (det_exp(x), f64::exp(x));
            assert!((a - b).abs() <= b * 1e-12, "{x}: {a} vs {b}");
        }
    }

    #[test]
    fn lognormal_median() {
        let mut r = Rng::derive(7, 0, 0);
        let mut v: Vec<f64> = (0..20001).map(|_| r.lognormal(12.0, 0.8)).collect();
        v.sort_by(f64::total_cmp);
        let med = v[10000];
        assert!((11.0..13.0).contains(&med), "median {med}");
    }

    #[test]
    fn below_in_range() {
        let mut r = Rng::derive(1, 0, 0);
        for n in 1..50 {
            for _ in 0..50 {
                assert!(r.below(n) < n);
            }
        }
        assert_eq!(r.below(0), 0);
    }
}
