//! Sharded, thread-safe intern pool: key → dense `u32` id, equal keys → equal id.
//!
//! Lookups of present keys lock only one shard (64 shards, chosen by the key's Fx hash). Inserting
//! a new key also takes the id table lock to append, so ids stay dense (`0..len`). Raw ids follow
//! insertion order, which depends on thread timing in a parallel phase; they are not canonical.
//! Freeze the pool with [`Interner::freeze`] and renumber it with [`crate::canon`] before any id
//! reaches a dump, the IR, a diagnostic, or codegen ordering.
//!
//! A poisoned lock is recovered (the pool is append-only, so a panicking holder cannot leave a
//! shard half-updated in a way that breaks the key → id invariant).

use std::borrow::Borrow;
use std::fmt;
use std::hash::{BuildHasher, Hash};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use rustc_hash::{FxBuildHasher, FxHashMap};

use crate::canon::Remap;

const SHARD_BITS: u32 = 6;
const SHARDS: usize = 1 << SHARD_BITS;

/// The pool already holds `u32::MAX` keys (ids are `u32`; `u32::MAX` is kept free for sentinels).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolFull;

impl fmt::Display for PoolFull {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("intern pool full (u32 ids exhausted)")
    }
}

impl std::error::Error for PoolFull {}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Thread-safe interner for keys of type `K` (strings, type keys, const values).
pub struct Interner<K> {
    shards: Box<[Mutex<FxHashMap<K, u32>>]>,
    keys: Mutex<Vec<K>>,
}

/// String pool. `Arc<str>` keys make [`Interner::get`] cheap.
pub type StringPool = Interner<Arc<str>>;

impl<K: Hash + Eq + Clone> Default for Interner<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Hash + Eq + Clone> Interner<K> {
    pub fn new() -> Self {
        let shards = (0..SHARDS).map(|_| Mutex::new(FxHashMap::default())).collect();
        Interner { shards, keys: Mutex::new(Vec::new()) }
    }

    fn shard<Q: ?Sized + Hash>(&self, q: &Q) -> &Mutex<FxHashMap<K, u32>> {
        let h = FxBuildHasher.hash_one(q);
        let i = (h >> (64 - SHARD_BITS)) as usize;
        // `i < SHARDS` by construction; fall back to shard 0 rather than index-panic.
        self.shards.get(i).unwrap_or(&self.shards[0])
    }

    /// Id of `key`, inserting it if absent.
    pub fn intern(&self, key: K) -> Result<u32, PoolFull> {
        let mut shard = lock(self.shard(&key));
        if let Some(&id) = shard.get(&key) {
            return Ok(id);
        }
        let id = self.push(key.clone())?;
        shard.insert(key, id);
        Ok(id)
    }

    /// Id of the key equal to `q`, building the owned key with `make` only on a miss.
    pub fn intern_with<Q>(&self, q: &Q, make: impl FnOnce(&Q) -> K) -> Result<u32, PoolFull>
    where
        K: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
    {
        let mut shard = lock(self.shard(q));
        if let Some(&id) = shard.get(q) {
            return Ok(id);
        }
        let key = make(q);
        let id = self.push(key.clone())?;
        shard.insert(key, id);
        Ok(id)
    }

    fn push(&self, key: K) -> Result<u32, PoolFull> {
        let mut keys = lock(&self.keys);
        let id = u32::try_from(keys.len()).map_err(|_| PoolFull)?;
        if id == u32::MAX {
            return Err(PoolFull);
        }
        keys.push(key);
        Ok(id)
    }

    /// Id of the key equal to `q`, without inserting.
    pub fn lookup<Q>(&self, q: &Q) -> Option<u32>
    where
        K: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
    {
        lock(self.shard(q)).get(q).copied()
    }

    /// The key with id `id` (a clone).
    pub fn get(&self, id: u32) -> Option<K> {
        lock(&self.keys).get(id as usize).cloned()
    }

    pub fn len(&self) -> usize {
        lock(&self.keys).len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// End the insertion phase: a lock-free, read-only table with the same ids.
    pub fn freeze(self) -> FrozenPool<K> {
        let keys = self.keys.into_inner().unwrap_or_else(PoisonError::into_inner);
        FrozenPool::from_keys_unchecked(keys)
    }
}

impl Interner<Arc<str>> {
    /// Id of `s`, allocating only when `s` is new.
    pub fn intern_str(&self, s: &str) -> Result<u32, PoolFull> {
        self.intern_with(s, |q: &str| Arc::from(q))
    }
}

/// A read-only pool: ids `0..len`, key by id and id by key.
#[derive(Clone, Debug)]
pub struct FrozenPool<K> {
    keys: Vec<K>,
    ids: FxHashMap<K, u32>,
}

impl<K: Hash + Eq + Clone> FrozenPool<K> {
    /// Build from keys in id order. `keys` must be distinct (true for an interner's table).
    fn from_keys_unchecked(keys: Vec<K>) -> Self {
        let mut ids = FxHashMap::default();
        ids.reserve(keys.len());
        for (i, k) in keys.iter().enumerate() {
            ids.insert(k.clone(), i as u32);
        }
        FrozenPool { keys, ids }
    }

    pub fn get(&self, id: u32) -> Option<&K> {
        self.keys.get(id as usize)
    }

    pub fn lookup<Q>(&self, q: &Q) -> Option<u32>
    where
        K: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
    {
        self.ids.get(q).copied()
    }

    /// Keys in id order.
    pub fn keys(&self) -> &[K] {
        &self.keys
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Renumber: key `k` with old id `i` gets id `remap.new_of(i)`. `None` if `remap` does not
    /// cover exactly this pool's ids.
    pub fn renumber(self, remap: &Remap) -> Option<FrozenPool<K>> {
        let keys = remap.permute(self.keys).ok()?;
        Some(FrozenPool::from_keys_unchecked(keys))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_keys_equal_ids_dense() {
        let p = StringPool::new();
        let a = p.intern_str("alpha").unwrap();
        let b = p.intern_str("beta").unwrap();
        assert_eq!(p.intern_str("alpha").unwrap(), a);
        assert_eq!(p.intern(Arc::from("beta")).unwrap(), b);
        assert_ne!(a, b);
        assert_eq!((a, b), (0, 1));
        assert_eq!(p.len(), 2);
        assert_eq!(p.lookup("beta"), Some(b));
        assert_eq!(p.lookup("gamma"), None);
        assert_eq!(p.get(a).as_deref(), Some("alpha"));
        assert_eq!(p.get(7), None);
    }

    #[test]
    fn generic_keys() {
        #[derive(Clone, PartialEq, Eq, Hash, Debug)]
        enum Ty {
            Int(u8),
            Ptr(u32),
        }
        let p = Interner::<Ty>::new();
        let i = p.intern(Ty::Int(32)).unwrap();
        let q = p.intern(Ty::Ptr(i)).unwrap();
        assert_eq!(p.intern(Ty::Ptr(i)).unwrap(), q);
        assert_eq!(p.intern(Ty::Int(32)).unwrap(), i);
        let f = p.freeze();
        assert_eq!(f.get(q), Some(&Ty::Ptr(i)));
        assert_eq!(f.lookup(&Ty::Int(32)), Some(i));
    }

    #[test]
    fn concurrent_inserts_agree() {
        let p = StringPool::new();
        let words: Vec<String> = (0..2000).map(|i| format!("w{}", i % 500)).collect();
        let ids: Vec<Vec<u32>> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..8)
                .map(|t| {
                    let p = &p;
                    let words = &words;
                    s.spawn(move || {
                        let mut out = Vec::new();
                        for k in 0..words.len() {
                            let w = &words[(k * 7 + t * 131) % words.len()];
                            out.push((w.clone(), p.intern_str(w).unwrap()));
                        }
                        out.sort();
                        out.into_iter().map(|(_, id)| id).collect()
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        assert_eq!(p.len(), 500);
        for v in &ids[1..] {
            assert_eq!(v, &ids[0]);
        }
        let f = p.freeze();
        let mut seen: Vec<u32> = (0..500).map(|i| f.lookup(format!("w{i}").as_str()).unwrap()).collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..500).collect::<Vec<u32>>());
    }

    #[test]
    fn renumber_frozen() {
        let p = StringPool::new();
        for s in ["c", "a", "b"] {
            p.intern_str(s).unwrap();
        }
        let f = p.freeze();
        let remap = crate::canon::rank_by_key(f.keys(), |k| k.clone()).unwrap();
        let g = f.renumber(&remap).unwrap();
        let keys: Vec<&str> = g.keys().iter().map(|k| &**k).collect();
        assert_eq!(keys, ["a", "b", "c"]);
        assert_eq!(g.lookup("c"), Some(2));
    }
}
