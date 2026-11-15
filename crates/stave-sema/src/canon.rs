//! Canonical decl ids: rank of the decl key (see [`crate::decls`] docs), then every
//! decl-id-holding table is rewritten.

use rayon::slice::ParallelSliceMut;
use stave_intern::Remap;

use crate::decls::{Decl, DeclKind};
use crate::ids::DeclId;
use crate::modules::Module;
use crate::resolve::{ItemRes, Res};

/// `(module path, steps (name, kind, tiebreak, name offset))`, compared lexicographically.
type Key = (Vec<u32>, Vec<(u32, u8, u32, u32)>);

fn keys(decls: &[Decl], modules: &[Module]) -> Vec<Key> {
    let mut out: Vec<Key> = Vec::with_capacity(decls.len());
    for d in decls {
        let m = modules.get(d.module.idx());
        let path = |k: usize| -> Vec<u32> {
            m.map_or(Vec::new(), |m| m.path[..k.min(m.path.len())].iter().map(|s| s.0).collect())
        };
        let key = if d.kind == DeclKind::Module {
            let n = m.map_or(0, |m| m.path.len());
            let pkg = m.map_or(u32::MAX, |m| m.package.0);
            (path(n.saturating_sub(1)), vec![(d.name.0, d.kind as u8, pkg, 0)])
        } else if let Some(parent) = d.parent.get().filter(|p| p.idx() < out.len()) {
            let (pp, mut steps) = out[parent.idx()].clone();
            steps.push((d.name.0, d.kind as u8, d.index, d.name_span.offset));
            (pp, steps)
        } else {
            let n = m.map_or(0, |m| m.path.len());
            (path(n), vec![(d.name.0, d.kind as u8, d.item.0, d.name_span.offset)])
        };
        out.push(key);
    }
    out
}

/// The canonical remap of `decls` (raw id → canonical id): the rank of each key. Keys are
/// unique (every step has a tiebreak), so the parallel sort's order is total.
pub(crate) fn decl_remap(decls: &[Decl], modules: &[Module]) -> Option<Remap> {
    let k = keys(decls, modules);
    let n = u32::try_from(k.len()).ok()?;
    let mut idx: Vec<u32> = (0..n).collect();
    idx.par_sort_unstable_by(|&a, &b| k[a as usize].cmp(&k[b as usize]).then(a.cmp(&b)));
    let mut old_to_new = vec![0u32; idx.len()];
    for (new, &old) in idx.iter().enumerate() {
        old_to_new[old as usize] = new as u32;
    }
    Remap::from_old_to_new(old_to_new).ok()
}

#[inline]
pub(crate) fn map_decl(r: &Remap, d: DeclId) -> DeclId {
    if d.is_none() { d } else { r.new_of(d.0).map_or(DeclId::NONE, DeclId) }
}

pub(crate) fn map_res(r: &Remap, x: Res) -> Res {
    match x {
        Res::Decl(d) => Res::Decl(map_decl(r, d)),
        other => other,
    }
}

pub(crate) fn remap_item_res(r: &Remap, ir: &mut ItemRes) {
    for u in &mut ir.uses {
        u.res = map_res(r, u.res);
    }
    for d in &mut ir.deferred {
        d.base = map_res(r, d.base);
    }
}
