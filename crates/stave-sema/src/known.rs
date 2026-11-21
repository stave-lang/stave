//! The compiler-known items of a program ([`stave_ir::Known`]), resolved once by identity: an
//! entry names an item of a module of a built-in package, so a user module, item, or dependency
//! spelled the same way is never one of them. Everything after resolution asks this table by
//! entry ([`KnownItems::decl`]) or by decl ([`KnownItems::of`]).

use rustc_hash::FxHashMap;
use stave_ir::Known;

use crate::ids::{DeclId, ModuleId};
use crate::modules::Module;
use crate::package::{Package, PackageKind};
use crate::resolve::TypeHead;
use crate::{Program, Strings};

/// The decl of each [`Known`] entry the program has.
#[derive(Clone, Debug, Default)]
pub struct KnownItems {
    decls: Vec<DeclId>,
    of: FxHashMap<DeclId, Known>,
}

/// The module of a built-in package whose path segments are those of `path` (`std.core`).
pub(crate) fn builtin_module(
    modules: &[Module],
    packages: &[Package],
    strings: &Strings,
    path: &str,
) -> Option<ModuleId> {
    modules
        .iter()
        .find(|m| {
            m.path.len() == path.split('.').count()
                && m.path.iter().zip(path.split('.')).all(|(s, t)| strings.get(*s) == t)
                && packages.get(m.package.idx()).is_some_and(|p| p.kind == PackageKind::Builtin)
        })
        .map(|m| m.id)
}

impl KnownItems {
    /// Resolves every entry of [`Known::ALL`] in `prog` (an entry whose module, item, or method
    /// is missing has no decl).
    pub(crate) fn resolve(prog: &Program) -> KnownItems {
        let mut modules: FxHashMap<&'static str, Option<ModuleId>> = FxHashMap::default();
        let mut decls = vec![DeclId::NONE; Known::ALL.len()];
        let mut of = FxHashMap::default();
        for &k in Known::ALL {
            let m = *modules
                .entry(k.module())
                .or_insert_with(|| builtin_module(&prog.modules, &prog.packages, &prog.strings, k.module()));
            let item = m.and_then(|m| prog.module_item(m, prog.strings.lookup(k.name())?));
            let d = match (item, k.method()) {
                (Some(d), None) => Some(d),
                (Some(adt), Some(name)) => prog
                    .strings
                    .lookup(name)
                    .and_then(|s| prog.methods.inherent.get(&(TypeHead::Adt(adt), s)))
                    .and_then(|v| v.first().copied()),
                (None, _) => None,
            };
            if let Some(d) = d.filter(|d| d.is_some()) {
                decls[k as usize] = d;
                of.insert(d, k);
            }
        }
        KnownItems { decls, of }
    }

    /// The decl of entry `k`, if the program has it.
    pub fn decl(&self, k: Known) -> Option<DeclId> {
        self.decls.get(k as usize).copied().filter(|d| d.is_some())
    }

    /// The decl of entry `k`, `DeclId::NONE` if the program does not have it.
    pub fn decl_or_none(&self, k: Known) -> DeclId {
        self.decl(k).unwrap_or(DeclId::NONE)
    }

    /// The entry decl `d` is, if it is one.
    pub fn of(&self, d: DeclId) -> Option<Known> {
        self.of.get(&d).copied()
    }

    /// Whether decl `d` is entry `k`.
    pub fn is(&self, d: DeclId, k: Known) -> bool {
        d.is_some() && self.decl(k) == Some(d)
    }
}
