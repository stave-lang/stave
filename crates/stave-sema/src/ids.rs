//! Typed `u32` ids. Every id the public API exposes is canonical:
//! the rank of a stable sort key, never an insertion or thread order.

use stave_intern::RawId;

macro_rules! id {
    ($(#[$m:meta])* $name:ident) => {
        $(#[$m])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(pub u32);

        impl $name {
            /// "No id" (`u32::MAX`).
            pub const NONE: $name = $name(u32::MAX);

            #[inline]
            pub fn idx(self) -> usize {
                self.0 as usize
            }

            #[inline]
            pub fn is_none(self) -> bool {
                self.0 == u32::MAX
            }

            #[inline]
            pub fn is_some(self) -> bool {
                self.0 != u32::MAX
            }

            /// `None` for [`Self::NONE`].
            #[inline]
            pub fn get(self) -> Option<$name> {
                if self.is_none() { None } else { Some(self) }
            }
        }

        impl RawId for $name {
            fn raw(self) -> u32 {
                self.0
            }
            fn from_raw(raw: u32) -> Self {
                $name(raw)
            }
        }
    };
}

id!(
    /// Interned string: rank of (xxHash32 seed 0, bytes) (stage-gates `strings` `S`).
    Sym
);
id!(
    /// Package: rank of (name, directory).
    PackageId
);
id!(
    /// Module (one per source file): rank of (module path text, package).
    ModuleId
);
id!(
    /// Declaration: rank of the decl key (see [`crate::decls`]).
    DeclId
);
id!(
    /// Item: rank of (file, start offset) (stage-gates `I`).
    ItemId
);

/// A local binding inside one item: its binding node (`Param`, `SelfParam`, `Pattern`,
/// `FieldPat`, `TailBind`). Every binding node binds exactly one name (each `tail masked (v, m)`
/// name has its own `TailBind`), so dumps print the node alone (`local I N2`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BindingRef {
    pub node: u32,
}

/// A generic parameter of an item: its `GenericParam` node, its index in its `Generics` list,
/// and the list's level (0 = the item's own list or the impl/trait list, 1 = a method's list).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GenericRef {
    pub node: u32,
    pub index: u16,
    pub level: u8,
    pub is_const: bool,
}
