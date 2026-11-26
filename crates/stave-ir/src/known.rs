//! Compiler-known items: the items of the built-in packages that stages treat specially
//! (contracts/intrinsics.md "Paths fixed by the compiler", the lang types and traits). Their one
//! table is `crates/stave-ir/tables/known.toml`, from which `tools/gen-tables` writes the
//! `known_items!` invocation this file includes (and stage1's table). The front end resolves each
//! entry once, by package identity (an item of a module of a built-in package), and stamps the
//! functions it lowers with their entry ([`crate::Func::known`]); later stages ask for an entry,
//! never for a path. A user item spelled like one is an ordinary item everywhere.

use crate::intrinsics::ITy;

macro_rules! known_items {
    ($($v:ident = $module:literal $name:literal $(. $method:literal)?;)*) => {
        /// A compiler-known item.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub enum Known {
            $($v,)*
        }

        impl Known {
            /// Every entry, in declaration order.
            pub const ALL: &'static [Known] = &[$(Known::$v,)*];

            /// The module of the item in its built-in package, segments joined by `.`.
            pub fn module(self) -> &'static str {
                match self {
                    $(Known::$v => $module,)*
                }
            }

            /// The item's name in that module.
            pub fn name(self) -> &'static str {
                match self {
                    $(Known::$v => $name,)*
                }
            }

            /// An inherent method of that item, when the entry is the method.
            pub fn method(self) -> Option<&'static str> {
                match self {
                    $(Known::$v => known_items!(@method $($method)?),)*
                }
            }
        }
    };
    (@method) => {
        None
    };
    (@method $m:literal) => {
        Some($m)
    };
}

include!("generated/known_items.rs");

/// The operator traits of `std.core`, in the order of the front end's operator table.
pub const OP_TRAITS: [Known; 19] = [
    Known::OpAdd,
    Known::OpWrapAdd,
    Known::OpSatAdd,
    Known::OpSub,
    Known::OpWrapSub,
    Known::OpSatSub,
    Known::OpMul,
    Known::OpWrapMul,
    Known::OpSatMul,
    Known::OpDiv,
    Known::OpRem,
    Known::OpShl,
    Known::OpWrapShl,
    Known::OpShr,
    Known::OpBitAnd,
    Known::OpBitOr,
    Known::OpBitXor,
    Known::OpNeg,
    Known::OpNot,
];

impl Known {
    /// The item's path (`runtime.panic.trap`, `std.string.String.new`): text for messages and
    /// dumps, never a key.
    pub fn path(self) -> String {
        match self.method() {
            Some(m) => format!("{}.{}.{m}", self.module(), self.name()),
            None => format!("{}.{}", self.module(), self.name()),
        }
    }
}

/// Whether the known item is a panic report entry, which runs on the alternate signal stack
/// ([`crate::Func::alt_stack`]): every trap and every std panic enters the report through one of
/// these. Any other item runs on the stack it is called on.
pub fn reports_on_alt_stack(k: Known) -> bool {
    REPORTS.contains(&k)
}

/// The signature (params, result) of a known item that code generation or the interpreter calls
/// or enters with an argument list of its own (the start stub, the trap and signal reports, the
/// thread stubs, `cpu_init`, `copy_bytes`, the checked allocator); `None` for the others, which
/// are only ever called by IR calls. A result of `()` is no result; `-> never` is none either.
pub fn abi_signature(k: Known) -> Option<(&'static [ITy], ITy)> {
    abi_table(k)
}

/// Whether the release optimizer inlines the known item like an ordinary fn (contracts/
/// opt-passes.md rule F8); it never inlines any other known item. None of them has an ABI
/// signature or is a panic report entry (the table generator refuses that).
pub fn inlined(k: Known) -> bool {
    INLINED.contains(&k)
}

/// The param of a known item that lowering passes a site row in, as an int constant
/// (contracts/intrinsics.md "Paths fixed by the compiler"): the user-site store, the error-trace
/// hooks, and the failed `ensures` report. `None` for the others.
pub fn site_param(k: Known) -> Option<usize> {
    match k {
        Known::SetUserSite | Known::ErrtracePush | Known::ErrtraceRestart | Known::EnsuresFailed => Some(0),
        _ => None,
    }
}

/// Whether `f` is the fn a failed `requires` check calls.
pub fn is_requires_failed(f: &crate::Func) -> bool {
    f.known == Some(Known::RequiresFailed)
}

/// Whether `f` is the fn a failed `requires` or `ensures` check calls.
pub fn is_contract_failed(f: &crate::Func) -> bool {
    matches!(f.known, Some(Known::RequiresFailed | Known::EnsuresFailed))
}

impl std::fmt::Display for Known {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.path())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn site_param_of_every_item() {
        let with: Vec<(Known, usize)> =
            Known::ALL.iter().filter_map(|&k| site_param(k).map(|p| (k, p))).collect();
        assert_eq!(
            with,
            [
                (Known::EnsuresFailed, 0),
                (Known::SetUserSite, 0),
                (Known::ErrtracePush, 0),
                (Known::ErrtraceRestart, 0),
            ]
        );
    }

    #[test]
    fn inlined_items_have_no_abi_and_no_report() {
        for &k in Known::ALL.iter().filter(|&&k| inlined(k)) {
            assert!(abi_signature(k).is_none(), "{k:?}");
            assert!(!reports_on_alt_stack(k), "{k:?}");
        }
        assert!(inlined(Known::SetUserSite));
        assert!(!inlined(Known::Trap));
    }
}
