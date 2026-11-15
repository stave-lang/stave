//! The construct mix (`tools/gen-corpus/mix.toml`).

use serde::Deserialize;

/// The default mix, compiled in.
pub const DEFAULT_MIX: &str = include_str!("../mix.toml");

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Mix {
    pub tiers: Tiers,
    pub modules: Modules,
    pub items: Items,
    pub fn_body: FnBody,
    pub generics: Generics,
    pub adt: Adt,
    pub other: Other,
    pub fns: Fns,
    pub generic_fns: GenericFns,
    pub stmt: Stmt,
    pub expr: Expr,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tiers {
    pub m1: bool,
    pub m2: bool,
    pub m3: bool,
    pub m6: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Modules {
    pub per_10k_lines: u32,
    pub max_depth: u32,
    pub max_imports: u32,
    pub io_permille: u32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Items {
    pub r#fn: u32,
    pub adt: u32,
    pub r#impl: u32,
    pub other: u32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FnBody {
    pub median: f64,
    pub sigma: f64,
    pub cap: u32,
    pub doc_permille: u32,
    pub private_permille: u32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Generics {
    pub share_permille: u32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Adt {
    pub r#struct: u32,
    pub r#enum: u32,
    pub max_fields: u32,
    pub max_variants: u32,
    pub nested_field_permille: u32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Other {
    pub r#const: u32,
    pub alias: u32,
    pub r#trait: u32,
    pub import: u32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fns {
    pub scalar: u32,
    pub struct_ref: u32,
    pub struct_mut: u32,
    pub ctor: u32,
    pub enum_make: u32,
    pub enum_eval: u32,
    pub struct_val: u32,
    pub ref_field: u32,
    pub ref_pick: u32,
    pub mut_field: u32,
    pub option: u32,
    pub result: u32,
    pub tuple: u32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenericFns {
    pub pick: u32,
    pub ref_pick: u32,
    pub bound: u32,
    pub const_n: u32,
    pub ctor: u32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stmt {
    pub compound_permille: u32,
    pub max_nest: u32,
    pub r#let: u32,
    pub assign: u32,
    pub print: u32,
    pub let_struct: u32,
    pub field_write: u32,
    pub mut_call: u32,
    pub call: u32,
    pub consume: u32,
    pub tuple: u32,
    pub early_return: u32,
    pub let_enum: u32,
    pub r#try: u32,
    pub array: u32,
    pub comment: u32,
    pub r#if: u32,
    pub r#while: u32,
    pub r#for: u32,
    pub r#loop: u32,
    pub match_int: u32,
    pub match_enum: u32,
    pub block: u32,
    pub ref_block: u32,
    /// `[m2]` closures with `&` and `&mut` captures.
    pub closure: u32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Expr {
    pub max_depth: u32,
    pub leaf: u32,
    pub binop: u32,
    pub cast: u32,
    pub unary: u32,
    pub call: u32,
    pub r#if: u32,
    pub r#match: u32,
    pub trapping_permille: u32,
}

impl Mix {
    /// Parses and validates a mix file.
    pub fn parse(text: &str) -> Result<Mix, String> {
        let mut mix: Mix = toml::from_str(text).map_err(|e| format!("mix: {e}"))?;
        mix.validate()?;
        if !mix.tiers.m2 {
            let f = &mut mix.fns;
            f.struct_ref = 0;
            f.struct_mut = 0;
            f.ref_field = 0;
            f.ref_pick = 0;
            f.mut_field = 0;
            mix.generic_fns.ref_pick = 0;
            mix.stmt.ref_block = 0;
            mix.stmt.closure = 0;
        }
        Ok(mix)
    }

    /// The compiled-in default mix.
    pub fn default_mix() -> Result<Mix, String> {
        Mix::parse(DEFAULT_MIX)
    }

    fn validate(&self) -> Result<(), String> {
        if !self.tiers.m1 {
            return Err("mix: tier m1 is the base tier and cannot be disabled".into());
        }
        if self.tiers.m3 {
            return Err("mix: tier m3 (SIMD kernels) is not implemented yet".into());
        }
        if self.tiers.m6 {
            return Err("mix: tier m6 (full traits) is not implemented yet".into());
        }
        let m = &self.modules;
        if m.per_10k_lines == 0 || !(1..=3).contains(&m.max_depth) {
            return Err("mix: modules.per_10k_lines must be > 0 and modules.max_depth in 1..=3".into());
        }
        let i = &self.items;
        if i.r#fn + i.adt + i.r#impl + i.other == 0 {
            return Err("mix: items weights are all zero".into());
        }
        let b = &self.fn_body;
        if !(1.0..=400.0).contains(&b.median) || !(0.0..=3.0).contains(&b.sigma) || b.cap == 0 {
            return Err("mix: fn_body needs 1 <= median <= 400, 0 <= sigma <= 3, cap > 0".into());
        }
        if self.adt.max_fields == 0 || self.adt.max_variants < 2 {
            return Err("mix: adt.max_fields must be >= 1 and adt.max_variants >= 2".into());
        }
        if self.adt.r#struct + self.adt.r#enum == 0 {
            return Err("mix: adt weights are all zero".into());
        }
        if self.stmt.r#let == 0 {
            return Err("mix: stmt.let must be > 0 (it is the fallback statement)".into());
        }
        if self.expr.max_depth == 0 || self.expr.max_depth > 6 {
            return Err("mix: expr.max_depth must be in 1..=6".into());
        }
        Ok(())
    }
}
