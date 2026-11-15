//! diaggen: checks `contracts/diagnostics.toml` against the human code tables and generates the
//! template tables of stage0 (Rust) and stage1 (Stave).

pub mod emit_rust;
pub mod emit_stave;
pub mod md;
pub mod registry;
pub mod template;

use std::path::{Path, PathBuf};

pub use registry::{Errors, Registry, check, parse_registry};

/// The registry, relative to the repository root.
pub const REGISTRY: &str = "contracts/diagnostics.toml";
/// The human tables whose code rows must match the registry: the main table, and the GPU kernel
/// (E11xx) and GPU program (E12xx) tables it delegates to.
pub const TABLES: [&str; 3] =
    ["contracts/diagnostic-codes.md", "contracts/gpu-diagnostics.md", "contracts/gpu-program-diagnostics.md"];
/// Generated files, relative to the repository root.
pub const RUST_OUT: &str = "crates/stave-diag/src/generated/templates.rs";
pub const STAVE_OUT: &str = "stage1/src/generated/diag_templates.stv";

/// Finds the repository root: the first ancestor of this crate's manifest directory that holds
/// the registry (as ll1gen does with the grammar), whatever the current directory.
pub fn find_root() -> Option<PathBuf> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .find(|p| p.join(REGISTRY).is_file())
        .map(Path::to_path_buf)
}

/// Reads and checks the registry under `root`.
pub fn load(root: &Path) -> Result<Registry, String> {
    let read = |rel: &str| std::fs::read_to_string(root.join(rel)).map_err(|e| format!("{rel}: {e}"));
    let toml_text = read(REGISTRY)?;
    let mut tables = Vec::new();
    for t in TABLES {
        tables.push((t, read(t)?));
    }
    let refs: Vec<(&str, &str)> = tables.iter().map(|(n, t)| (*n, t.as_str())).collect();
    check(&toml_text, &refs).map_err(|e| e.to_string())
}

/// The generated files, as (path relative to root, content).
pub fn generate(reg: &Registry) -> [(&'static str, String); 2] {
    [(RUST_OUT, emit_rust::emit(reg)), (STAVE_OUT, emit_stave::emit(reg))]
}
