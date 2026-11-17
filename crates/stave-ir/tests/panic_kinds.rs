//! The panic kinds agree everywhere they are listed: `PanicKind::ALL` order is the kind number
//! the runtime reads (`runtime/panic.stv` `KIND_*` and its report names), and the checked-IR
//! contract and LANG.md list every kind.

use stave_ir::PanicKind;
use std::path::Path;

fn repo_file(rel: &str) -> String {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

#[test]
fn runtime_and_docs_list_every_kind() {
    let runtime = repo_file("runtime/panic.stv");
    let ir = repo_file("contracts/checked-ir.md");
    let lang = repo_file("LANG.md");
    let lang_kinds =
        lang.split("Kinds: `").nth(1).and_then(|s| s.split('`').next()).expect("LANG.md: the `Kinds:` list");
    let lang_kinds: Vec<&str> = lang_kinds.split(' ').collect();
    for (i, k) in PanicKind::ALL.iter().enumerate() {
        let op = k.op_name();
        let konst = format!("KIND_{}", op.trim_start_matches("panic.").to_uppercase());
        assert!(
            runtime.contains(&format!("pub const {konst}: u32 = {i};")),
            "runtime/panic.stv: {konst} = {i}"
        );
        assert!(
            runtime.contains(&format!("kind == {konst} {{ l.lit(\"{}\"); }}", k.name())),
            "runtime/panic.stv: report name of {konst}"
        );
        assert!(ir.contains(op), "contracts/checked-ir.md: {op}");
        assert_eq!(lang_kinds.get(i), Some(&k.name()), "LANG.md kinds, position {i}");
    }
    assert!(runtime.contains(&format!("pub const NKIND: u32 = {};", PanicKind::ALL.len())));
    assert_eq!(lang_kinds.len(), PanicKind::ALL.len(), "LANG.md kinds");
}
