//! Fuzz the whole syntax layer: lex, item cut, parse, print. No panic on any bytes; ASTs are well
//! formed; diagnostics render; printed text re-parses to the same AST.
#![no_main]

use stave_diag::{FileId, SourceFiles, emit_human, emit_json};
use stave_syntax::{parse_file, print_file};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let pf = parse_file(data, FileId(0));
    for i in 0..pf.items.len() {
        let ast = pf.item_ast(i);
        assert_eq!(ast.is_empty(), !pf.items[i].ok);
        for (n, node) in ast.nodes.iter().enumerate() {
            assert!(node.b as usize <= n, "subtree of node {n} starts inside the item");
            assert!((node.a as usize) < pf.tokens.len());
        }
        if let Some(root) = ast.root() {
            assert_eq!(ast.get(root).map(|r| r.b as usize), Some(ast.len() - 1), "root spans the item");
        }
    }
    let mut files = SourceFiles::new();
    if files.add("fuzz", "fuzz.stv", data).is_ok() {
        emit_json(&pf.diagnostics, &files).expect("spans locate");
        emit_human(&pf.diagnostics, &files).expect("spans locate");
    }
    let printed = print_file(&pf, data);
    if pf.stop.is_none() {
        assert_eq!(printed.fallbacks, 0, "printer fell back");
        let again = parse_file(printed.text.as_bytes(), FileId(0));
        assert!(again.ast == pf.ast, "printed text parses to another AST");
    }
});
