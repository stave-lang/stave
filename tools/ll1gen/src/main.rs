//! `ll1gen [--check] [--root DIR] [--grammar FILE]`
//!
//! Reads `contracts/grammar.ebnf`, checks it is LL(1) up to the listed overrides, and writes the
//! generated tables (paths relative to the repository root). `--check` writes nothing and exits 1
//! if any generated file is stale. Exit 1 on any grammar error, unlisted conflict or stale
//! override; exit 2 on a usage error.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

const USAGE: &str = "usage: ll1gen [--check] [--root DIR] [--grammar FILE]";

/// The grammar, relative to the repository root.
const GRAMMAR: &str = "contracts/grammar.ebnf";

fn main() -> ExitCode {
    let mut check = false;
    // The repository root: the first ancestor of this crate's manifest directory that holds the
    // grammar (as diaggen does with its registry), whatever the current directory.
    let mut root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .find(|p| p.join(GRAMMAR).is_file())
        .map_or_else(|| PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../..")), Path::to_path_buf);
    let mut grammar: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--check" => check = true,
            "--root" | "--grammar" => {
                let Some(v) = args.next() else {
                    eprintln!("ll1gen: {a} needs a value\n{USAGE}");
                    return ExitCode::from(2);
                };
                if a == "--root" {
                    root = PathBuf::from(v);
                } else {
                    grammar = Some(PathBuf::from(v));
                }
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            _ => {
                eprintln!("ll1gen: unknown argument `{a}`\n{USAGE}");
                return ExitCode::from(2);
            }
        }
    }
    let (gpath, label) = match grammar {
        Some(g) => {
            let label = g.display().to_string();
            (g, label)
        }
        None => (root.join(GRAMMAR), GRAMMAR.to_string()),
    };
    match run(&root, &gpath, &label, check) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msgs) => {
            for m in msgs {
                eprintln!("ll1gen: {m}");
            }
            ExitCode::from(1)
        }
    }
}

fn run(root: &Path, gpath: &Path, label: &str, check: bool) -> Result<(), Vec<String>> {
    let src = std::fs::read_to_string(gpath).map_err(|e| vec![format!("{}: {e}", gpath.display())])?;
    let (t, outputs) = ll1gen::generate(&src, label, &ll1gen::Config::contract()).map_err(|errs| {
        errs.iter()
            .map(|e| match e.line {
                Some(l) => format!("{label}:{l}: {}", e.msg),
                None => format!("{label}: {}", e.msg),
            })
            .collect::<Vec<_>>()
    })?;
    let mut stale = Vec::new();
    for o in &outputs {
        let path = root.join(&o.path);
        let current = std::fs::read(&path).ok();
        if current.as_deref() == Some(o.content.as_bytes()) {
            continue;
        }
        if check {
            stale.push(format!("stale generated file {} (run `tools/ci-step/run.sh gen`)", o.path));
            continue;
        }
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| vec![format!("{}: {e}", dir.display())])?;
        }
        std::fs::write(&path, &o.content).map_err(|e| vec![format!("{}: {e}", path.display())])?;
        println!("ll1gen: wrote {}", o.path);
    }
    if !stale.is_empty() {
        return Err(stale);
    }
    println!(
        "ll1gen: {} nonterminals ({} rules), {} terminals, {} productions, {} overrides{}",
        t.nts.len(),
        t.nkind,
        t.nterm(),
        t.prods.len(),
        t.overrides.len(),
        if check { ", up to date" } else { "" }
    );
    Ok(())
}
