//! `ll1gen [--check] [--root DIR] [--profile NAME [--grammar FILE]]`
//!
//! For every profile (or only `--profile NAME`; see `ll1gen::PROFILES`): reads the profile's
//! grammar, checks it is LL(1) up to the profile's listed overrides, and writes the generated
//! tables (paths relative to the repository root). `--grammar` replaces the selected profile's
//! grammar file. `--check` writes nothing and exits 1 if any generated file is stale. Exit 1 on
//! any grammar error, unlisted conflict or stale override in any profile (every profile is still
//! processed and reported); exit 2 on a usage error.

use ll1gen::{PROFILES, Profile};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const USAGE: &str = "usage: ll1gen [--check] [--root DIR] [--profile NAME [--grammar FILE]]";

fn main() -> ExitCode {
    let mut check = false;
    // The repository root: the first ancestor of this crate's manifest directory that holds the
    // language grammar (as diaggen does with its registry), whatever the current directory.
    let mut root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .find(|p| p.join(ll1gen::LANGUAGE.grammar).is_file())
        .map_or_else(|| PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../..")), Path::to_path_buf);
    let mut grammar: Option<PathBuf> = None;
    let mut profile: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--check" => check = true,
            "--root" | "--grammar" | "--profile" => {
                let Some(v) = args.next() else {
                    eprintln!("ll1gen: {a} needs a value\n{USAGE}");
                    return ExitCode::from(2);
                };
                match a.as_str() {
                    "--root" => root = PathBuf::from(v),
                    "--grammar" => grammar = Some(PathBuf::from(v)),
                    _ => profile = Some(v),
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
    if grammar.is_some() && profile.is_none() {
        eprintln!("ll1gen: --grammar needs --profile\n{USAGE}");
        return ExitCode::from(2);
    }
    let selected: Vec<&Profile> = match &profile {
        None => PROFILES.to_vec(),
        Some(n) => match PROFILES.iter().find(|p| p.name == n.as_str()) {
            Some(p) => vec![*p],
            None => {
                let names: Vec<&str> = PROFILES.iter().map(|p| p.name).collect();
                eprintln!("ll1gen: unknown profile `{n}` (profiles: {})\n{USAGE}", names.join(", "));
                return ExitCode::from(2);
            }
        },
    };
    let mut failed = false;
    for prof in selected {
        let (gpath, label) = match &grammar {
            Some(g) => (g.clone(), g.display().to_string()),
            None => (root.join(prof.grammar), prof.grammar.to_string()),
        };
        if let Err(msgs) = run(&root, prof, &gpath, &label, check) {
            failed = true;
            for m in msgs {
                eprintln!("ll1gen: {}: {m}", prof.name);
            }
        }
    }
    if failed { ExitCode::from(1) } else { ExitCode::SUCCESS }
}

fn run(root: &Path, prof: &Profile, gpath: &Path, label: &str, check: bool) -> Result<(), Vec<String>> {
    let src = std::fs::read_to_string(gpath).map_err(|e| vec![format!("{}: {e}", gpath.display())])?;
    let (t, outputs) = ll1gen::generate(&src, label, prof).map_err(|errs| {
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
        "ll1gen: {}: {} nonterminals ({} rules), {} terminals, {} productions, {} overrides{}",
        prof.name,
        t.nts.len(),
        t.nkind,
        t.nterm(),
        t.prods.len(),
        t.overrides.len(),
        if check { ", up to date" } else { "" }
    );
    Ok(())
}
