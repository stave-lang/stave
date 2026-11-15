//! `diaggen` writes the generated template tables; `diaggen --check` exits 1 if a checked-in
//! file differs from what it would write (or the registry fails its checks).

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let check = match args.as_slice() {
        [] => false,
        [a] if a == "--check" => true,
        _ => {
            eprintln!("usage: diaggen [--check]");
            return ExitCode::from(2);
        }
    };
    let Some(root) = diaggen::find_root() else {
        eprintln!(
            "diaggen: repository root not found (no {} above {})",
            diaggen::REGISTRY,
            env!("CARGO_MANIFEST_DIR")
        );
        return ExitCode::from(2);
    };
    let reg = match diaggen::load(&root) {
        Ok(r) => r,
        Err(e) => {
            eprint!("diaggen: {}", if e.ends_with('\n') { e } else { e + "\n" });
            return ExitCode::from(1);
        }
    };
    let mut stale = false;
    for (rel, content) in diaggen::generate(&reg) {
        let path = root.join(rel);
        let current = std::fs::read(&path).ok();
        if current.as_deref() == Some(content.as_bytes()) {
            continue;
        }
        if check {
            eprintln!("diaggen: {rel} is stale; run `tools/ci-step/run.sh gen`");
            stale = true;
            continue;
        }
        if let Some(dir) = path.parent()
            && let Err(e) = std::fs::create_dir_all(dir)
        {
            eprintln!("diaggen: {}: {e}", dir.display());
            return ExitCode::from(1);
        }
        if let Err(e) = std::fs::write(&path, content) {
            eprintln!("diaggen: {rel}: {e}");
            return ExitCode::from(1);
        }
        println!("diaggen: wrote {rel}");
    }
    if stale { ExitCode::from(1) } else { ExitCode::SUCCESS }
}
