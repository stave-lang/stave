//! `gen-tables` writes the generated tables (SIMD lowering, optimizer); `gen-tables --check` exits 1
//! if a checked-in file differs from what it would write (or a table fails its checks).

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let check = match args.as_slice() {
        [] => false,
        [a] if a == "--check" => true,
        _ => {
            eprintln!("usage: gen-tables [--check]");
            return ExitCode::from(2);
        }
    };
    let Some(root) = gen_tables::find_root() else {
        eprintln!(
            "gen-tables: repository root not found (no {} above {})",
            gen_tables::ARCHES[0].table,
            env!("CARGO_MANIFEST_DIR")
        );
        return ExitCode::from(2);
    };
    let files = match gen_tables::generate(&root) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("gen-tables: {e}");
            return ExitCode::from(1);
        }
    };
    let mut stale = false;
    for (rel, content) in files {
        let path = root.join(rel);
        if std::fs::read(&path).ok().as_deref() == Some(content.as_bytes()) {
            continue;
        }
        if check {
            eprintln!("gen-tables: {rel} is stale; run `tools/ci-step/run.sh gen`");
            stale = true;
            continue;
        }
        if let Some(dir) = path.parent()
            && let Err(e) = std::fs::create_dir_all(dir)
        {
            eprintln!("gen-tables: {}: {e}", dir.display());
            return ExitCode::from(1);
        }
        if let Err(e) = std::fs::write(&path, content) {
            eprintln!("gen-tables: {rel}: {e}");
            return ExitCode::from(1);
        }
        println!("gen-tables: wrote {rel}");
    }
    if stale { ExitCode::from(1) } else { ExitCode::SUCCESS }
}
