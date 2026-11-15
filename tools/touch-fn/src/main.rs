//! `touch-fn DIR [--seed S]`: edits one fn body, prints `file:line`.

use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "usage: touch-fn DIR [--seed S]";

fn main() -> ExitCode {
    let mut dir: Option<PathBuf> = None;
    let mut seed = 0u64;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--seed" => match it.next().map(|v| v.parse::<u64>()) {
                Some(Ok(v)) => seed = v,
                _ => {
                    eprintln!("touch-fn: --seed needs an unsigned integer\n{USAGE}");
                    return ExitCode::from(2);
                }
            },
            "-h" | "--help" => {
                eprintln!("{USAGE}");
                return ExitCode::from(2);
            }
            _ if dir.is_none() && !a.starts_with('-') => dir = Some(PathBuf::from(a)),
            _ => {
                eprintln!("touch-fn: unexpected argument {a:?}\n{USAGE}");
                return ExitCode::from(2);
            }
        }
    }
    let Some(dir) = dir else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    match touch_fn::touch(&dir, seed) {
        Ok(t) => {
            println!("{}:{}", t.file.display(), t.line);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("touch-fn: {e}");
            ExitCode::FAILURE
        }
    }
}
