//! `lang-budget [FILE] [--limit N]`: prints `FILE: <n> tokens (limit N, <bytes> bytes / 4)`, exits 1
//! above the limit. Defaults: `LANG.md`, 15000.
//!
//! The token count is pinned to `ceil(bytes / 4)` of the UTF-8 file: deterministic, offline, and
//! independent of any vocabulary. Changing the divisor changes every count, so it shows up as a diff here.

use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "usage: lang-budget [FILE] [--limit N]";
const DEFAULT_FILE: &str = "LANG.md";
const DEFAULT_LIMIT: usize = 15_000;
const BYTES_PER_TOKEN: usize = 4;

fn count_tokens(text: &str) -> usize {
    text.len().div_ceil(BYTES_PER_TOKEN)
}

fn main() -> ExitCode {
    let mut file: Option<PathBuf> = None;
    let mut limit = DEFAULT_LIMIT;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--limit" => match it.next().map(|v| v.parse::<usize>()) {
                Some(Ok(v)) => limit = v,
                _ => {
                    eprintln!("lang-budget: --limit needs an unsigned integer\n{USAGE}");
                    return ExitCode::from(2);
                }
            },
            "-h" | "--help" => {
                eprintln!("{USAGE}");
                return ExitCode::from(2);
            }
            _ if file.is_none() && !a.starts_with('-') => file = Some(PathBuf::from(a)),
            _ => {
                eprintln!("lang-budget: unexpected argument {a:?}\n{USAGE}");
                return ExitCode::from(2);
            }
        }
    }
    let file = file.unwrap_or_else(|| PathBuf::from(DEFAULT_FILE));
    let text = match std::fs::read_to_string(&file) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("lang-budget: {}: {e}", file.display());
            return ExitCode::from(2);
        }
    };
    let n = count_tokens(&text);
    println!("{}: {n} tokens (limit {limit}, {} bytes / {BYTES_PER_TOKEN})", file.display(), text.len());
    if n > limit {
        eprintln!("lang-budget: {} is {} tokens over the limit", file.display(), n - limit);
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::count_tokens;

    #[test]
    fn fixed_strings() {
        assert_eq!(count_tokens(""), 0);
        assert_eq!(count_tokens("fn main() {}\n"), 4);
        // 'é' is 2 bytes, '→' is 3: 5 bytes
        assert_eq!(count_tokens("é→"), 2);
    }

    #[test]
    fn rounds_up() {
        assert_eq!(count_tokens("abc"), 1);
        assert_eq!(count_tokens("abcd"), 1);
        assert_eq!(count_tokens("abcde"), 2);
    }
}
