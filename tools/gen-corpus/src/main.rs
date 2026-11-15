//! `gen-corpus --lines N --seed S -o DIR [--mix PATH] [--threads T] [--force]`, or
//! `gen-corpus --pair --lines N --seed S -o DIR [--units K] [--reps R] [--force]`.

use gen_corpus::{Mix, Options, PairOptions, generate, generate_pair};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

const USAGE: &str = "usage: gen-corpus --lines N --seed S -o DIR [--mix PATH] [--threads T] [--force]
       gen-corpus --pair --lines N --seed S -o DIR [--units K] [--reps R] [--force]";

enum Req {
    Corpus(Box<Options>),
    Pair(PairOptions, PathBuf, bool),
}

fn parse_args() -> Result<Req, String> {
    let mut lines = None;
    let mut seed = None;
    let mut out = None;
    let mut mix_path: Option<PathBuf> = None;
    let mut threads = 0usize;
    let mut force = false;
    let mut pair = false;
    let mut units = 0u32;
    let mut reps = 1u32;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut val = |name: &str| it.next().ok_or_else(|| format!("{name} needs a value"));
        match a.as_str() {
            "--lines" => lines = Some(val("--lines")?.parse::<u64>().map_err(|e| format!("--lines: {e}"))?),
            "--seed" => seed = Some(val("--seed")?.parse::<u64>().map_err(|e| format!("--seed: {e}"))?),
            "-o" | "--out" => out = Some(PathBuf::from(val("-o")?)),
            "--mix" => mix_path = Some(PathBuf::from(val("--mix")?)),
            "--threads" => {
                threads = val("--threads")?.parse::<usize>().map_err(|e| format!("--threads: {e}"))?
            }
            "--force" => force = true,
            "--pair" => pair = true,
            "--units" => units = val("--units")?.parse::<u32>().map_err(|e| format!("--units: {e}"))?,
            "--reps" => reps = val("--reps")?.parse::<u32>().map_err(|e| format!("--reps: {e}"))?,
            "-h" | "--help" => return Err(String::new()),
            _ => return Err(format!("unknown argument {a:?}")),
        }
    }
    let lines = lines.ok_or("--lines is required")?;
    // Plans take about 12 bytes per line (120 MB at 10M lines); the cap keeps memory under 2 GB.
    if lines == 0 || lines > 100_000_000 {
        return Err("--lines must be in 1..=100000000".into());
    }
    if pair {
        if units > 4096 || reps == 0 {
            return Err("--units must be in 0..=4096 and --reps at least 1".into());
        }
        let seed = seed.ok_or("--seed is required")?;
        let out = out.ok_or("-o is required")?;
        return Ok(Req::Pair(PairOptions { lines, seed, units, reps }, out, force));
    }
    let mix = match mix_path {
        Some(p) => {
            let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
            Mix::parse(&text)?
        }
        None => Mix::default_mix()?,
    };
    Ok(Req::Corpus(Box::new(Options {
        lines,
        seed: seed.ok_or("--seed is required")?,
        out: out.ok_or("-o is required")?,
        mix,
        threads,
        force,
    })))
}

fn main() -> ExitCode {
    let opts = match parse_args() {
        Ok(o) => o,
        Err(e) => {
            if !e.is_empty() {
                eprintln!("gen-corpus: {e}");
            }
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    let opts = match opts {
        Req::Corpus(o) => *o,
        Req::Pair(o, out, force) => {
            return match generate_pair(&o, &out, force) {
                Ok(s) => {
                    eprintln!(
                        "gen-corpus: pair: {} Stave lines, {} C lines, {} units, {} fns",
                        s.stave_lines, s.c_lines, s.units, s.fns
                    );
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("gen-corpus: {e}");
                    ExitCode::FAILURE
                }
            };
        }
    };
    let t0 = Instant::now();
    match generate(&opts) {
        Ok(s) => {
            let dt = t0.elapsed().as_secs_f64();
            eprintln!(
                "gen-corpus: {} lines, {} bytes, {} modules + main, {} items, {} fns in {:.3} s ({:.2}M lines/s)",
                s.lines,
                s.bytes,
                s.modules,
                s.items,
                s.fns,
                dt,
                s.lines as f64 / dt.max(1e-9) / 1e6
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("gen-corpus: {e}");
            ExitCode::FAILURE
        }
    }
}
