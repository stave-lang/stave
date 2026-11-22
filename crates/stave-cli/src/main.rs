//! The `stave` command (contracts/cli.md). Logic lives in the library (`stave_cli`).

// System allocator: mimalloc measured ~5x slower here on cold multi-threaded runs (57 ms vs 11 ms
// for `stave check` on 100k lines at 16 threads).

#[cfg(feature = "alloc-count")]
#[global_allocator]
static ALLOC: stave_driver::prof::CountingAlloc = stave_driver::prof::CountingAlloc;

fn main() {
    let code = stave_cli::guarded(|| {
        let mut argv = Vec::new();
        for a in std::env::args_os().skip(1) {
            match a.into_string() {
                Ok(s) => argv.push(s),
                Err(a) => {
                    let msg =
                        format!("stave: argument {a:?} is not valid UTF-8\nrun `stave --help` for usage\n");
                    let _ = std::io::Write::write_all(&mut std::io::stderr(), msg.as_bytes());
                    return stave_cli::EXIT_USAGE;
                }
            }
        }
        // stderr unlocked: the panic hook writes to it from whichever thread panics (the
        // pipeline runs on pool threads), and a lock held here for the whole run would block
        // that write forever
        stave_cli::run(&argv, &mut std::io::stdout().lock(), &mut std::io::stderr())
    });
    std::process::exit(code);
}
