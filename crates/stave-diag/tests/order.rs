//! Sort key (package, file path, primary byte offset, code, message);
//! location-less diagnostics first by (code, message); identical (file, offset, code, message)
//! deduplicated; any insertion order and any thread count give the same output bytes.

mod common;

use proptest::prelude::*;
use stave_diag::{Diagnostic, FileId, Sink, SourceFiles, Span, canonical_order, emit_human, emit_json};

fn files() -> (SourceFiles, [FileId; 4]) {
    let text = "x\n".repeat(64);
    let mut f = SourceFiles::new();
    // Registered out of order on purpose: ids must not decide the order.
    let b = f.add("pkg-b", "src/a.stv", text.as_bytes()).expect("file");
    let a2 = f.add("pkg-a", "src/z.stv", text.as_bytes()).expect("file");
    let a1 = f.add("pkg-a", "src/b.stv", text.as_bytes()).expect("file");
    let a0 = f.add("pkg-a", "src/a.stv", text.as_bytes()).expect("file");
    (f, [a0, a1, a2, b])
}

fn unresolved(file: FileId, offset: u32, name: &str) -> Diagnostic {
    Diagnostic::new("E0101", None, &[("name", name)], Some(Span::new(file, offset, 1)), &[]).expect("E0101")
}

fn duplicate(file: FileId, offset: u32, name: &str, first: u32) -> Diagnostic {
    let holes = [("name", name), ("scope", "app")];
    Diagnostic::new(
        "E0105",
        None,
        &holes,
        Some(Span::new(file, offset, 1)),
        &[("first", Span::new(file, first, 1))],
    )
    .expect("E0105")
}

fn no_gpu(reason: &str) -> Diagnostic {
    Diagnostic::new("E1302", None, &[("reason", reason)], None, &[]).expect("E1302")
}

fn key(d: &Diagnostic) -> (String, Option<(u32, u32)>, String) {
    (d.code().to_string(), d.primary().map(|s| (s.file.0, s.offset)), d.message().to_string())
}

#[test]
fn canonical_key_order() {
    let (f, [a0, a1, a2, b]) = files();
    let input = vec![
        unresolved(b, 0, "q"),
        unresolved(a2, 0, "q"),
        duplicate(a0, 10, "z", 2),
        unresolved(a0, 10, "z"),
        unresolved(a0, 10, "a"),
        unresolved(a0, 4, "m"),
        unresolved(a1, 0, "q"),
        no_gpu("no device"),
        Diagnostic::new("E1301", None, &[("needed", "9"), ("available", "1")], None, &[]).expect("E1301"),
        no_gpu("daemon not reachable"),
    ];
    let out = canonical_order(input, &f);
    let got: Vec<_> = out.iter().map(key).collect();
    let want = vec![
        // location-less first, by (code, message)
        ("E1301".to_string(), None, out[0].message().to_string()),
        ("E1302".to_string(), None, "`--compiler gpu`: no usable GPU (daemon not reachable)".to_string()),
        ("E1302".to_string(), None, "`--compiler gpu`: no usable GPU (no device)".to_string()),
        // pkg-a/src/a.stv by offset, then code, then message
        ("E0101".to_string(), Some((a0.0, 4)), "cannot find `m` in this scope".to_string()),
        ("E0101".to_string(), Some((a0.0, 10)), "cannot find `a` in this scope".to_string()),
        ("E0101".to_string(), Some((a0.0, 10)), "cannot find `z` in this scope".to_string()),
        ("E0105".to_string(), Some((a0.0, 10)), out[6].message().to_string()),
        // then pkg-a/src/b.stv, pkg-a/src/z.stv, pkg-b/src/a.stv
        ("E0101".to_string(), Some((a1.0, 0)), "cannot find `q` in this scope".to_string()),
        ("E0101".to_string(), Some((a2.0, 0)), "cannot find `q` in this scope".to_string()),
        ("E0101".to_string(), Some((b.0, 0)), "cannot find `q` in this scope".to_string()),
    ];
    assert_eq!(got, want);
}

#[test]
fn duplicates_removed() {
    let (f, [a0, a1, ..]) = files();
    let input = vec![
        unresolved(a0, 3, "x"),
        unresolved(a0, 3, "x"),
        // same (file, offset, code, message), different related span: still a duplicate
        duplicate(a0, 5, "y", 1),
        duplicate(a0, 5, "y", 2),
        // differs in file, offset, or message: kept
        unresolved(a1, 3, "x"),
        unresolved(a0, 4, "x"),
        unresolved(a0, 3, "w"),
        no_gpu("no device"),
        no_gpu("no device"),
    ];
    let out = canonical_order(input, &f);
    assert_eq!(out.len(), 6);
    // The kept duplicate does not depend on input order: the smallest related span wins.
    let e0105 = out.iter().find(|d| d.code() == "E0105").expect("E0105 kept");
    assert_eq!(e0105.related()[1].span.offset, 1);
    let rev = canonical_order(vec![duplicate(a0, 5, "y", 2), duplicate(a0, 5, "y", 1)], &f);
    assert_eq!(rev[0].related()[1].span.offset, 1);
}

#[test]
fn unknown_files_sort_after_known_ones_without_panicking() {
    let (f, [a0, ..]) = files();
    let ghost = FileId(99);
    let out = canonical_order(vec![unresolved(ghost, 0, "g"), unresolved(a0, 9, "a")], &f);
    assert_eq!(out[0].primary().map(|s| s.file), Some(a0));
    assert!(emit_json(&out, &f).is_err(), "unknown file is an emit error, not a panic");
}

fn emit_all(diags: &[Diagnostic], f: &SourceFiles) -> (String, String) {
    (emit_human(diags, f).expect("human"), emit_json(diags, f).expect("json"))
}

/// One generated diagnostic: (kind, file index, offset, name index).
fn make(f: &[FileId; 4], (kind, file, offset, name): (u8, usize, u32, usize)) -> Diagnostic {
    const NAMES: [&str; 4] = ["a", "b", "ab", "é"];
    const REASONS: [&str; 4] =
        ["stage1 image missing", "no NVIDIA driver", "no device", "daemon not reachable"];
    let file = f[file % 4];
    let name = NAMES[name % 4];
    match kind % 3 {
        0 => unresolved(file, offset, name),
        1 => duplicate(file, offset, name, offset / 2),
        _ => no_gpu(REASONS[(offset as usize + name.len()) % 4]),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn order_independent_of_insertion_and_threads(
        specs in prop::collection::vec((0u8..3, 0usize..4, 0u32..16, 0usize..4), 0..48),
        perm_seed in any::<u64>(),
        threads in 1usize..8,
    ) {
        let (f, ids) = files();
        let diags: Vec<Diagnostic> = specs.iter().map(|s| make(&ids, *s)).collect();
        let reference = emit_all(&canonical_order(diags.clone(), &f), &f);

        // A deterministic shuffle from the seed (Fisher-Yates over an xorshift sequence).
        let mut shuffled = diags.clone();
        let mut x = perm_seed | 1;
        for i in (1..shuffled.len()).rev() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            shuffled.swap(i, (x % (i as u64 + 1)) as usize);
        }
        let sink = Sink::new();
        let chunk = shuffled.len().div_ceil(threads).max(1);
        std::thread::scope(|s| {
            for part in shuffled.chunks(chunk) {
                let sink = &sink;
                s.spawn(move || {
                    for d in part {
                        sink.push(d.clone());
                    }
                });
            }
        });
        let out = sink.finish(&f);
        prop_assert_eq!(emit_all(&out, &f), reference);

        // The output is sorted and free of duplicates.
        let mut seen = std::collections::BTreeSet::new();
        for d in &out {
            prop_assert!(seen.insert(key(d)), "duplicate {:?}", key(d));
        }
        let locless = out.iter().take_while(|d| d.primary().is_none()).count();
        prop_assert!(out[locless..].iter().all(|d| d.primary().is_some()));
    }
}
