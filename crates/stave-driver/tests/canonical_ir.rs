//! The checked IR the driver lowers is canonical (`stave_ir::canon`): renumbering it again
//! changes nothing, the same module stored with its functions in another order renumbers back
//! to it, and the sema shuffle knob (`Options::shuffle_seed`) gives an equal module.

use std::path::{Path, PathBuf};

use stave_driver::pipeline::{self, Goal, Options};
use stave_driver::shipped::Library;
use stave_ir::canon::{canonicalize, is_identity, permute_by_func};
use stave_ir::{FuncId, InstanceId, Module, Operand};

fn repo() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
}

/// Programs with `requires` callers (unchecked entries), closures, statics and generics.
const PROGRAMS: [&str; 5] = [
    "tests/conformance/run/contract_calls.stv",
    "tests/conformance/run/contract_calls_release.stv",
    "tests/conformance/run/contract_borrow.stv",
    "tests/conformance/run/bounds_proofs.stv",
    "tests/conformance/run/const_image_value.stv",
];

/// A generated package, made once for all tests of this file.
fn corpus() -> PathBuf {
    static DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    DIR.get_or_init(|| {
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("canon2k");
        gen_corpus::generate(&gen_corpus::Options {
            lines: 2_000,
            seed: 7,
            out: dir.clone(),
            mix: gen_corpus::Mix::default_mix().unwrap(),
            threads: 0,
            force: true,
        })
        .unwrap();
        dir
    })
    .clone()
}

/// The checked IR and the tracked (borrow side table) module of `root`.
fn lowered(root: &Path, release: bool, shuffle: Option<u64>) -> (Module, Module) {
    let mut opts = Options::new(Goal::Check).threads(Some(2));
    opts.lower = true;
    opts.tracked = true;
    opts.release = release;
    opts.shuffle_seed = shuffle;
    opts.target = Some(stave_target::Triple::parse("x86_64-linux").unwrap());
    let out = pipeline::run(root, &Library::in_dir(repo()), &opts).unwrap();
    assert!(!out.has_errors() && out.failure.is_none(), "{}: {:?}", root.display(), out.failure);
    (out.module.expect("checked IR"), out.tracked.expect("tracked IR"))
}

/// `m` with its functions stored in reverse order, every reference and table moved.
fn reversed(m: &Module) -> Module {
    let n = m.funcs.len();
    let remap: Vec<FuncId> = (0..n).map(|i| FuncId((n - 1 - i) as u32)).collect();
    let mut p = m.clone();
    p.funcs.reverse();
    p.main = m.main.map(|g| remap[g.index()]);
    for f in &mut p.funcs {
        for o in f.body.iter_mut().flat_map(|b| &mut b.operands) {
            if let Operand::Func(g) = o {
                *g = remap[g.index()];
            }
        }
    }
    if !p.stated.is_empty() {
        permute_by_func(&mut p.stated, &remap);
    }
    if !p.clauses.is_empty() {
        permute_by_func(&mut p.clauses, &remap);
    }
    p
}

/// The borrow side table, which module equality leaves out (the text does not carry it).
fn side_table(m: &Module) -> impl PartialEq + std::fmt::Debug + '_ {
    let bodies: Vec<_> = m.funcs.iter().map(|f| f.body.as_ref().map(|b| (&b.borrows, &b.events))).collect();
    (&m.frozen, &m.clauses, bodies)
}

fn assert_canonical(what: &str, m: &Module) {
    let mut again = m.clone();
    assert!(is_identity(&canonicalize(&mut again)), "{what}: functions moved");
    assert!(again == *m, "{what}: renumbering a lowered module changed it");
    assert!(side_table(&again) == side_table(m), "{what}: renumbering changed the side table");
    let mut back = reversed(m);
    canonicalize(&mut back);
    assert!(back == *m, "{what}: the reversed module does not renumber back");
    assert!(side_table(&back) == side_table(m), "{what}: the reversed side table does not renumber back");
}

#[test]
fn lowered_modules_are_canonical() {
    let mut roots: Vec<PathBuf> = PROGRAMS.iter().map(|p| repo().join(p)).collect();
    roots.push(corpus());
    let mut moved = 0;
    for root in &roots {
        for release in [false, true] {
            let (m, tracked) = lowered(root, release, None);
            let what = format!("{} release={release}", root.display());
            assert_canonical(&what, &m);
            assert_canonical(&format!("{what} tracked"), &tracked);
            // lowering appends the unchecked entries after every other fn; canonical, one is
            // followed by a fn that is not one
            let entry = |f: &stave_ir::Func| f.instance.0.last() == Some(&InstanceId::UNCHECKED_ENTRY);
            moved += m.funcs.windows(2).filter(|w| entry(&w[0]) && !entry(&w[1])).count();
        }
    }
    assert!(moved > 0, "no unchecked entry moved");
}

#[test]
fn sema_shuffle_gives_an_equal_module() {
    for root in [repo().join(PROGRAMS[0]), corpus()] {
        for release in [false, true] {
            let (m, _) = lowered(&root, release, None);
            let (s, _) = lowered(&root, release, Some(3));
            assert!(m == s, "{} release={release}: the shuffled lowering differs", root.display());
        }
    }
}
