//! The check-removal pass on real front-end IR. `tests/conformance/run/bounds_proofs.stv`,
//! lowered by the driver, keeps exactly the checks its look-alike functions need, and the same
//! ones in a debug and in a release build. Over the whole run suite, every function a release
//! build lowers has the checked IR the debug build gives it, so both keep the same checks. The
//! two programs that pin this in the `ir` dump goldens are the run tests of the same name.
//! Removal changes no program: every run-suite program gives the same output, exit status, and
//! trap in the reference interpreter with all its checks and with the proven ones removed. And
//! on front-end IR too, the witness checker refuses every corrupted witness and every made-up
//! witness for a check a look-alike function keeps.

use std::path::{Path, PathBuf};

#[path = "../../../tests/support/checked_ir.rs"]
mod checked_ir;
#[allow(dead_code)]
#[path = "../../stave-ir/tests/common/forge.rs"]
mod forge;

use stave_driver::interp::{self, Limits};
use stave_driver::pipeline::{self, Goal, Options, Output};
use stave_driver::shipped::Library;
use stave_ir::bounds::Proof;
use stave_ir::{Module, Op};

fn repo() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
}

/// The pipeline through lowering (no code generation) in debug or release mode.
fn lowered(root: &Path, release: bool) -> Output {
    let mut opts = Options::new(Goal::Check).threads(Some(2));
    opts.lower = true;
    opts.release = release;
    opts.target = Some(stave_target::Triple::parse("x86_64-linux").unwrap());
    pipeline::run(root, &Library::in_dir(repo()), &opts).unwrap()
}

/// Checks removed per function path of `m` (the module `o` lowered).
fn removed_per_fn(o: &Output, m: &Module) -> std::collections::BTreeMap<String, usize> {
    let mut out = std::collections::BTreeMap::new();
    for p in &o.removed_checks {
        *out.entry(m.funcs[p.witness.func.index()].path.clone()).or_default() += p.proofs.len();
    }
    out
}

fn has_body(m: &Module, path: &str) -> bool {
    m.funcs.iter().any(|f| f.path == path && f.body.is_some())
}

/// The checked IR of the program, and the rule that removed each check, per function path.
fn checked(file: &str, release: bool) -> (Module, Vec<(String, Proof)>) {
    let out = lowered(&repo().join(file), release);
    assert!(!out.has_errors(), "{file}: {:?}", out.diagnostics);
    let m = out.module.expect("checked IR");
    stave_ir::verify_module(&m).unwrap_or_else(|e| panic!("{e:?}"));
    stave_ir::bounds::audit(&m).unwrap_or_else(|u| panic!("unguarded after removal: {u:?}"));
    let mut proofs = Vec::new();
    for p in &out.removed_checks {
        assert_eq!(p.proofs.len(), p.witness.removed.len());
        let path = &m.funcs[p.witness.func.index()].path;
        proofs.extend(p.proofs.iter().map(|&proof| (path.clone(), proof)));
    }
    (m, proofs)
}

fn kept(m: &Module, path: &str) -> usize {
    let f = m.funcs.iter().find(|f| f.path == path).unwrap_or_else(|| panic!("no fn {path}"));
    let b = f.body.as_ref().unwrap();
    b.blocks
        .iter()
        .flat_map(|bd| bd.insts.indices())
        .filter(|&i| b.insts.op[i as usize] == Op::BoundsCheck)
        .count()
}

#[test]
fn proofs_and_kept_checks() {
    for release in [false, true] {
        proofs_and_kept_checks_in(release);
    }
}

fn proofs_and_kept_checks_in(release: bool) {
    let (m, proofs) = checked("tests/conformance/run/bounds_proofs.stv", release);
    let of = |path: &str| -> Vec<Proof> { proofs.iter().filter(|p| p.0 == path).map(|p| p.1).collect() };
    for (path, n) in [
        ("main.dot", 0),
        ("main.dot_or_zero", 0),
        ("main.rises", 0),
        ("main.tail_sum", 0),
        ("main.pair_sum", 0),
        ("main.window_sum", 0),
        ("main.chunk_sum", 0),
        ("main.quad_sum", 0),
        ("main.le_only", 1),
        ("main.eq_on_one_path", 1),
        ("main.next_under_len", 1),
        ("main.prev_wrapping", 1),
        ("main.at_offset", 1),
        ("main.chunk_le_sat", 1),
        ("main.chunk_short_test", 1),
        ("main.chunk_wrapping", 1),
        ("main.chunk_other", 1),
        ("main.chunk_shrunk", 1),
        ("main.chunk_moved", 1),
        ("main.at_required", 0),
        ("main.mask_n", 0),
        ("main.mod_n", 0),
        ("main.mask_len", 0),
        ("main.mod_len", 0),
        ("main.half", 0),
        ("main.by_byte", 0),
        ("main.mask_wide", 1),
        ("main.mod_wide", 1),
        ("main.by_u16", 1),
        ("main.by_signed", 1),
    ] {
        assert_eq!(kept(&m, path), n, "{path}: {:?}", of(path));
    }
    assert!(
        matches!(of("main.dot")[..], [Proof::LoopBound { .. }, Proof::LenEq { .. }]),
        "{:?}",
        of("main.dot")
    );
    assert!(matches!(of("main.dot_or_zero")[..], [Proof::LoopBound { .. }, Proof::LenEq { .. }]));
    assert!(matches!(of("main.rises")[..], [Proof::LoopBound { .. }, Proof::Offset { k: -1, .. }]));
    assert!(matches!(of("main.tail_sum")[..], [Proof::Offset { k: 1, .. }]));
    assert!(matches!(of("main.pair_sum")[..], [Proof::Offset { k: 0, .. }, Proof::Offset { k: 1, .. }]));
    // `i + 4 <= xs.len()` proves the vector's last lane `i + 3`, and every scalar read below it
    assert!(matches!(of("main.chunk_sum")[..], [Proof::Offset { k: 3, .. }]), "{:?}", of("main.chunk_sum"));
    assert!(matches!(of("main.quad_sum")[..], [Proof::Offset { k: 0, .. }, .., Proof::Offset { k: 3, .. }]));
    // `&xs[0..n]` has length `n` and tests `n <= xs.len()`
    assert!(
        matches!(of("main.window_sum")[..], [Proof::Offset { k: 0, .. }, Proof::Offset { k: 0, .. }]),
        "{:?}",
        of("main.window_sum")
    );
    // the clause's `cbr` is the only way into the body: `i < xs.len()` holds there
    assert!(matches!(of("main.at_required")[..], [Proof::LoopBound { .. }]), "{:?}", of("main.at_required"));
    // bounded by the bit op or the byte's type alone
    for path in ["main.mask_n", "main.mod_n", "main.mask_len", "main.mod_len", "main.half", "main.by_byte"] {
        assert!(matches!(of(path)[..], [Proof::Offset { k: 0, .. }]), "{path}: {:?}", of(path));
    }
}

/// Package dirs and single `.stv` files directly under `run/`, sorted.
fn run_programs() -> Vec<PathBuf> {
    let dir = repo().join("tests/conformance/run");
    let mut v: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            let name = p.file_name().unwrap().to_string_lossy();
            !name.starts_with('.')
                && !name.starts_with('_')
                && (p.join("stave.toml").is_file() || p.extension().is_some_and(|x| x == "stv"))
        })
        .collect();
    v.sort();
    v
}

/// The lowered IR of a checked program before the removal pass: every check in place.
fn unremoved(out: &Output) -> Option<Module> {
    stave_sema::lower(&out.program, out.typeck.as_ref()?, stave_sema::Mode::Debug).ok()
}

/// The `check.bounds` of the fns lowering made: an unchecked entry made as a fn of its own is a
/// copy of its fn's body, kept checks included, and is not counted.
fn count_checks(m: &Module) -> usize {
    m.funcs
        .iter()
        .filter(|f| f.instance.0.last() != Some(&stave_ir::InstanceId::UNCHECKED_ENTRY))
        .filter_map(|f| f.body.as_ref())
        .map(|b| b.insts.op.iter().filter(|op| **op == Op::BoundsCheck).count())
        .sum()
}

fn data_files(p: &Path) -> Vec<(String, Vec<u8>)> {
    if !p.is_dir() {
        return Vec::new();
    }
    std::fs::read_dir(p)
        .unwrap()
        .flatten()
        .filter(|e| e.path().is_file())
        .map(|e| (e.file_name().to_string_lossy().into_owned(), std::fs::read(e.path()).unwrap()))
        .collect()
}

/// The interpreter lays out, loads, and stores a value nested hundreds of levels deep (the
/// `W256<Loud>` of `drop_deep_type`) on a small native stack.
#[test]
fn deep_values_need_no_deep_stack() {
    let p = repo().join("tests/conformance/run/drop_deep_type.stv");
    let out = lowered(&p, false);
    let m = out.tracked.or(out.module).expect("checked IR");
    let main = stave_driver::build::main_path(&out.program).unwrap();
    let want = std::fs::read(repo().join("tests/conformance/run/drop_deep_type.stdout")).unwrap();
    let target = stave_target::Triple::parse("x86_64-linux").unwrap();
    let ran = std::thread::Builder::new()
        .stack_size(256 << 10)
        .spawn(move || interp::run(&m, &main, target, &[], &[], Limits::default()).map(|r| r.ran))
        .unwrap()
        .join()
        .unwrap()
        .unwrap();
    assert_eq!((ran.exit, ran.stdout), (0, want));
}

/// Over the run suite, per program:
///
/// - Debug = release in which checks remain: each function the release build lowers (what
///   `main` reaches, the shipped std and runtime included) has the ops of the debug build's
///   function of that path, and so the same `check.bounds`.
/// - Removing the proven checks changes nothing the program does: the lowered IR with every
///   check and the checked IR without the proven ones print the same, exit the same, and trap
///   the same in the reference interpreter (which bounds-checks every access of its own memory,
///   so an access that lost a check it needed would fault).
#[test]
fn modes_agree_and_removal_changes_no_program() {
    use rayon::prelude::*;
    let target = stave_target::Triple::parse("x86_64-linux").unwrap();
    // Per program that lowers: release fns, checks kept, checks removed, whether it traps.
    let per_program: Vec<Option<[usize; 4]>> = run_programs()
        .par_iter()
        .map(|p| {
            let (dbg, rel) = (lowered(p, false), lowered(p, true));
            let (Some(d), Some(r)) = (&dbg.module, &rel.module) else {
                assert_eq!(dbg.module.is_some(), rel.module.is_some(), "{}", p.display());
                return None;
            };
            let differ = checked_ir::not_in(r, d);
            assert!(differ.is_empty(), "{}: release differs from debug in {differ:?}", p.display());
            let (kd, kr) = (checked_ir::kept_checks(d), checked_ir::kept_checks(r));
            for (path, n) in &kr {
                assert!(kd.get(path).is_some_and(|m| m >= n), "{}: {path} keeps {n} checks", p.display());
            }
            let removed = |o: &Output| o.removed_checks.iter().map(|x| x.bounds_removed()).sum::<usize>();
            // a debug build also lowers fns `main` does not reach: compare the fns both lower
            let (fd, fr) = (removed_per_fn(&dbg, d), removed_per_fn(&rel, r));
            let both = |path: &String| has_body(d, path) && has_body(r, path);
            for path in fd.keys().chain(fr.keys()).filter(|x| both(x)) {
                let (a, b) = (fd.get(path).copied().unwrap_or(0), fr.get(path).copied().unwrap_or(0));
                assert_eq!(a, b, "{}: {path} removes {a} checks in debug, {b} in release", p.display());
            }

            let before = unremoved(&dbg).unwrap_or_else(|| panic!("{}: does not lower", p.display()));
            assert_eq!(count_checks(&before) - count_checks(d), removed(&dbg), "{}", p.display());
            let main = stave_driver::build::main_path(&dbg.program)?;
            let files = data_files(p);
            let run =
                |m: &Module| interp::run(m, &main, target, &[], &files, Limits::default()).map(|r| r.ran);
            let (want, got) = (run(&before), run(d));
            assert_eq!(got, want, "{}: removal changed the run", p.display());
            let fns = r.funcs.iter().filter(|f| f.body.is_some()).count();
            Some([fns, kr.values().sum(), removed(&rel), usize::from(want.is_ok_and(|r| r.exit == 101))])
        })
        .collect();
    let ran: Vec<[usize; 4]> = per_program.into_iter().flatten().collect();
    let total = |k: usize| ran.iter().map(|x| x[k]).sum::<usize>();
    let (fns, kept, removed, trapped) = (total(0), total(1), total(2), total(3));
    assert!(ran.len() >= 50, "only {} programs lowered", ran.len());
    assert!(fns > 0 && removed > 0 && kept > 0, "{fns} fns, {removed} removed, {kept} kept");
    assert!(trapped >= 5, "only {trapped} programs trap");
}

/// A fn reached only from a `test fn` is lowered by a debug build (every non-generic fn is a root)
/// and not by a release build: its removed check counts in debug only, and the modes still agree
/// on every fn both lower.
#[test]
fn a_debug_only_fn_keeps_its_proof() {
    let p = repo().join("tests/conformance/run/bounds_test_only.stv");
    let (dbg, rel) = (lowered(&p, false), lowered(&p, true));
    let (Some(d), Some(r)) = (&dbg.module, &rel.module) else { panic!("does not lower") };
    let (fd, fr) = (removed_per_fn(&dbg, d), removed_per_fn(&rel, r));
    assert_eq!(fd.get("main.total").copied(), Some(1), "{fd:?}");
    assert!(has_body(d, "main.total") && !has_body(r, "main.total"));
    assert!(!fr.contains_key("main.total"), "{fr:?}");
}

/// On front-end IR (the program, the shipped std, and the runtime): the checker accepts every
/// witness the pass writes and refuses every corruption of one; no made-up witness removes a
/// check a look-alike function keeps.
#[test]
fn no_forged_witness_passes() {
    for (file, look_alikes) in [
        ("tests/conformance/run/bounds_removed.stv", &[][..]),
        ("tests/conformance/run/bounds_kept.stv", &["main.first_n", "main.pairs"][..]),
        (
            "tests/conformance/run/bounds_proofs.stv",
            &[
                "main.le_only",
                "main.eq_on_one_path",
                "main.next_under_len",
                "main.prev_wrapping",
                "main.at_offset",
                "main.chunk_le_sat",
                "main.chunk_short_test",
                "main.chunk_wrapping",
                "main.chunk_other",
                "main.chunk_shrunk",
                "main.chunk_moved",
                "main.mask_wide",
                "main.mod_wide",
                "main.by_u16",
                "main.by_signed",
            ][..],
        ),
    ] {
        let checked = lowered(&repo().join(file), false);
        let m = unremoved(&checked).unwrap_or_else(|| panic!("{file} does not lower"));
        let corrupted = forge::mutants_are_refused(&m);
        assert!(corrupted >= 20, "{file}: only {corrupted} corrupted witnesses");
        let mut forged = 0;
        for path in look_alikes {
            let f = m.find_func(path).unwrap_or_else(|| panic!("{file}: no fn {path}"));
            assert_eq!(forge::kept_checks(&m, f).len(), 1, "{file}: {path}");
            forged += forge::forgeries_are_refused(&m, f);
        }
        assert_eq!(forged == 0, look_alikes.is_empty(), "{file}");
    }
}

/// The two programs whose remaining checks the `ir` dump goldens pin per mode
/// (`tests/conformance/dumps/ir_bounds_*`: `ir.txt` debug, `ir.release.txt` release) are the
/// run tests of the same name, which run natively in both modes: the dump case differs only in
/// its `//~` annotation lines. In both modes `bounds_removed` keeps no check and `bounds_kept`
/// keeps the two that can fail.
#[test]
fn dump_cases_are_the_run_programs() {
    let code = |p: PathBuf| -> String {
        let text = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        text.lines().filter(|l| !l.starts_with("//~")).map(|l| format!("{l}\n")).collect()
    };
    for (name, want) in
        [("bounds_removed", &[][..]), ("bounds_kept", &[("main.first_n", 1), ("main.pairs", 1)][..])]
    {
        let run = repo().join(format!("tests/conformance/run/{name}.stv"));
        let dump = repo().join(format!("tests/conformance/dumps/ir_{name}/input/src/main.stv"));
        assert_eq!(code(run.clone()), code(dump), "{name}: the dump case is not the run program");
        for release in [false, true] {
            let (m, _) = checked(&format!("tests/conformance/run/{name}.stv"), release);
            let kept: Vec<(String, usize)> = checked_ir::kept_checks(&m)
                .into_iter()
                .filter(|(path, _)| path.starts_with("main."))
                .collect();
            let want: Vec<(String, usize)> = want.iter().map(|(p, n)| (p.to_string(), *n)).collect();
            assert_eq!(kept, want, "{name}, release: {release}");
        }
    }
}

/// The contract failure calls left in function `path` of `m`: its kept contract checks.
fn kept_contracts(m: &Module, path: &str) -> usize {
    let f = m.funcs.iter().find(|f| f.path == path).unwrap_or_else(|| panic!("no fn {path}"));
    let b = f.body.as_ref().unwrap();
    (0..b.insts.len() as u32)
        .filter(|&i| {
            b.insts.op[i as usize] == Op::Call
                && matches!(b.args(i).first(), Some(stave_ir::Operand::Func(g))
                    if stave_ir::known::is_contract_failed(&m.funcs[g.index()]))
        })
        .count()
}

/// `tests/conformance/run/contract_removed.stv` keeps exactly the contract checks the facts do
/// not decide, the same in a debug and a release build: of `clamp` only the `requires` (its six
/// `ensures` checks, two per return, are proved), of `within` the `result <= hi` at the first
/// and the last return.
#[test]
fn contract_checks_kept_and_removed() {
    use stave_ir::bounds::Clause;
    for release in [false, true] {
        let (m, proofs) = checked("tests/conformance/run/contract_removed.stv", release);
        let removed = |path: &str| {
            proofs.iter().filter(|p| p.0 == path && p.1 == Proof::Clause(Clause::Ensures)).count()
        };
        for (path, kept, gone) in [("main.clamp", 1, 6), ("main.within", 2, 4)] {
            assert_eq!((kept_contracts(&m, path), removed(path)), (kept, gone), "{path}, release: {release}");
        }
    }
}

/// On front-end IR, the checker accepts every contract witness the pass writes (removals and
/// refutations), refuses every corruption of one, and refuses every made-up removal or
/// refutation of a contract check the pass leaves alone.
#[test]
fn no_forged_contract_witness_passes() {
    let file = "tests/conformance/run/contract_removed.stv";
    let checked = lowered(&repo().join(file), false);
    let m = unremoved(&checked).unwrap_or_else(|| panic!("{file} does not lower"));
    assert!(forge::mutants_are_refused(&m) >= 20);
    let f = m.find_func("main.within").expect("main.within");
    assert_eq!(forge::contract_checks(&m, f).len(), 6);
    assert!(forge::contract_forgeries_are_refused(&m, f) > 100);
}
