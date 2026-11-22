//! The two layout implementations agree: the type checker's (`size_of`, `align_of`, std.soa
//! columns: `stave_sema::traits::layout_of`, `field_offsets`) and the IR's (the interpreter and
//! the backends: `stave_ir::layout::Layouts`). Each param type of `probe` below, and every part
//! of it down to the scalars, gets the same size, alignment, field offsets, and array stride
//! from both.
use std::path::{Path, PathBuf};

use stave_ir::layout::{Layouts, Shape};
use stave_sema::types::{ConstArg, Interner, TypeKind};
use stave_sema::{Config, DeclId, DeclKind, Mode, Program, TypeId, TypeTable, Typeck};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

const SRC: &str = "import std.vec;

struct P { a: u8, b: u64, c: u16 }
struct Z {}
struct W<T> { x: u8, v: T, y: u8 }
struct N { m: [[u16; 3]; 2], t: (u8, u32), w: W<[u8; 3]>, z: Z }
struct V { a: u8, v: f32x4, b: u16, m: mask32x4 }
struct A { e: [E; 2], f: [[[u8; 3]; 2]; 2], g: [W<u16>; 3] }
view struct R { r: &u32, s: &[u8], t: &str, p: *const u8, f: fn(u32) -> u32, q: Option<&u64> }
enum E { A, B(u8, u64), C { x: u16, y: [u32; 2] }, D(W<u16>) }
enum G<T> { None, Some(T), Two(T, u8) }
enum U { X, Y }

fn probe(a: P, b: Z, c: W<u8>, d: W<u64>, e: W<W<u16>>, f: N, g: V, h: E, i: G<u32>, j: G<P>,
    k: G<vec.Vec<u8>>, l: [[[u8; 3]; 2]; 2], m: (u8, u16, u64), n: [f32x4; 3], o: mask32x4,
    q: A, r: R, s: Option<u8>, t: Result<u16, u64>, u: bool, v: char, w: (), x: vec.Vec<u32>,
    y: U, z: G<G<(u8, u32)>>, aa: (W<i8>, [i64x2; 2]), ab: u16x8, ac: f64) {
}

fn main() -> i32 {
    return 0;
}
";

fn load(dir: &Path) -> (Program, Typeck, stave_ir::Module) {
    std::fs::create_dir_all(dir.join("src")).unwrap();
    std::fs::write(dir.join("stave.toml"), "name = \"main\"\nkind = \"bin\"\n").unwrap();
    std::fs::write(dir.join("src/main.stv"), SRC).unwrap();
    let mut c = Config::new(dir).with_shipped(repo().join("std"), repo().join("runtime"));
    c.target = stave_target::Triple::parse("x86_64-linux").ok();
    let p = Program::load(&c);
    let tc = stave_sema::check(&p);
    let d = tc.all_diagnostics(&p);
    assert!(d.is_empty(), "{}", stave_diag::emit_human(&d, &p.source_map).unwrap_or_default());
    let m = stave_sema::lower(&p, &tc, Mode::Debug).unwrap_or_else(|e| panic!("{} lowering errors", e.len()));
    (p, tc, m)
}

struct Cx<'a> {
    tc: &'a Typeck,
    tys: TypeTable,
    ir: &'a Layouts,
    /// (path, what) per comparison made
    checked: Vec<String>,
}

impl Cx<'_> {
    /// Compares sema type `t` with IR type `it` at `path`, then their parts.
    fn compare(&mut self, t: TypeId, it: stave_ir::TypeId, path: &str) {
        let sema = stave_sema::traits::layout_of(&mut self.tys, &self.tc.sigs, t);
        let ir = self.ir.get_layout(it).unwrap_or_else(|e| panic!("{path}: no IR layout: {e}"));
        assert_eq!(sema, Some((ir.size, ir.align)), "{path}: (size, align), sema vs IR");
        self.checked.push(format!("{path}: size {} align {}", ir.size, ir.align));
        match ir.shape.clone() {
            Shape::Fields(fs) => {
                let parts = self.parts(t, path);
                let offs = stave_sema::traits::field_offsets(&mut self.tys, &self.tc.sigs, t)
                    .unwrap_or_else(|| panic!("{path}: no sema field offsets"));
                assert_eq!(parts.len(), 1, "{path}: one variant");
                let want: Vec<u64> = fs.iter().map(|f| f.0).collect();
                assert_eq!(offs[0], want, "{path}: field offsets, sema vs IR");
                assert_eq!(parts[0].len(), fs.len(), "{path}: field count");
                for (k, (&st, &(_, ft))) in parts[0].iter().zip(fs.iter()).enumerate() {
                    self.compare(st, ft, &format!("{path}.{k}"));
                }
            }
            Shape::Enum(vs) => {
                let parts = self.parts(t, path);
                let offs = stave_sema::traits::field_offsets(&mut self.tys, &self.tc.sigs, t)
                    .unwrap_or_else(|| panic!("{path}: no sema field offsets"));
                assert_eq!(parts.len(), vs.len(), "{path}: variant count");
                for (vi, v) in vs.iter().enumerate() {
                    let want: Vec<u64> = v.iter().map(|f| f.0).collect();
                    assert_eq!(offs[vi], want, "{path}: variant {vi} field offsets, sema vs IR");
                    for (k, (&st, &(_, ft))) in parts[vi].iter().zip(v.iter()).enumerate() {
                        self.compare(st, ft, &format!("{path}#{vi}.{k}"));
                    }
                }
            }
            Shape::Array { elem, stride, len } => {
                let TypeKind::Array { elem: se, len: ConstArg::Value(n) } = self.tys.kind(t).clone() else {
                    panic!("{path}: IR array, sema {:?}", self.tys.kind(t));
                };
                assert_eq!(n, len, "{path}: array length");
                let es = stave_sema::traits::layout_of(&mut self.tys, &self.tc.sigs, se).map(|l| l.0);
                assert_eq!(es, Some(stride), "{path}: stride, sema element size vs IR");
                self.compare(se, elem, &format!("{path}[_]"));
            }
            Shape::Scalar | Shape::Fat | Shape::Bytes => {}
        }
    }

    /// The field types of each variant of sema type `t` (one variant for a struct or tuple).
    fn parts(&mut self, t: TypeId, path: &str) -> Vec<Vec<TypeId>> {
        match self.tys.kind(t).clone() {
            TypeKind::Adt { def, args } => {
                let adt = self.tc.sigs.adt(def).unwrap_or_else(|| panic!("{path}: no ADT"));
                adt.variants
                    .iter()
                    .map(|v| v.fields.iter().map(|f| self.tys.subst(f.ty, &args, None)).collect())
                    .collect()
            }
            TypeKind::Tuple(ts) => vec![ts.to_vec()],
            k => panic!("{path}: IR record, sema {k:?}"),
        }
    }
}

#[test]
fn sema_and_ir_layouts_agree() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("layout_cross");
    let (p, tc, m) = load(&dir);
    let probe = (0..p.decls.len() as u32)
        .map(DeclId)
        .find(|&d| {
            p.decl(d).is_some_and(|x| {
                x.kind == DeclKind::Fn
                    && p.strings.get(x.name) == "probe"
                    && p.module(x.module).is_some_and(|m| p.packages[m.package.idx()].is_root)
            })
        })
        .expect("probe decl");
    let params = tc.sigs.fn_sig(probe).expect("probe sig").params.clone();
    let f = m.funcs.iter().find(|f| f.path == "main.probe").expect("probe lowered");
    assert_eq!(f.params.len(), params.len(), "probe params, IR vs sema");
    let lays = Layouts::compute(&m.types);
    let mut cx = Cx { tc: &tc, tys: tc.types.clone(), ir: &lays, checked: Vec::new() };
    for (i, (&t, &it)) in params.iter().zip(f.params.iter()).enumerate() {
        cx.compare(t, it, &format!("param {i}"));
    }
    // every param and its parts: generic instances, enums with payloads, nested arrays,
    // vectors and masks, references and fn values
    assert!(cx.checked.len() >= 100, "{} comparisons:\n{}", cx.checked.len(), cx.checked.join("\n"));
}
