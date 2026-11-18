//! Kinds and origins API: `kinds::kind_of` per type, `origins::of_sig` per origin
//! table row, `of_fn_type` (a call through a fn value borrows from every ref/view argument), and
//! `SigOrigins::weakened` (fn item → fn value).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use stave_sema::origins::{self, Flow, ParamClass, ResultOrigin};
use stave_sema::traits::World;
use stave_sema::types::{Interner, TypeKind};
use stave_sema::{Config, DeclId, DeclKind, Fs, Program, Typeck};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn mem(src: &str) -> (Program, Typeck) {
    let mut m = BTreeMap::new();
    m.insert(PathBuf::from("/m/stave.toml"), b"name = \"m\"\nkind = \"bin\"\n".to_vec());
    m.insert(PathBuf::from("/m/src/main.stv"), src.as_bytes().to_vec());
    for e in std::fs::read_dir(repo().join("std")).unwrap() {
        let e = e.unwrap().path();
        if e.extension().is_some_and(|x| x == "stv") {
            m.insert(PathBuf::from("/std").join(e.file_name().unwrap()), std::fs::read(&e).unwrap());
        }
    }
    for sub in ["", "x86_64", "aarch64"] {
        for e in std::fs::read_dir(repo().join("runtime").join(sub)).unwrap() {
            let e = e.unwrap().path();
            if e.is_file() {
                let rel = if sub.is_empty() {
                    PathBuf::from(e.file_name().unwrap())
                } else {
                    Path::new(sub).join(e.file_name().unwrap())
                };
                m.insert(PathBuf::from("/runtime").join(rel), std::fs::read(&e).unwrap());
            }
        }
    }
    let mut c = Config::new("/m").with_shipped("/std", "/runtime");
    c.fs = Fs::Mem(m);
    c.target = stave_target::Triple::parse("x86_64-linux").ok();
    let p = Program::load(&c);
    let tc = stave_sema::check(&p);
    (p, tc)
}

fn codes(p: &Program, tc: &Typeck) -> Vec<String> {
    tc.all_diagnostics(p).iter().map(|d| d.code().to_string()).collect()
}

fn root_fn(p: &Program, name: &str) -> DeclId {
    (0..p.decls.len() as u32)
        .map(DeclId)
        .find(|&d| {
            p.decl(d).is_some_and(|x| {
                x.kind == DeclKind::Fn
                    && p.strings.get(x.name) == name
                    && p.module(x.module).is_some_and(|m| p.packages[m.package.idx()].is_root)
            })
        })
        .expect("decl")
}

const SRC: &str = "view struct Lexer { src: &str, pos: usize }\n\
    fn by_ref(x: &u32, y: &u32) -> &u32 from y + y { return y; }\n\
    fn by_view(l: Lexer) -> Lexer from l { return l; }\n\
    fn by_ref_view(l: &mut Lexer, n: u32) -> &str from copy l + l { return l.src; }\n\
    fn owned(x: &u32, n: u32) -> u32 { return n; }\n\
    fn generic<T: view>(x: T, y: &T) -> T from x { return x; }\n";

#[test]
fn sig_origins_follow_the_table() {
    let (p, tc) = mem(SRC);
    assert!(codes(&p, &tc).is_empty(), "{:?}", codes(&p, &tc));
    let of = |name: &str| origins::of_sig(&p, &tc.types, tc.sigs.fn_sig(root_fn(&p, name)).unwrap());
    let arg = |i: u8| ResultOrigin { param: i, flow: Flow::Arg };
    let r = of("by_ref");
    assert!(r.ret_view && !r.error);
    assert_eq!(r.origins, [arg(1)], "deduplicated");
    assert_eq!(of("by_view").origins, [arg(0)]);
    let rv = of("by_ref_view");
    assert_eq!(rv.origins, [arg(0), ResultOrigin { param: 0, flow: Flow::Pointee }]);
    assert_eq!(rv.weakened().origins, [arg(0)], "from copy weakened to from");
    let o = of("owned");
    assert!(!o.ret_view && o.origins.is_empty() && !o.error);
    assert_eq!(of("generic").origins, [arg(0)]);

    // param classes (origin table rows)
    let sig = tc.sigs.fn_sig(root_fn(&p, "by_ref_view")).unwrap();
    let class = |t| origins::classify(&p, &tc.types, t, &sig.generics);
    assert_eq!(class(sig.params[0]), Some(ParamClass::RefView));
    assert_eq!(class(sig.params[1]), Some(ParamClass::Owned));
    let g = tc.sigs.fn_sig(root_fn(&p, "generic")).unwrap();
    assert_eq!(origins::classify(&p, &tc.types, g.params[0], &g.generics), Some(ParamClass::View));
    assert_eq!(origins::classify(&p, &tc.types, g.params[1], &g.generics), Some(ParamClass::RefView));
    let r = tc.sigs.fn_sig(root_fn(&p, "by_ref")).unwrap();
    assert_eq!(origins::classify(&p, &tc.types, r.params[0], &[]), Some(ParamClass::Ref));
}

#[test]
fn fn_value_calls_borrow_from_every_ref_or_view_arg() {
    let (p, tc) = mem(SRC);
    // fn(&u32, u32, Lexer, &Lexer) -> &u32: params 0, 2, 3 (never Pointee); non-view ret: none
    let sig = tc.sigs.fn_sig(root_fn(&p, "by_ref_view")).unwrap();
    let by_ref = tc.sigs.fn_sig(root_fn(&p, "by_ref")).unwrap();
    let by_view = tc.sigs.fn_sig(root_fn(&p, "by_view")).unwrap();
    let params = [by_ref.params[0], sig.params[1], by_view.params[0], sig.params[0]];
    let f = origins::of_fn_type(&p, &tc.types, &params, by_ref.ret, &[]);
    assert!(f.ret_view && !f.error);
    let ps: Vec<(u8, Flow)> = f.origins.iter().map(|o| (o.param, o.flow)).collect();
    assert_eq!(ps, [(0, Flow::Arg), (2, Flow::Arg), (3, Flow::Arg)]);
    let none = origins::of_fn_type(&p, &tc.types, &params, sig.params[1], &[]);
    assert!(!none.ret_view && none.origins.is_empty());
}

#[test]
fn invalid_from_clauses_flag_the_signature_a_missing_one_is_empty() {
    let (p, tc) = mem("fn a(n: u32, r: &u32) -> &u32 from n { return r; }\n\
         fn b(r: &u32) -> &u32 from copy r { return r; }\n\
         fn c(r: &u32) -> &str { return \"lit\"; }\n");
    assert_eq!(codes(&p, &tc), ["E0520", "E0521"]);
    for name in ["a", "b"] {
        let s = origins::of_sig(&p, &tc.types, tc.sigs.fn_sig(root_fn(&p, name)).unwrap());
        assert!(s.error && s.origins.is_empty(), "{name}");
    }
    // no `from`: the empty origin set, not an error
    let s = origins::of_sig(&p, &tc.types, tc.sigs.fn_sig(root_fn(&p, "c")).unwrap());
    assert!(s.ret_view && !s.error && s.origins.is_empty());
}

#[test]
fn kinds_of_types() {
    let (p, tc) = mem(SRC);
    let w = World { prog: &p, sigs: &tc.sigs };
    let mut tys = tc.types.clone();
    let sig = tc.sigs.fn_sig(root_fn(&p, "by_ref_view")).unwrap();
    let by_ref = tc.sigs.fn_sig(root_fn(&p, "by_ref")).unwrap();
    let k = |tys: &mut stave_sema::TypeTable, t| stave_sema::kinds::kind_of(w, tys, t, &[]);
    // &mut Lexer: a view, not Copy, not Flat
    let m = k(&mut tys, sig.params[0]);
    assert!(m.view && !m.copy && !m.flat && m.sized);
    // u32: Copy, Send, Sync, Flat, not a view
    let n = k(&mut tys, sig.params[1]);
    assert!(!n.view && n.copy && n.send && n.sync && n.flat);
    // &u32: a Copy view, Send and Sync, not Flat
    let r = k(&mut tys, by_ref.params[0]);
    assert!(r.view && r.copy && r.send && r.sync && !r.flat);
    // fn(&u32) -> &u32: a Copy view, neither Send nor Sync
    let ft = tys.intern(TypeKind::Fn {
        params: vec![by_ref.params[0]].into(),
        ret: by_ref.ret,
        features: 0,
        deny: 0,
    });
    let f = k(&mut tys, ft);
    assert!(f.view && f.copy && !f.send && !f.sync && !f.flat);
    // str: unsized
    let TypeKind::Ref { inner, .. } = *tys.kind(sig.ret) else { panic!("&str") };
    assert!(!k(&mut tys, inner).sized);
}
