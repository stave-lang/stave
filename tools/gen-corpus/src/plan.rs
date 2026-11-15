//! Phase 1: module layout (paths, imports, line budgets) and per-module item plans.

use crate::mix::Mix;
use crate::model::*;
use crate::names::{NOUNS, VERBS, push_pascal};
use crate::rng::Rng;

/// Smallest module budget the planner accepts (header + run fn + a few items).
pub const MIN_MODULE_LINES: u32 = 20;

/// Rank of `run_<k>`: above every planned fn, so it may call all of them.
pub const RUN_RANK: u8 = 4;

/// Module layout: everything decided before item planning.
pub struct Layout {
    pub paths: Vec<Vec<String>>,
    pub imports: Vec<Vec<u32>>,
    pub is_sink: Vec<bool>,
    pub sinks: Vec<u32>,
    pub budgets: Vec<u32>,
    /// Filler statements in `main` (only when there are no modules).
    pub main_pad: u32,
}

fn module_count(lines: u64, mix: &Mix) -> u32 {
    if lines < 60 {
        return 0;
    }
    let m = (lines * u64::from(mix.modules.per_10k_lines) + 5000) / 10_000;
    let cap = (lines - 10) / 40;
    m.clamp(1, cap.max(1)).min(u64::from(u32::MAX / 4)) as u32
}

fn dir_name(level: u32, idx: u32) -> String {
    let w = NOUNS[((idx as usize) * 7 + level as usize * 13) % NOUNS.len()];
    if level == 1 { format!("{w}_g{idx}") } else { format!("{w}_s{idx}") }
}

/// Decides paths, imports, sinks and per-module line budgets for a corpus of `lines` lines.
pub fn layout(lines: u64, seed: u64, mix: &Mix) -> Result<Layout, String> {
    let m = module_count(lines, mix);
    let mut paths = Vec::with_capacity(m as usize);
    let mut imports: Vec<Vec<u32>> = Vec::with_capacity(m as usize);
    let mut imported = vec![false; m as usize];
    let d1 = (m / 24).max(1);
    let max_depth = mix.modules.max_depth;
    for k in 0..m {
        let mut r = Rng::derive(seed, u64::from(k), 1);
        let depth = 1 + r.weighted(&[30, 40, 30][..max_depth as usize]) as u32;
        let mut path = Vec::with_capacity(depth as usize);
        let g = r.below(d1);
        if depth >= 2 {
            path.push(dir_name(1, g));
        }
        if depth >= 3 {
            path.push(dir_name(2, r.below(4)));
        }
        path.push(format!("{}_{k}", NOUNS[r.below(NOUNS.len() as u32) as usize]));
        paths.push(path);
        let n = r.below(mix.modules.max_imports + 1).min(k);
        let mut im: Vec<u32> = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let j = if r.chance(700) { k - 1 - r.below(k.min(16)) } else { r.below(k) };
            if !im.contains(&j) {
                im.push(j);
            }
        }
        im.sort_unstable();
        for &j in &im {
            imported[j as usize] = true;
        }
        imports.push(im);
    }
    let is_sink: Vec<bool> = imported.iter().map(|&i| !i).collect();
    let sinks: Vec<u32> = (0..m).filter(|&k| is_sink[k as usize]).collect();
    let main = u64::from(main_lines(sinks.len() as u32, m));
    let mut budgets = Vec::with_capacity(m as usize);
    let mut main_pad = 0;
    if m == 0 {
        main_pad = lines.saturating_sub(main) as u32;
    } else {
        let total = lines.saturating_sub(main);
        let mut r = Rng::derive(seed, 0, 2);
        let w: Vec<u64> = (0..m).map(|_| 60 + u64::from(r.below(81))).collect();
        let wsum: u64 = w.iter().sum();
        let mut given = 0u64;
        for &x in &w {
            let b = total * x / wsum;
            given += b;
            budgets.push(b);
        }
        let mut left = total - given;
        for b in budgets.iter_mut() {
            if left == 0 {
                break;
            }
            *b += 1;
            left -= 1;
        }
        for (k, &b) in budgets.iter().enumerate() {
            if b < u64::from(MIN_MODULE_LINES) || b > u64::from(u32::MAX / 2) {
                return Err(format!("internal: module {k} budget {b} out of range"));
            }
        }
    }
    Ok(Layout {
        paths,
        imports,
        is_sink,
        sinks,
        budgets: budgets.into_iter().map(|b| b as u32).collect(),
        main_pad,
    })
}

struct Planner<'a> {
    r: Rng,
    mix: &'a Mix,
    p: ModPlan,
    item_no: u32,
    plain: Vec<u32>,
    flat: Vec<u32>,
    generic: Vec<u32>,
    bound_traits: Vec<u32>,
    /// Chance that a fn or struct is generic, so that `generics.share_permille` of all items
    /// are generic (fns, structs, and the impls of generic structs can be).
    gen_permille: u32,
}

fn gen_permille(mix: &Mix) -> u32 {
    let (it, a) = (&mix.items, &mix.adt);
    let total = u64::from(it.r#fn + it.adt + it.r#impl + it.other);
    let sf = u64::from(a.r#struct) * 1000 / u64::from((a.r#struct + a.r#enum).max(1));
    let capable = u64::from(it.r#fn) * 1000 + u64::from(it.adt + it.r#impl) * sf;
    (u64::from(mix.generics.share_permille) * total * 1000 / capable.max(1)).min(1000) as u32
}

/// Plans module `k` (independent of every other module's plan).
pub fn plan_module(k: u32, lay: &Layout, seed: u64, mix: &Mix) -> ModPlan {
    let mut r = Rng::derive(seed, u64::from(k), 3);
    let p = ModPlan {
        path: lay.paths[k as usize].clone(),
        imports: lay.imports[k as usize].clone(),
        uses_io: r.chance(mix.modules.io_permille),
        budget: lay.budgets[k as usize],
        ..ModPlan::default()
    };
    let mut pl = Planner {
        r,
        mix,
        p,
        item_no: 0,
        plain: Vec::new(),
        flat: Vec::new(),
        generic: Vec::new(),
        bound_traits: Vec::new(),
        gen_permille: gen_permille(mix),
    };
    pl.run(k, lay);
    pl.p
}

impl Planner<'_> {
    fn body_len(&mut self) -> u32 {
        let b = &self.mix.fn_body;
        let x = self.r.lognormal(b.median, b.sigma);
        ((x + 0.5) as u32).clamp(1, b.cap)
    }

    fn sc(&mut self) -> Sc {
        Sc::ALL[self.r.weighted(&Sc::WEIGHTS)]
    }

    fn int_sc(&mut self) -> Sc {
        Sc::INTS[self.r.weighted(&Sc::WEIGHTS[..6])]
    }

    fn noun(&mut self) -> &'static str {
        NOUNS[self.r.below(NOUNS.len() as u32) as usize]
    }

    /// `n` distinct nouns.
    fn nouns(&mut self, n: u32) -> Vec<&'static str> {
        const STEPS: [usize; 5] = [1, 5, 7, 11, 13];
        let start = self.r.below(NOUNS.len() as u32) as usize;
        let step = STEPS[self.r.below(5) as usize];
        (0..n as usize).map(|i| NOUNS[(start + i * step) % NOUNS.len()]).collect()
    }

    fn pascal(&mut self) -> String {
        let mut s = String::new();
        push_pascal(&mut s, self.noun());
        s.push_str(&self.item_no.to_string());
        s
    }

    fn run(&mut self, k: u32, lay: &Layout) {
        let budget = self.p.budget;
        let mut rem = budget.saturating_sub(header_lines(&self.p));
        let mut rb = self.body_len().min((rem / 3).max(1));
        if rem < 3 + rb {
            rb = 1;
        }
        rem = rem.saturating_sub(3 + rb);
        let mut fails = 0;
        while rem >= 3 && fails < 8 {
            self.item_no += 1;
            let it = &self.mix.items;
            let used = match self.r.weighted(&[it.r#fn, it.adt, it.r#impl, it.other]) {
                0 => self.plan_fn(rem),
                1 => self.plan_adt(rem),
                2 => self.plan_impl(rem).or_else(|| self.plan_adt(rem)),
                _ => self.plan_other(rem, k, lay),
            };
            match used {
                Some(n) => rem -= n,
                None => fails += 1,
            }
        }
        self.p.filler = rem;
        let f = FnDef {
            name: format!("run_{k}"),
            public: true,
            kind: FnKind::Plain,
            params: Vec::new(),
            ret: Ret::Sc(Sc::U64),
            body: rb,
            doc: false,
            rank: RUN_RANK,
        };
        self.p.items.push(Item::Fn(self.p.fns.len() as u32));
        self.p.fns.push(f);
    }

    fn pick(&mut self, xs: &[u32]) -> u32 {
        xs[self.r.below(xs.len() as u32) as usize]
    }

    /// A scalar field type of struct `s` (structs always have one: field 0 is an int).
    fn scalar_field(&mut self, s: u32) -> Sc {
        let fs: Vec<Sc> = self.p.structs[s as usize]
            .fields
            .iter()
            .filter_map(|f| if let FTy::Sc(x) = f.ty { Some(x) } else { None })
            .collect();
        fs[self.r.below(fs.len() as u32) as usize]
    }

    fn scalars(&mut self, lo: u32, hi: u32) -> Vec<PTy> {
        let n = self.r.range(lo, hi);
        (0..n).map(|_| PTy::Sc(self.sc())).collect()
    }

    fn plan_fn(&mut self, rem: u32) -> Option<u32> {
        let doc = self.r.chance(self.mix.fn_body.doc_permille);
        let mut body = self.body_len();
        let over = 1 + u32::from(doc) + 2;
        if over + body > rem {
            if rem <= over {
                return None;
            }
            body = rem - over;
        }
        let (kind, params, ret) = self.signature();
        let rank = match kind {
            FnKind::Bound(_) => 1 + self.r.below(3) as u8,
            _ => self.r.below(4) as u8,
        };
        let name =
            format!("{}_{}{}", VERBS[self.r.below(VERBS.len() as u32) as usize], self.noun(), self.item_no);
        let public = !self.r.chance(self.mix.fn_body.private_permille);
        self.p.items.push(Item::Fn(self.p.fns.len() as u32));
        self.p.fns.push(FnDef { name, public, kind, params, ret, body, doc, rank });
        Some(over + body)
    }

    fn signature(&mut self) -> (FnKind, Vec<PTy>, Ret) {
        use PTy as P;
        if self.r.chance(self.gen_permille) {
            let g = &self.mix.generic_fns;
            let w = [
                g.pick,
                g.ref_pick,
                if self.bound_traits.is_empty() { 0 } else { g.bound },
                g.const_n,
                if self.generic.is_empty() { 0 } else { g.ctor },
            ];
            match self.r.weighted(&w) {
                0 if w[0] > 0 => return (FnKind::Pick, vec![P::Sc(Sc::Bool), P::T, P::T], Ret::T),
                1 if w[1] > 0 => {
                    return (FnKind::RefPick, vec![P::RefT, P::RefT, P::Sc(Sc::Bool)], Ret::RefT);
                }
                2 if w[2] > 0 => {
                    let bt = self.bound_traits.clone();
                    let tr = self.pick(&bt);
                    let m0 = &self.p.traits[tr as usize].methods[0];
                    let k = m0.arg.unwrap_or(m0.ret);
                    return (FnKind::Bound(tr), vec![P::RefT, P::Sc(k)], Ret::Sc(m0.ret));
                }
                3 if w[3] > 0 => return (FnKind::Const, vec![P::Sc(Sc::U32)], Ret::Sc(Sc::U32)),
                4 if w[4] > 0 => {
                    let gs = self.generic.clone();
                    let s = self.pick(&gs);
                    return (FnKind::GenCtor(s), vec![P::T], Ret::Cell(s));
                }
                _ => {}
            }
        }
        let f = &self.mix.fns;
        let has_s = !self.plain.is_empty();
        let has_e = !self.p.enums.is_empty();
        let w = [
            f.scalar,
            if has_s { f.struct_ref } else { 0 },
            if has_s { f.struct_mut } else { 0 },
            if has_s { f.ctor } else { 0 },
            if has_e { f.enum_make } else { 0 },
            if has_e { f.enum_eval } else { 0 },
            if has_s { f.struct_val } else { 0 },
            if has_s { f.ref_field } else { 0 },
            if has_s { f.ref_pick } else { 0 },
            if has_s { f.mut_field } else { 0 },
            f.option,
            if has_e { f.result } else { 0 },
            f.tuple,
        ];
        let plain = self.plain.clone();
        let ne = self.p.enums.len() as u32;
        let kind = FnKind::Plain;
        match self.r.weighted(&w) {
            1 if has_s => {
                let s = self.pick(&plain);
                let mut ps = vec![P::Ref(s)];
                ps.extend(self.scalars(0, 2));
                (kind, ps, Ret::Sc(self.sc()))
            }
            2 if has_s => {
                let s = self.pick(&plain);
                (kind, vec![P::Mut(s), P::Sc(self.int_sc())], Ret::Unit)
            }
            3 if has_s => {
                let s = self.pick(&plain);
                (kind, self.scalars(0, 3), Ret::S(s))
            }
            4 if has_e => {
                let e = self.r.below(ne);
                (kind, vec![P::Sc(self.int_sc())], Ret::E(e))
            }
            5 if has_e => {
                let e = self.r.below(ne);
                (kind, vec![P::EVal(e), P::Sc(self.int_sc())], Ret::Sc(self.int_sc()))
            }
            6 if has_s => {
                let s = self.pick(&plain);
                (kind, vec![P::Val(s), P::Sc(self.int_sc())], Ret::Sc(self.sc()))
            }
            7 if has_s => {
                let s = self.pick(&plain);
                let t = self.scalar_field(s);
                (kind, vec![P::Ref(s), P::Sc(Sc::Bool)], Ret::RefSc(t))
            }
            8 if has_s => {
                let s = self.pick(&plain);
                (kind, vec![P::Ref(s), P::Ref(s), P::Sc(Sc::Bool)], Ret::RefS(s))
            }
            9 if has_s => {
                let s = self.pick(&plain);
                let t = self.scalar_field(s);
                (kind, vec![P::Mut(s)], Ret::MutSc(t))
            }
            10 => (kind, self.scalars(1, 3), Ret::Opt(self.sc())),
            11 if has_e => {
                let e = self.r.below(ne);
                (kind, self.scalars(1, 3), Ret::Res(self.sc(), e))
            }
            12 => {
                let ps = self.scalars(1, 3);
                (kind, ps, Ret::Tup(self.sc(), self.sc()))
            }
            _ => {
                let ps = self.scalars(0, 4);
                let ret = if self.r.chance(200) { Ret::Unit } else { Ret::Sc(self.sc()) };
                (kind, ps, ret)
            }
        }
    }

    fn plan_adt(&mut self, rem: u32) -> Option<u32> {
        let a = &self.mix.adt;
        if self.r.weighted(&[a.r#struct, a.r#enum]) == 0 {
            self.plan_struct(rem)
        } else {
            self.plan_enum(rem)
        }
    }

    fn plan_struct(&mut self, rem: u32) -> Option<u32> {
        let a = &self.mix.adt;
        let (max_fields, nested) = (a.max_fields, a.nested_field_permille);
        let generic = self.r.chance(self.gen_permille);
        let name = self.pascal();
        let mut fields = Vec::new();
        if generic {
            let n = self.r.range(2, 4.min(max_fields.max(2)));
            let names = self.nouns(n);
            fields.push(Field { name: names[0], ty: FTy::T });
            fields.push(Field { name: names[1], ty: FTy::Sc(Sc::U32) });
            for &nm in &names[2..] {
                fields.push(Field { name: nm, ty: FTy::Sc(self.sc()) });
            }
        } else {
            let n = self.r.range(1, max_fields);
            let names = self.nouns(n);
            for (i, &nm) in names.iter().enumerate() {
                let ty = if i == 0 {
                    FTy::Sc(self.int_sc())
                } else if !self.flat.is_empty() && self.r.chance(nested) {
                    let fl = self.flat.clone();
                    FTy::S(self.pick(&fl))
                } else {
                    FTy::Sc(self.sc())
                };
                fields.push(Field { name: nm, ty });
            }
        }
        let flat = fields.iter().all(|f| matches!(f.ty, FTy::Sc(_)));
        let s = StructDef { name, generic, fields, imp: None, traits: Vec::new() };
        let cost = 1 + struct_lines(&s);
        if cost > rem {
            return None;
        }
        let idx = self.p.structs.len() as u32;
        if generic {
            self.generic.push(idx);
        } else {
            self.plain.push(idx);
            if flat {
                self.flat.push(idx);
            }
        }
        self.p.items.push(Item::Struct(idx));
        self.p.structs.push(s);
        Some(cost)
    }

    fn plan_enum(&mut self, rem: u32) -> Option<u32> {
        let nv = self.r.range(2, self.mix.adt.max_variants);
        if 1 + 2 + nv > rem {
            return None;
        }
        let name = self.pascal();
        let vnames = self.nouns(nv);
        let mut variants = Vec::with_capacity(nv as usize);
        for &vn in &vnames {
            let kind = match self.r.weighted(&[30, 40, 30]) {
                0 => VKind::Unit,
                1 => {
                    let n = self.r.range(1, 3);
                    VKind::Tup((0..n).map(|_| self.sc()).collect())
                }
                _ => {
                    let n = self.r.range(1, 3);
                    let fns = self.nouns(n);
                    VKind::Rec(fns.into_iter().map(|f| (f, self.sc())).collect())
                }
            };
            variants.push(VariantDef { name: vn, kind });
        }
        let e = EnumDef { name, variants };
        let cost = 1 + enum_lines(&e);
        self.p.items.push(Item::Enum(self.p.enums.len() as u32));
        self.p.enums.push(e);
        Some(cost)
    }

    fn plan_impl(&mut self, rem: u32) -> Option<u32> {
        let target =
            (0..self.p.structs.len() as u32).rev().find(|&i| self.p.structs[i as usize].imp.is_none())?;
        let s = self.p.structs[target as usize].clone();
        let m2 = self.mix.tiers.m2;
        let mut kinds: Vec<MKind> = Vec::new();
        if s.generic {
            if m2 {
                kinds.push(MKind::GGet);
            }
            kinds.push(MKind::GPut);
            kinds.push(MKind::GCount);
        } else {
            let n = self.r.range(2, 5);
            for _ in 0..n * 2 {
                if kinds.len() as u32 >= n {
                    break;
                }
                let fi = self.r.below(s.fields.len() as u32);
                let FTy::Sc(t) = s.fields[fi as usize].ty else { continue };
                let k = match self.r.weighted(&[10, 30, 20, 10, if m2 { 15 } else { 0 }, 15]) {
                    0 => MKind::Ctor,
                    1 => MKind::Get(fi),
                    2 if t.is_int() => MKind::Bump(fi),
                    3 => MKind::Consume(fi),
                    4 => MKind::RefGet(fi),
                    5 if t != Sc::Bool => MKind::Mix(fi),
                    _ => MKind::Get(fi),
                };
                if !kinds.contains(&k) {
                    kinds.push(k);
                }
            }
        }
        if kinds.is_empty() {
            return None;
        }
        let methods: Vec<Method> = kinds
            .into_iter()
            .map(|k| {
                let name = match k {
                    MKind::Ctor => "make".to_string(),
                    MKind::Get(f) => format!("get_{}", s.fields[f as usize].name),
                    MKind::Bump(f) => format!("bump_{}", s.fields[f as usize].name),
                    MKind::Consume(f) => format!("into_{}", s.fields[f as usize].name),
                    MKind::RefGet(f) => format!("ref_{}", s.fields[f as usize].name),
                    MKind::Mix(f) => format!("mix_{}", s.fields[f as usize].name),
                    MKind::GGet => format!("get_{}", s.fields[0].name),
                    MKind::GPut => format!("put_{}", s.fields[0].name),
                    MKind::GCount => format!("count_{}", s.fields[1].name),
                };
                let min = if k == MKind::GPut { 2 } else { 1 };
                let body = self.body_len().max(min);
                Method { name, kind: k, body, rank: self.r.below(3) as u8 }
            })
            .collect();
        let mut imp = ImplDef { target, methods };
        if 1 + impl_lines(&imp) > rem {
            for m in imp.methods.iter_mut() {
                m.body = if m.kind == MKind::GPut { 2 } else { 1 };
            }
            if 1 + impl_lines(&imp) > rem {
                return None;
            }
            // Grow bodies back while there is room.
            let mut room = rem - 1 - impl_lines(&imp);
            for m in imp.methods.iter_mut() {
                let add = room.min(4);
                m.body += add;
                room -= add;
            }
        }
        let cost = 1 + impl_lines(&imp);
        let idx = self.p.impls.len() as u32;
        self.p.structs[target as usize].imp = Some(idx);
        self.p.items.push(Item::Impl(idx));
        self.p.impls.push(imp);
        Some(cost)
    }

    fn plan_other(&mut self, rem: u32, k: u32, lay: &Layout) -> Option<u32> {
        let o = &self.mix.other;
        let w = [o.r#const, o.alias, if self.plain.is_empty() { 0 } else { o.r#trait }, o.import];
        match self.r.weighted(&w) {
            0 => self.plan_const(),
            1 => {
                let name = self.pascal();
                let ty = self.int_sc();
                self.p.items.push(Item::Alias(self.p.aliases.len() as u32));
                self.p.aliases.push(AliasDef { name, ty });
                Some(2)
            }
            2 if w[2] > 0 => self.plan_trait(rem),
            3 => {
                for _ in 0..4 {
                    if k == 0 {
                        break;
                    }
                    let j = self.r.below(k);
                    if !lay.is_sink[j as usize] && !self.p.imports.contains(&j) {
                        let at = self.p.imports.partition_point(|&x| x < j);
                        self.p.imports.insert(at, j);
                        return Some(1);
                    }
                }
                self.plan_const()
            }
            _ => self.plan_const(),
        }
    }

    fn plan_const(&mut self) -> Option<u32> {
        let mut up = String::from("K_");
        up.push_str(&self.noun().to_ascii_uppercase());
        up.push('_');
        up.push_str(&self.item_no.to_string());
        let ints: Vec<u32> = (0..self.p.aliases.len() as u32).collect();
        let (ty, alias) = if !ints.is_empty() && self.r.chance(300) {
            let a = self.pick(&ints);
            (self.p.aliases[a as usize].ty, Some(a))
        } else {
            (self.int_sc(), None)
        };
        self.p.items.push(Item::Const(self.p.consts.len() as u32));
        self.p.consts.push(ConstDef { name: up, ty, alias });
        Some(2)
    }

    fn plan_trait(&mut self, rem: u32) -> Option<u32> {
        let n = self.r.range(1, 3);
        let mut methods = Vec::with_capacity(n as usize);
        for i in 0..n {
            let verb = VERBS[self.r.below(VERBS.len() as u32) as usize];
            let arg = if self.r.chance(400) { Some(self.int_sc()) } else { None };
            methods.push(TraitMethod {
                name: format!("{verb}_t{}_{i}", self.item_no),
                arg,
                ret: self.int_sc(),
            });
        }
        let name = self.pascal();
        let t = TraitDef { name, methods };
        let tr = self.p.traits.len() as u32;
        let nimpl = self.r.range(1, 2).min(self.plain.len() as u32);
        let mut targets: Vec<u32> = Vec::new();
        let plain = self.plain.clone();
        for _ in 0..nimpl * 3 {
            if targets.len() as u32 >= nimpl {
                break;
            }
            let s = self.pick(&plain);
            if !targets.contains(&s) {
                targets.push(s);
            }
        }
        let mut impls: Vec<TraitImpl> = targets
            .iter()
            .map(|&s| TraitImpl { tr, target: s, bodies: (0..n).map(|_| self.body_len()).collect() })
            .collect();
        let cost = |t: &TraitDef, is: &[TraitImpl]| {
            1 + trait_lines(t) + is.iter().map(|i| 1 + timpl_lines(i)).sum::<u32>()
        };
        if cost(&t, &impls) > rem {
            for i in impls.iter_mut() {
                i.bodies.iter_mut().for_each(|b| *b = 1);
            }
            if cost(&t, &impls) > rem {
                return None;
            }
        }
        let c = cost(&t, &impls);
        self.p.items.push(Item::Trait(tr));
        self.p.traits.push(t);
        for i in impls {
            let ti = self.p.timpls.len() as u32;
            self.p.structs[i.target as usize].traits.push((tr, ti));
            self.p.items.push(Item::TraitImpl(ti));
            self.p.timpls.push(i);
        }
        self.bound_traits.push(tr);
        Some(c)
    }
}
