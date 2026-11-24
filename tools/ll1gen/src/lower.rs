//! Lowering EBNF to BNF.
//!
//! `?`, `*`, `+`, multi-alternative groups and separator lists become helper nonterminals named
//! `Rule#n` that record their origin rule. The separator-list rewrite turns
//! `X1..Xk (S X1..Xk)* S?` (S one terminal) into `L = X T ; T = S R | ε ; R = X T | ε`, which is
//! LL(1) where the plain lowering is not (S both continues and ends the list).
//!
//! Production order: a helper's productions are added while lowering the
//! alternative that uses it, before that alternative's own production.

use crate::Error;
use crate::ebnf::{Expr, Rule, same};
use std::collections::BTreeMap;

/// The end-of-input terminal. It is always a lexical class.
pub const EOF: &str = "EOF";

/// True if `name` is a terminal (a lexical class of the token stream, or `EOF`), not a rule.
pub fn is_class(classes: &[&str], name: &str) -> bool {
    name == EOF || classes.contains(&name)
}

/// A BNF symbol before terminal interning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sym {
    T(String),
    N(usize),
}

/// A nonterminal: a grammar rule (`helper == false`, index = rule index) or a helper.
#[derive(Debug, Clone)]
pub struct Nt {
    pub name: String,
    /// Index of the rule this nonterminal comes from (itself for a rule).
    pub origin: usize,
    pub helper: bool,
    /// For a helper: its number `n` in `Rule#n`; 0 for a rule.
    pub serial: u32,
    pub line: usize,
}

#[derive(Debug, Clone)]
pub struct Prod {
    pub lhs: usize,
    pub rhs: Vec<Sym>,
}

/// The lowered grammar. Nonterminals `0..rules` are the rules in file order; the start rule is 0.
#[derive(Debug, Clone)]
pub struct Grammar {
    pub nts: Vec<Nt>,
    pub prods: Vec<Prod>,
    pub rules: usize,
}

struct Lowering<'a> {
    classes: &'a [&'a str],
    g: Grammar,
    names: BTreeMap<&'a str, usize>,
    serials: Vec<u32>,
}

impl Lowering<'_> {
    fn fresh(&mut self, base: usize) -> usize {
        self.serials[base] += 1;
        let serial = self.serials[base];
        let name = format!("{}#{serial}", self.g.nts[base].name);
        let line = self.g.nts[base].line;
        self.g.nts.push(Nt { name, origin: base, helper: true, serial, line });
        self.g.nts.len() - 1
    }

    fn add(&mut self, lhs: usize, rhs: Vec<Sym>) {
        self.g.prods.push(Prod { lhs, rhs });
    }

    fn lower(&mut self, base: usize, e: &Expr) -> Result<Vec<Sym>, Error> {
        Ok(match e {
            Expr::Term(t, _) => vec![Sym::T(t.clone())],
            Expr::Name(n, line) => {
                if is_class(self.classes, n) {
                    vec![Sym::T(n.clone())]
                } else if let Some(&i) = self.names.get(n.as_str()) {
                    vec![Sym::N(i)]
                } else {
                    let rule = &self.g.nts[base].name;
                    return Err(Error::at(*line, format!("undefined nonterminal `{n}` in rule `{rule}`")));
                }
            }
            Expr::Seq(items) => {
                let mut r = Vec::new();
                for it in sepfix(items) {
                    r.extend(self.lower(base, &it)?);
                }
                r
            }
            Expr::Alt(alts) => {
                if let [one] = alts.as_slice() {
                    return self.lower(base, one);
                }
                let n = self.fresh(base);
                for s in alts {
                    let rhs = self.lower(base, s)?;
                    self.add(n, rhs);
                }
                vec![Sym::N(n)]
            }
            Expr::Group(a) => self.lower(base, a)?,
            Expr::Opt(x) => {
                let n = self.fresh(base);
                let rhs = self.lower(base, x)?;
                self.add(n, rhs);
                self.add(n, Vec::new());
                vec![Sym::N(n)]
            }
            Expr::Star(x) => {
                let n = self.fresh(base);
                let mut rhs = self.lower(base, x)?;
                rhs.push(Sym::N(n));
                self.add(n, rhs);
                self.add(n, Vec::new());
                vec![Sym::N(n)]
            }
            Expr::Plus(x) => {
                let mut r = self.lower(base, x)?;
                r.extend(self.lower(base, &Expr::Star(x.clone()))?);
                r
            }
            Expr::Sep { item, sep } => {
                let x = self.lower(base, item)?;
                let s = self.lower(base, sep)?;
                let t = self.fresh(base);
                let r = self.fresh(base);
                self.add(t, [s, vec![Sym::N(r)]].concat());
                self.add(t, Vec::new());
                self.add(r, [x.clone(), vec![Sym::N(t)]].concat());
                self.add(r, Vec::new());
                [x, vec![Sym::N(t)]].concat()
            }
        })
    }
}

/// The separator-list rewrite on one sequence: `X1..Xk (S X1..Xk)* S?` with S a terminal and
/// k >= 1 becomes `Sep { X1..Xk, S }`.
fn sepfix(items: &[Expr]) -> Vec<Expr> {
    let mut out: Vec<Expr> = Vec::new();
    let mut i = 0;
    while i < items.len() {
        let it = &items[i];
        if let Expr::Star(inner) = it
            && let Expr::Group(alt) = inner.as_ref()
            && let Expr::Alt(seqs) = alt.as_ref()
            && let [Expr::Seq(body)] = seqs.as_slice()
            && let [s @ Expr::Term(..), xs @ ..] = body.as_slice()
            && !xs.is_empty()
            && out.len() >= xs.len()
            && out[out.len() - xs.len()..].iter().zip(xs).all(|(a, b)| same(a, b))
            && let Some(Expr::Opt(o)) = items.get(i + 1)
            && same(o, s)
        {
            out.truncate(out.len() - xs.len());
            out.push(Expr::Sep { item: Box::new(Expr::Seq(xs.to_vec())), sep: Box::new(s.clone()) });
            i += 2;
            continue;
        }
        out.push(it.clone());
        i += 1;
    }
    out
}

/// Lowers the rules; rule `i` becomes nonterminal `i`. `classes` are the lexical class names.
pub fn lower(rules: &[Rule], classes: &[&str]) -> Result<Grammar, Error> {
    let mut names = BTreeMap::new();
    let mut nts = Vec::new();
    for (i, r) in rules.iter().enumerate() {
        if is_class(classes, &r.name) {
            return Err(Error::at(r.line, format!("rule `{}` redefines a lexical class", r.name)));
        }
        names.insert(r.name.as_str(), i);
        nts.push(Nt { name: r.name.clone(), origin: i, helper: false, serial: 0, line: r.line });
    }
    let mut l = Lowering {
        classes,
        g: Grammar { nts, prods: Vec::new(), rules: rules.len() },
        names,
        serials: vec![0; rules.len()],
    };
    for (i, r) in rules.iter().enumerate() {
        let Expr::Alt(alts) = &r.body else {
            return Err(Error::at(r.line, "rule body is not an alternative list"));
        };
        for s in alts {
            let rhs = l.lower(i, s)?;
            l.add(i, rhs);
        }
    }
    Ok(l.g)
}
