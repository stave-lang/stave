//! Nullable, FIRST, FOLLOW and the LL(1) table with every conflict kept (resolution is in
//! [`crate::build_tables`]).

use std::collections::BTreeSet;

/// An interned BNF symbol: terminal id or nonterminal index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum S {
    T(u16),
    N(usize),
}

/// Input: `prods[i] = (lhs, rhs)`, `nnt` nonterminals, `nterm` terminals, start nonterminal
/// `start`, end-of-input terminal `eof`.
pub struct Input<'a> {
    pub prods: &'a [(usize, Vec<S>)],
    pub nnt: usize,
    pub nterm: usize,
    pub start: usize,
    pub eof: u16,
}

pub struct Analysis {
    pub nullable: Vec<bool>,
    pub first: Vec<BTreeSet<u16>>,
    pub follow: Vec<BTreeSet<u16>>,
    /// `cells[nt][term]`: the productions predicted there, ascending, without duplicates.
    pub cells: Vec<Vec<Vec<usize>>>,
}

/// One LL(1) conflict: more than one production predicted at (nonterminal, terminal).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conflict {
    pub nt: usize,
    pub term: u16,
    pub prods: Vec<usize>,
}

fn first_of_seq(rhs: &[S], first: &[BTreeSet<u16>], nullable: &[bool]) -> (BTreeSet<u16>, bool) {
    let mut out = BTreeSet::new();
    for s in rhs {
        match *s {
            S::T(t) => {
                out.insert(t);
                return (out, false);
            }
            S::N(n) => {
                out.extend(first[n].iter().copied());
                if !nullable[n] {
                    return (out, false);
                }
            }
        }
    }
    (out, true)
}

pub fn analyze(inp: &Input) -> Analysis {
    let mut first = vec![BTreeSet::new(); inp.nnt];
    let mut nullable = vec![false; inp.nnt];
    let mut changed = true;
    while changed {
        changed = false;
        for (l, r) in inp.prods {
            let (f, nl) = first_of_seq(r, &first, &nullable);
            if !f.is_subset(&first[*l]) {
                first[*l].extend(f);
                changed = true;
            }
            if nl && !nullable[*l] {
                nullable[*l] = true;
                changed = true;
            }
        }
    }
    let mut follow = vec![BTreeSet::new(); inp.nnt];
    follow[inp.start].insert(inp.eof);
    changed = true;
    while changed {
        changed = false;
        for (l, r) in inp.prods {
            for (i, s) in r.iter().enumerate() {
                let S::N(v) = *s else { continue };
                let (mut add, nl) = first_of_seq(&r[i + 1..], &first, &nullable);
                if nl {
                    add.extend(follow[*l].iter().copied());
                }
                if !add.is_subset(&follow[v]) {
                    follow[v].extend(add);
                    changed = true;
                }
            }
        }
    }
    let mut cells = vec![vec![Vec::new(); inp.nterm]; inp.nnt];
    for (pi, (l, r)) in inp.prods.iter().enumerate() {
        let (mut f, nl) = first_of_seq(r, &first, &nullable);
        if nl {
            f.extend(follow[*l].iter().copied());
        }
        for t in f {
            let cell: &mut Vec<usize> = &mut cells[*l][t as usize];
            if cell.last() != Some(&pi) {
                cell.push(pi);
            }
        }
    }
    Analysis { nullable, first, follow, cells }
}

impl Analysis {
    /// Every cell with more than one production, by (nonterminal, terminal).
    pub fn conflicts(&self) -> Vec<Conflict> {
        let mut out = Vec::new();
        for (nt, row) in self.cells.iter().enumerate() {
            for (t, ps) in row.iter().enumerate() {
                if ps.len() > 1 {
                    out.push(Conflict { nt, term: t as u16, prods: ps.clone() });
                }
            }
        }
        out
    }
}
