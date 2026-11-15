//! Emits `stage1/src/generated/{ll1_table,terminals,node_kinds}.stv`: the same data as the Rust
//! tables, as top-level `pub const` items with explicit types (Stave v0.4). Names are bytes plus
//! offsets: a Stave const cannot hold a reference (LANG.md §3), so there is no `&str` table.

use crate::emit_rust::{list, pack, packed_rhs, split_arrays};
use crate::naming::Names;
use crate::{Output, STAVE_DIR, Tables};
use std::fmt::Write;

const W: usize = 100;

fn nums<T: ToString>(xs: &[T]) -> impl Iterator<Item = String> + '_ {
    xs.iter().map(|x| x.to_string())
}

/// `pub const NAME: [ty; len] = [ ... ];`
fn array<T: ToString>(o: &mut String, name: &str, ty: &str, xs: &[T]) {
    let _ = writeln!(o, "pub const {name}: [{ty}; {}] = [", xs.len());
    *o += &pack(nums(xs), "    ", W);
    *o += "];\n";
}

/// Strings as one byte array and `n + 1` offsets: string i is `BYTES[OFF[i]..OFF[i + 1]]`.
fn strings(o: &mut String, name: &str, what: &str, ss: &[&str]) {
    let mut bytes = Vec::new();
    let mut off = vec![0u32];
    for s in ss {
        bytes.extend_from_slice(s.as_bytes());
        off.push(bytes.len() as u32);
    }
    let _ = writeln!(o, "// {what} i (UTF-8) is {name}[{name}_OFF[i]..{name}_OFF[i + 1]].");
    array(o, &format!("{name}_OFF"), "u32", &off);
    array(o, name, "u8", &bytes);
}

fn terminals_stave(t: &Tables, names: &Names, header: &str) -> String {
    let n = t.nterm();
    let mut o = format!("// {header}\n");
    o += "// Terminals in sorted spelling order (byte order), which is the E0004 expected-set order.\n";
    o += "// T_* are the terminal ids of ll1_table.stv.\n\n";
    let _ = writeln!(o, "pub const NTERM: usize = {n};\n");
    for (i, (c, s)) in names.term_const.iter().zip(&t.terms).enumerate() {
        let _ = writeln!(o, "pub const T_{c}: u16 = {i}; // {s}");
    }
    o.push('\n');
    let texts: Vec<&str> = t.terms.iter().map(String::as_str).collect();
    strings(&mut o, "TERM_TEXT", "Spelling of terminal", &texts);
    o += "\n// True for a lexical class (Ident, IntLit, ..., EOF), false for a fixed spelling.\n";
    array(&mut o, "TERM_IS_CLASS", "bool", &t.term_is_class);
    let (first, rest) = split_arrays(t);
    let pairs = |part: usize| -> String {
        t.split
            .iter()
            .map(|&(a, f, r)| {
                let to = if part == 1 { f } else { r };
                format!("{} -> {}", t.terms[a as usize], t.terms[to as usize])
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    o += "\n// Split actions (grammar header): D3 `>>` `>=` `>>=` start with the `>` that closes type\n";
    o += "// arguments; the token `&&` in unary position is two `&`. A nonterminal with no action on a\n";
    o += "// split token uses the action of the token's first terminal; an expected first terminal\n";
    o += "// consumes that part of the token and leaves the rest as the lookahead.\n";
    let _ = writeln!(o, "// The first terminal of a split token ({}); -1 otherwise.", pairs(1));
    array(&mut o, "SPLIT_FIRST", "i16", &first);
    let _ = writeln!(o, "// The terminal left after the first one ({}); -1 otherwise.", pairs(2));
    array(&mut o, "SPLIT_REST", "i16", &rest);
    o
}

fn node_kinds_stave(t: &Tables, names: &Names, header: &str) -> String {
    let mut o = format!("// {header}\n");
    o += "// AST node kinds: one per named grammar rule. Helper nonterminals make no\n";
    o += "// node. A kind's id equals its nonterminal id in ll1_table.stv.\n\n";
    let _ = writeln!(o, "pub const NKIND: usize = {};\n", t.nkind);
    for (i, (c, k)) in names.kind_const.iter().zip(&names.kind).enumerate() {
        let _ = writeln!(o, "pub const KIND_{c}: u8 = {i}; // {k}");
    }
    o.push('\n');
    let ks: Vec<&str> = names.kind.iter().map(String::as_str).collect();
    strings(&mut o, "NODE_KIND_NAME", "Rule name of node kind", &ks);
    o
}

fn ll1_table_stave(t: &Tables, names: &Names, header: &str) -> String {
    let (rhs, off) = packed_rhs(t);
    let mut o = format!("// {header}\n");
    o += "// LL(1) parse table. Nonterminal ids 0..NKIND are the named rules (start rule\n";
    o += "// first, id = node kind); NKIND..NNT are helpers from `? * +` and groups, each with its\n";
    o += "// origin rule. Every conflict is one of the listed greedy overrides, resolved to the first\n";
    o += "// listed alternative.\n\n";
    let _ = writeln!(o, "pub const NNT: usize = {};", t.nts.len());
    let _ = writeln!(o, "pub const NPROD: usize = {};", t.prods.len());
    let _ = writeln!(o, "pub const NRHS: usize = {};", rhs.len());
    o += "// Row length of TABLE (the number of terminals).\n";
    let _ = writeln!(o, "pub const TABLE_STRIDE: usize = {};", t.nterm());
    o += "// The start nonterminal (File).\npub const START: u16 = 0;\n";
    o += "// Marks a nonterminal in PROD_RHS; the low bits are its id.\npub const NT_BIT: u16 = 32768;\n\n";
    let nn: Vec<&str> = t.nts.iter().map(|n| n.name.as_str()).collect();
    strings(&mut o, "NT_NAME", "Name of nonterminal (Rule#n for helpers)", &nn);
    o += "\n// Origin rule per nonterminal: the nonterminal id (= node kind) of the rule it comes from.\n";
    let origin: Vec<u16> = t.nts.iter().map(|n| n.origin).collect();
    array(&mut o, "NT_ORIGIN", "u16", &origin);
    o += "\n// True for a named rule (makes an AST node), false for a helper.\n";
    let named: Vec<bool> = t.nts.iter().map(|n| n.named).collect();
    array(&mut o, "NT_NAMED", "bool", &named);
    o += "\n// Left-hand side per production.\n";
    let lhs: Vec<u16> = t.prods.iter().map(|p| p.lhs).collect();
    array(&mut o, "PROD_LHS", "u16", &lhs);
    o += "\n// PROD_RHS[PROD_OFF[p]..PROD_OFF[p + 1]] is production p's right-hand side.\n";
    array(&mut o, "PROD_OFF", "u16", &off);
    o += "\n// Right-hand sides, each reversed (push order): a terminal id, or NT_BIT | nonterminal.\n";
    array(&mut o, "PROD_RHS", "u16", &rhs);
    o += "\n// TABLE[nt * TABLE_STRIDE + term]: the production to expand, or -1 (syntax error).\n";
    let _ = writeln!(o, "pub const TABLE: [i16; {}] = [", t.table.len());
    for row in t.table.chunks(t.nterm()) {
        o += &pack(nums(row), "    ", W);
    }
    o += "];\n\n";
    o += "// Override cells: the grammar's listed LL(1) conflicts.\n";
    let _ = writeln!(o, "pub const NOVERRIDE: usize = {};", t.overrides.len());
    for (i, ov) in t.overrides.iter().enumerate() {
        let _ = writeln!(o, "// ({}, \"{}\"): {}.", ov.rule, ov.token, ov.why);
        let _ = writeln!(o, "pub const {}: usize = {i};", names.override_const[i]);
    }
    let n = t.overrides.len();
    let _ = writeln!(o, "pub const OVERRIDE_ORIGIN: [u16; {n}] = [{}];", list(t, |c| c.origin));
    let _ = writeln!(o, "pub const OVERRIDE_NT: [u16; {n}] = [{}];", list(t, |c| c.nt));
    let _ = writeln!(o, "pub const OVERRIDE_TERM: [u16; {n}] = [{}];", list(t, |c| c.term));
    let _ = writeln!(o, "pub const OVERRIDE_PROD: [u16; {n}] = [{}];", list(t, |c| c.prod));
    o
}

pub fn emit(t: &Tables, names: &Names, header: &str) -> Vec<Output> {
    let out = |file: &str, content: String| Output { path: format!("{STAVE_DIR}/{file}"), content };
    vec![
        out("ll1_table.stv", ll1_table_stave(t, names, header)),
        out("terminals.stv", terminals_stave(t, names, header)),
        out("node_kinds.stv", node_kinds_stave(t, names, header)),
    ]
}
