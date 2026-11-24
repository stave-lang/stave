//! Emits `{ll1_table,terminals,node_kinds}.stv` (file stems prefixed by the profile) into the
//! profile's Stave directory: the same data as the Rust tables, as top-level `pub const` items
//! with explicit types (Stave v0.4). Every top-level name carries the profile's symbol prefix: all
//! files of a directory share one namespace. Names are bytes plus offsets: a Stave const cannot
//! hold a reference (LANG.md §3), so there is no `&str` table.

use crate::emit_rust::{list, pack, packed_rhs, split_arrays};
use crate::naming::Names;
use crate::{Output, Profile, Tables};
use std::fmt::Write;

const W: usize = 100;

/// Writes the lines `(code, comment)`, a comment after its code. The comments of consecutive
/// commented lines start in one column, one space after the longest code of the run: the layout
/// `stave fmt` gives.
fn commented(o: &mut String, lines: &[(String, String)]) {
    let mut k = 0;
    while k < lines.len() {
        let run = lines[k..].iter().take_while(|l| !l.1.is_empty()).count();
        if run == 0 {
            let _ = writeln!(o, "{}", lines[k].0);
            k += 1;
            continue;
        }
        let width = lines[k..k + run].iter().map(|l| l.0.chars().count()).max().unwrap_or(0);
        for (code, comment) in &lines[k..k + run] {
            let pad = width - code.chars().count();
            let _ = writeln!(o, "{code}{:pad$} // {comment}", "");
        }
        k += run;
    }
}

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

fn terminals_stave(t: &Tables, names: &Names, header: &str, prof: &Profile) -> String {
    let n = t.nterm();
    let (p, sp) = (prof.file_prefix, prof.symbol_prefix);
    let mut o = format!("// {header}\n");
    let _ = writeln!(o, "// Terminals in sorted spelling order (byte order){}.", prof.order_note);
    let _ = writeln!(o, "// {sp}T_* are the terminal ids of {p}ll1_table.stv.\n");
    let _ = writeln!(o, "pub const {sp}NTERM: usize = {n};\n");
    let consts: Vec<(String, String)> = (names.term_const.iter().zip(&t.terms).enumerate())
        .map(|(i, (c, s))| (format!("pub const {sp}T_{c}: u16 = {i};"), s.clone()))
        .collect();
    commented(&mut o, &consts);
    o.push('\n');
    let texts: Vec<&str> = t.terms.iter().map(String::as_str).collect();
    strings(&mut o, &format!("{sp}TERM_TEXT"), "Spelling of terminal", &texts);
    let [ex1, ex2] = prof.class_examples;
    let _ =
        writeln!(o, "\n// True for a lexical class ({ex1}, {ex2}, ..., EOF), false for a fixed spelling.");
    array(&mut o, &format!("{sp}TERM_IS_CLASS"), "bool", &t.term_is_class);
    if t.any_word.is_some() {
        o += "\n// True if a token of the terminal has the Word shape [A-Za-z_][A-Za-z0-9_.]*: a keyword\n";
        o += "// spelled in that shape, a word class, or the pseudo-class AnyWord. The terminal AnyWord\n";
        o += "// matches every such token; the table holds its actions in every Word-shaped column of the\n";
        o += "// row where it has one.\n";
        array(&mut o, &format!("{sp}TERM_IS_WORD"), "bool", &t.term_is_word);
    }
    if t.split.is_empty() {
        return o;
    }
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
    array(&mut o, &format!("{sp}SPLIT_FIRST"), "i16", &first);
    let _ = writeln!(o, "// The terminal left after the first one ({}); -1 otherwise.", pairs(2));
    array(&mut o, &format!("{sp}SPLIT_REST"), "i16", &rest);
    o
}

fn node_kinds_stave(t: &Tables, names: &Names, header: &str, prof: &Profile) -> String {
    let (p, sp) = (prof.file_prefix, prof.symbol_prefix);
    let mut o = format!("// {header}\n");
    o += "// AST node kinds: one per named grammar rule. Helper nonterminals make no\n";
    let _ = writeln!(o, "// node. A kind's id equals its nonterminal id in {p}ll1_table.stv.\n");
    let _ = writeln!(o, "pub const {sp}NKIND: usize = {};\n", t.nkind);
    let consts: Vec<(String, String)> = (names.kind_const.iter().zip(&names.kind).enumerate())
        .map(|(i, (c, k))| (format!("pub const {sp}KIND_{c}: u8 = {i};"), k.clone()))
        .collect();
    commented(&mut o, &consts);
    o.push('\n');
    let ks: Vec<&str> = names.kind.iter().map(String::as_str).collect();
    strings(&mut o, &format!("{sp}NODE_KIND_NAME"), "Rule name of node kind", &ks);
    o
}

fn ll1_table_stave(t: &Tables, names: &Names, header: &str, prof: &Profile) -> String {
    let (rhs, off) = packed_rhs(t);
    let sp = prof.symbol_prefix;
    let start = &t.nts[0].name;
    let mut o = format!("// {header}\n");
    o += "// LL(1) parse table. Nonterminal ids 0..NKIND are the named rules (start rule\n";
    o += "// first, id = node kind); NKIND..NNT are helpers from `? * +` and groups, each with its\n";
    if t.overrides.is_empty() {
        o += "// origin rule. The grammar is LL(1) with no override.\n\n";
    } else {
        o += "// origin rule. Every conflict is one of the listed greedy overrides, resolved to the first\n";
        o += "// listed alternative.\n\n";
    }
    let _ = writeln!(o, "pub const {sp}NNT: usize = {};", t.nts.len());
    let _ = writeln!(o, "pub const {sp}NPROD: usize = {};", t.prods.len());
    let _ = writeln!(o, "pub const {sp}NRHS: usize = {};", rhs.len());
    o += "// Row length of TABLE (the number of terminals).\n";
    let _ = writeln!(o, "pub const {sp}TABLE_STRIDE: usize = {};", t.nterm());
    let _ = writeln!(o, "// The start nonterminal ({start}).\npub const {sp}START: u16 = 0;");
    let _ = writeln!(
        o,
        "// Marks a nonterminal in PROD_RHS; the low bits are its id.\npub const {sp}NT_BIT: u16 = 32768;\n"
    );
    let nn: Vec<&str> = t.nts.iter().map(|n| n.name.as_str()).collect();
    strings(&mut o, &format!("{sp}NT_NAME"), "Name of nonterminal (Rule#n for helpers)", &nn);
    o += "\n// Origin rule per nonterminal: the nonterminal id (= node kind) of the rule it comes from.\n";
    let origin: Vec<u16> = t.nts.iter().map(|n| n.origin).collect();
    array(&mut o, &format!("{sp}NT_ORIGIN"), "u16", &origin);
    o += "\n// True for a named rule (makes an AST node), false for a helper.\n";
    let named: Vec<bool> = t.nts.iter().map(|n| n.named).collect();
    array(&mut o, &format!("{sp}NT_NAMED"), "bool", &named);
    o += "\n// True for a pass-through rule: its node is omitted when it holds exactly one child node\n";
    o += "// and consumed no terminal itself (the child takes its place).\n";
    let pass: Vec<bool> = t.nts.iter().map(|n| n.pass).collect();
    array(&mut o, &format!("{sp}NT_PASS"), "bool", &pass);
    o += "\n// Left-hand side per production.\n";
    let lhs: Vec<u16> = t.prods.iter().map(|p| p.lhs).collect();
    array(&mut o, &format!("{sp}PROD_LHS"), "u16", &lhs);
    o += "\n// PROD_RHS[PROD_OFF[p]..PROD_OFF[p + 1]] is production p's right-hand side.\n";
    array(&mut o, &format!("{sp}PROD_OFF"), "u16", &off);
    o += "\n// Right-hand sides, each reversed (push order): a terminal id, or NT_BIT | nonterminal.\n";
    array(&mut o, &format!("{sp}PROD_RHS"), "u16", &rhs);
    o += "\n// TABLE[nt * TABLE_STRIDE + term]: the production to expand, or -1 (syntax error).\n";
    let _ = writeln!(o, "pub const {sp}TABLE: [i16; {}] = [", t.table.len());
    for row in t.table.chunks(t.nterm()) {
        o += &pack(nums(row), "    ", W);
    }
    o += "];\n";
    if t.overrides.is_empty() {
        return o;
    }
    o += "\n// Override cells: the grammar's listed LL(1) conflicts.\n";
    let _ = writeln!(o, "pub const {sp}NOVERRIDE: usize = {};", t.overrides.len());
    for (i, ov) in t.overrides.iter().enumerate() {
        let _ = writeln!(o, "// ({}, \"{}\"): {}.", ov.rule, ov.token, ov.why);
        let _ = writeln!(o, "pub const {sp}{}: usize = {i};", names.override_const[i]);
    }
    let n = t.overrides.len();
    let _ = writeln!(o, "pub const {sp}OVERRIDE_ORIGIN: [u16; {n}] = [{}];", list(t, |c| c.origin));
    let _ = writeln!(o, "pub const {sp}OVERRIDE_NT: [u16; {n}] = [{}];", list(t, |c| c.nt));
    let _ = writeln!(o, "pub const {sp}OVERRIDE_TERM: [u16; {n}] = [{}];", list(t, |c| c.term));
    let _ = writeln!(o, "pub const {sp}OVERRIDE_PROD: [u16; {n}] = [{}];", list(t, |c| c.prod));
    o
}

pub fn emit(t: &Tables, names: &Names, header: &str, prof: &Profile) -> Vec<Output> {
    let (dir, p) = (prof.stave_dir, prof.file_prefix);
    let out = |file: &str, content: String| Output { path: format!("{dir}/{file}"), content };
    vec![
        out(&format!("{p}ll1_table.stv"), ll1_table_stave(t, names, header, prof)),
        out(&format!("{p}terminals.stv"), terminals_stave(t, names, header, prof)),
        out(&format!("{p}node_kinds.stv"), node_kinds_stave(t, names, header, prof)),
    ]
}
