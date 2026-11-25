//! `ll1gen`: LL(1) table generator. A [`Profile`] names one grammar and everything specific to it:
//! the grammar file, start rule, lexical classes, pass-through rules, listed overrides, output
//! directories and file / symbol prefixes. [`LANGUAGE`] (`contracts/grammar.ebnf`) drives the
//! stage0 parser (`stave-syntax`) and the stage1 GPU parser; [`IR_TEXT`]
//! (`contracts/ir-text.ebnf`) drives the readers of the IR text in both compilers.
//!
//! Pipeline: [`ebnf::parse`] → [`lower::lower`] → terminal interning → [`analyze::analyze`] →
//! `AnyWord` expansion → conflict resolution by the profile's listed greedy overrides →
//! [`Tables`] → [`emit_rust`] and [`emit_stave`].

pub mod analyze;
pub mod ebnf;
pub mod emit_rust;
pub mod emit_stave;
pub mod lower;

use analyze::S;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// A generator error, with the 1-based grammar line when there is one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    pub line: Option<usize>,
    pub msg: String,
}

impl Error {
    pub fn new(msg: impl Into<String>) -> Self {
        Error { line: None, msg: msg.into() }
    }

    pub fn at(line: usize, msg: impl Into<String>) -> Self {
        Error { line: Some(line), msg: msg.into() }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.line {
            Some(l) => write!(f, "line {l}: {}", self.msg),
            None => f.write_str(&self.msg),
        }
    }
}

impl std::error::Error for Error {}

/// A listed LL(1) conflict and its greedy resolution: the first listed alternative wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Override {
    /// Origin rule of the conflicting nonterminal.
    pub rule: &'static str,
    /// The lookahead terminal.
    pub token: &'static str,
    /// The rule of the grammar header that justifies it.
    pub why: &'static str,
}

/// The only conflicts `contracts/grammar.ebnf` may have (listed in the grammar header).
pub const OVERRIDES: [Override; 4] = [
    Override {
        rule: "Stmt",
        token: "unsafe",
        why: "statement-start `unsafe` always opens `Unsafe Block` (ExprStmt note)",
    },
    Override { rule: "TypeBase", token: "<", why: "D2: a `<` after a type name is always GenericArgs" },
    Override {
        rule: "TypeBase",
        token: "needs",
        why: "D5: `needs` directly after a fn type binds to that (innermost) fn type",
    },
    Override {
        rule: "TypeBase",
        token: "without",
        why: "D5: `without` directly after a fn type binds to that (innermost) fn type",
    },
];

/// Split actions (grammar header): (lexer token, the terminal it starts with, the remainder
/// token). D3: `>>`, `>=`, `>>=` start with the `>` that closes TypeArgs/GenericArgs. Unary `&&`:
/// two `&` tokens (`&&x` = `& &x`). A nonterminal with no action on a split token uses the action
/// of its first terminal; an expected first terminal consumes it and leaves the remainder.
pub const SPLIT: [(&str, &str, &str); 4] =
    [(">>", ">", ">"), (">=", ">", "="), (">>=", ">", ">="), ("&&", "&", "&")];

/// Rules and overrides that `contracts/parse-errors.md` names; generation fails without them.
pub const REQUIRED_RULES: [&str; 19] = [
    "Unsafe",
    "AndExpr",
    "CmpOp",
    "Stmt",
    "ExprStmt",
    "Param",
    "Params",
    "XParams",
    "FnDecl",
    "TraitFn",
    "KernelDecl",
    "DispatchDecl",
    "ExternItem",
    "ExternFn",
    "Primary",
    "AdtDecl",
    "Type",
    "TypeBase",
    "RetClause",
];
pub const REQUIRED_OVERRIDES: [(&str, &str); 1] = [("TypeBase", "<")];

/// Pass-through rules (grammar header): the expression precedence levels. A node of one of them
/// that holds exactly one child node and consumed no terminal itself is not emitted; its child
/// takes its place in the parent.
pub const PASS_THROUGH: [&str; 14] = [
    "Expr",
    "RangeExpr",
    "OrExpr",
    "AndExpr",
    "CmpExpr",
    "BitOr",
    "BitXor",
    "BitAnd",
    "Shift",
    "Add",
    "Mul",
    "Cast",
    "Unary",
    "Postfix",
];

/// The word-shaped pseudo-class of a grammar. Its terminal matches every token of the Word shape
/// `[A-Za-z_][A-Za-z0-9_.]*`: a keyword whose spelling has that shape, a token of one of
/// `word_classes`, and itself (the class of a Word that is neither a keyword nor another class).
#[derive(Debug, Clone, Copy)]
pub struct AnyWord<'a> {
    /// The class name as written in the grammar.
    pub name: &'a str,
    /// The other lexical classes whose tokens have the Word shape.
    pub word_classes: &'a [&'a str],
}

/// True if `s` has the Word shape `[A-Za-z_][A-Za-z0-9_.]*`.
pub fn is_word_shaped(s: &str) -> bool {
    let mut cs = s.chars();
    cs.next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && cs.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
}

/// What the generator enforces for one grammar.
#[derive(Debug, Clone, Copy)]
pub struct Config<'a> {
    /// The start rule: the first rule of the grammar.
    pub start: &'a str,
    /// Lexical class terminals (named in the grammar, not spelled). `EOF` is always one.
    pub classes: &'a [&'a str],
    /// The word-shaped pseudo-class, if the grammar uses one. It is a lexical class too.
    pub any_word: Option<AnyWord<'a>>,
    /// Split actions ([`SPLIT`] for the language grammar).
    pub split: &'a [(&'a str, &'a str, &'a str)],
    pub overrides: &'a [Override],
    pub required_rules: &'a [&'a str],
    pub required_overrides: &'a [(&'a str, &'a str)],
    /// Named rules whose node is omitted when it is a bare wrapper ([`PASS_THROUGH`]); each must
    /// exist and not derive ε.
    pub pass_through: &'a [&'a str],
}

/// Lexical classes of the language grammar (besides `EOF`).
pub const LANGUAGE_CLASSES: [&str; 6] = ["Ident", "IntLit", "FloatLit", "StrLit", "FmtLit", "CharLit"];

impl Config<'static> {
    /// The configuration for `contracts/grammar.ebnf`.
    pub const fn contract() -> Self {
        Config {
            start: "File",
            classes: &LANGUAGE_CLASSES,
            any_word: None,
            split: &SPLIT,
            overrides: &OVERRIDES,
            required_rules: &REQUIRED_RULES,
            required_overrides: &REQUIRED_OVERRIDES,
            pass_through: &PASS_THROUGH,
        }
    }

    /// The configuration for `contracts/ir-text.ebnf`: no override, no split token, no
    /// pass-through rule, the `AnyWord` pseudo-class.
    pub const fn ir_text() -> Self {
        Config {
            start: "Line",
            classes: &IR_TEXT_CLASSES,
            any_word: Some(AnyWord { name: "AnyWord", word_classes: &IR_TEXT_WORD_CLASSES }),
            split: &[],
            overrides: &[],
            required_rules: &[],
            required_overrides: &[],
            pass_through: &[],
        }
    }
}

/// Lexical classes of the IR text grammar (besides `EOF` and `AnyWord`).
pub const IR_TEXT_CLASSES: [&str; 11] =
    ["Val", "Blk", "TId", "SId", "DId", "FnRef", "Intr", "Int", "Hex", "Bytes", "String"];

/// The IR text classes whose tokens have the Word shape.
pub const IR_TEXT_WORD_CLASSES: [&str; 6] = ["Intr", "Blk", "TId", "SId", "DId", "Bytes"];

/// The integer type of the terminal ids in a profile's generated Stave constants, and with it
/// the most terminals the profile may have.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalIds {
    /// Ids are stage1 token kinds, stored in `u8` columns. The lexer needs two values above the
    /// ids: its error kind (`NTERM`) and 255 as no kind.
    U8,
    U16,
}

impl TerminalIds {
    pub const fn stave_type(self) -> &'static str {
        match self {
            TerminalIds::U8 => "u8",
            TerminalIds::U16 => "u16",
        }
    }

    pub const fn max_terminals(self) -> usize {
        match self {
            TerminalIds::U8 => 254,
            TerminalIds::U16 => u16::MAX as usize,
        }
    }

    fn why_limited(self) -> &'static str {
        match self {
            TerminalIds::U8 => {
                "token kinds are u8, and the lexer needs two values above the ids: its error kind and 255 as no kind"
            }
            TerminalIds::U16 => "table entries are 16-bit",
        }
    }
}

/// One generated table set: a grammar and everything specific to it.
#[derive(Debug, Clone, Copy)]
pub struct Profile {
    /// The name given to `--profile`.
    pub name: &'static str,
    /// The grammar file, relative to the repository root.
    pub grammar: &'static str,
    pub config: Config<'static>,
    /// Output directories, relative to the repository root.
    pub rust_dir: &'static str,
    pub stave_dir: &'static str,
    /// Prefix of every generated file stem (`ir_text_` gives `ir_text_ll1_table.rs`).
    pub file_prefix: &'static str,
    /// Prefix of every top-level name in the generated Stave files (they share one package
    /// namespace with the other grammars' tables).
    pub symbol_prefix: &'static str,
    /// Appended to "Terminals in sorted spelling order (byte order)" in the terminal files.
    pub order_note: &'static str,
    /// Two lexical class names, quoted in the terminal docs.
    pub class_examples: [&'static str; 2],
    /// The type of the terminal ids in the Stave constants.
    pub terminal_ids: TerminalIds,
}

/// Paths of the language grammar's generated files, relative to the repository root.
pub const RUST_DIR: &str = "crates/stave-syntax/src/generated";
pub const STAVE_DIR: &str = "stage1/src/generated";

/// `contracts/grammar.ebnf`.
pub const LANGUAGE: Profile = Profile {
    name: "language",
    grammar: "contracts/grammar.ebnf",
    config: Config::contract(),
    rust_dir: RUST_DIR,
    stave_dir: STAVE_DIR,
    file_prefix: "",
    symbol_prefix: "",
    order_note: ", which is the E0004 expected-set order",
    class_examples: ["Ident", "IntLit"],
    terminal_ids: TerminalIds::U8,
};

/// `contracts/ir-text.ebnf`: the grammar of one line of the IR text.
///
/// # Token classification the tables assume
///
/// A reader lexes one line (no `\n` inside) into terminal ids of `terminals`, then runs the PDA
/// from the start rule `Line`; the end of the line is the terminal `EOF`, appended once. Blank
/// and comment-only lines never reach the table. Spaces and tabs only separate tokens.
///
/// 1. Punctuation, maximal munch: `->` and `<=` are single tokens, then one of `( ) < > , : = !
///    | { }`. A `-` not followed by a digit and not starting `->` is a lexical error.
/// 2. `%` Num is `Val`; `@` Num (`.` Num)* (`.u`)? is `FnRef`; a double-quoted string with the
///    `strings` dump escapes is `String`; `0x` plus exactly 4, 8 or 16 lowercase hex digits is
///    `Hex`; `-`? Num is `Int` (no leading zero except `0` itself, no `-0`).
/// 3. A Word is `[A-Za-z_][A-Za-z0-9_.]*`, taken by maximal munch, then classified by the first
///    rule that matches, shape before keyword: `intrinsic.` `[a-z0-9_]+` is `Intr`; `b` Num is
///    `Blk`; `t` Num is `TId`; `s` Num is `SId`; `d` Num is `DId`; `x` followed by 1 to 64
///    pairs of lowercase hex digits is `Bytes`; a spelling that is a keyword terminal of the
///    table is that keyword (a class name such as `Val` is not a keyword); anything else is a `Name`, which has no terminal of its own and is
///    lexed as the `AnyWord` terminal. No keyword has a shape of the first six rules, so the
///    order is not observable for keywords (`b16`, the bfloat16 float key, is a `Blk`).
/// 4. `AnyWord` is the pseudo-class of the Word shape: `TERM_IS_WORD[t]` is true for every
///    keyword spelled in the Word shape, for `Intr`, `Blk`, `TId`, `SId`, `DId`, `Bytes` and for
///    `AnyWord` itself. The table already holds the fallback: in a row where `AnyWord` has an
///    action, every Word-shaped terminal whose own column was empty holds that action. Where
///    the PDA expects the terminal `AnyWord` it accepts any token with `TERM_IS_WORD` set;
///    every other expected terminal matches by equal id. `AnyWord` is expected in the `Path`
///    rule only.
/// 5. After the stack is empty the only token left must be the final `EOF`.
pub const IR_TEXT: Profile = Profile {
    name: "ir-text",
    grammar: "contracts/ir-text.ebnf",
    config: Config::ir_text(),
    rust_dir: "crates/stave-ir/src/generated",
    stave_dir: "stage1/src/generated",
    file_prefix: "ir_text_",
    symbol_prefix: "IR_",
    order_note: "",
    class_examples: ["Val", "TId"],
    terminal_ids: TerminalIds::U16,
};

/// Every profile `gen` runs, language first.
pub const PROFILES: [&Profile; 2] = [&LANGUAGE, &IR_TEXT];

/// An output nonterminal.
#[derive(Debug, Clone)]
pub struct NtOut {
    /// Rule name, or `Rule#n` for a helper.
    pub name: String,
    /// Id of the origin rule (= its nonterminal id = its node kind).
    pub origin: u16,
    pub named: bool,
    /// A pass-through rule ([`Config::pass_through`]); false for helpers.
    pub pass: bool,
}

/// An output production: `rhs` in grammar order (emitters reverse it).
#[derive(Debug, Clone)]
pub struct ProdOut {
    pub lhs: u16,
    pub rhs: Vec<Sym>,
}

/// An output symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sym {
    T(u16),
    N(u16),
}

/// One override as resolved in the table.
#[derive(Debug, Clone)]
pub struct OverrideCell {
    pub rule: String,
    pub token: String,
    pub why: String,
    pub origin: u16,
    pub nt: u16,
    pub term: u16,
    /// The chosen production (the first listed alternative).
    pub prod: u16,
    /// All productions that conflicted there, ascending.
    pub alternatives: Vec<u16>,
}

/// The generated tables.
///
/// Nonterminal ids: the start rule is 0, then the other rules sorted by name: ids `0..nkind` are
/// the named rules and equal their node kinds. Helpers follow, sorted by (origin name, n).
/// Terminal ids: sorted terminal spellings (byte order), the E0004 expected-set order.
/// Productions are numbered in lowering order.
#[derive(Debug, Clone)]
pub struct Tables {
    pub terms: Vec<String>,
    /// A lexical class (`Ident`, `IntLit`, ..., `EOF`) rather than a fixed spelling.
    pub term_is_class: Vec<bool>,
    pub nts: Vec<NtOut>,
    pub nkind: usize,
    pub prods: Vec<ProdOut>,
    /// `table[nt * nterm + term]`: production, or -1.
    pub table: Vec<i16>,
    pub overrides: Vec<OverrideCell>,
    /// Split actions ([`Config::split`]): (token, first, rest) terminal ids.
    pub split: Vec<(u16, u16, u16)>,
    /// The terminal id of the word-shaped pseudo-class ([`Config::any_word`]).
    pub any_word: Option<u16>,
    /// Per terminal: a token of it has the Word shape (a keyword spelled in the Word shape, a
    /// word class, the pseudo-class itself). Empty without a pseudo-class.
    pub term_is_word: Vec<bool>,
}

impl Tables {
    pub fn nterm(&self) -> usize {
        self.terms.len()
    }

    pub fn term_id(&self, text: &str) -> Option<u16> {
        self.terms.iter().position(|t| t == text).map(|i| i as u16)
    }

    pub fn nt_id(&self, name: &str) -> Option<u16> {
        self.nts.iter().position(|n| n.name == name).map(|i| i as u16)
    }

    pub fn action(&self, nt: u16, term: u16) -> i16 {
        self.table[nt as usize * self.nterm() + term as usize]
    }

    /// `A -> x 'y' Z` (ε for an empty right-hand side).
    pub fn show_prod(&self, p: usize) -> String {
        let pr = &self.prods[p];
        let rhs: Vec<String> = pr
            .rhs
            .iter()
            .map(|s| match *s {
                Sym::T(t) => format!("\"{}\"", self.terms[t as usize]),
                Sym::N(n) => self.nts[n as usize].name.clone(),
            })
            .collect();
        let rhs = if rhs.is_empty() { "ε".to_string() } else { rhs.join(" ") };
        format!("{} -> {rhs}", self.nts[pr.lhs as usize].name)
    }
}

/// Builds the tables from grammar source text. On failure returns every error found: unlisted
/// conflicts (naming origin rule, token and alternatives), stale overrides, missing rules.
pub fn build_tables(src: &str, cfg: &Config) -> Result<Tables, Vec<Error>> {
    let rules = ebnf::parse(src).map_err(|e| vec![e])?;
    let mut classes: Vec<&str> = cfg.classes.to_vec();
    classes.extend(cfg.any_word.map(|a| a.name));
    let g = lower::lower(&rules, &classes).map_err(|e| vec![e])?;
    if rules.first().map(|r| r.name.as_str()) != Some(cfg.start) {
        let found = rules.first().map_or("(none)", |r| r.name.as_str());
        let line = rules.first().map(|r| r.line);
        let msg = format!("the start rule `{}` must be the first rule, found `{found}`", cfg.start);
        return Err(vec![Error { line, msg }]);
    }

    // Terminals: every spelling used, plus EOF and the split tokens, in byte order.
    let mut tset: BTreeSet<String> = BTreeSet::new();
    for p in &g.prods {
        for s in &p.rhs {
            if let lower::Sym::T(t) = s {
                tset.insert(t.clone());
            }
        }
    }
    tset.insert(lower::EOF.to_string());
    for &(a, b, c) in cfg.split {
        tset.extend([a, b, c].map(String::from));
    }
    let terms: Vec<String> = tset.into_iter().collect();
    if terms.len() >= 0x8000 {
        return Err(vec![Error::new("too many terminals")]);
    }
    let tid: BTreeMap<&str, u16> = terms.iter().enumerate().map(|(i, t)| (t.as_str(), i as u16)).collect();
    let prods: Vec<(usize, Vec<S>)> = g
        .prods
        .iter()
        .map(|p| {
            let rhs = p
                .rhs
                .iter()
                .map(|s| match s {
                    lower::Sym::T(t) => S::T(tid[t.as_str()]),
                    lower::Sym::N(n) => S::N(*n),
                })
                .collect();
            (p.lhs, rhs)
        })
        .collect();
    let eof = tid[lower::EOF];
    let mut an = analyze::analyze(&analyze::Input {
        prods: &prods,
        nnt: g.nts.len(),
        nterm: terms.len(),
        start: 0,
        eof,
    });

    // Output nonterminal order: start, named rules by name, helpers by (origin name, serial).
    let mut order: Vec<usize> = (0..g.nts.len()).collect();
    order.sort_by(|&a, &b| {
        let (x, y) = (&g.nts[a], &g.nts[b]);
        let key = |n: &lower::Nt, i: usize| (n.helper, i != 0, g.nts[n.origin].name.clone(), n.serial);
        key(x, a).cmp(&key(y, b))
    });
    let mut newid = vec![0u16; g.nts.len()];
    for (new, &old) in order.iter().enumerate() {
        newid[old] = new as u16;
    }
    let nts: Vec<NtOut> = order
        .iter()
        .map(|&old| {
            let n = &g.nts[old];
            let pass = !n.helper && cfg.pass_through.contains(&n.name.as_str());
            NtOut { name: n.name.clone(), origin: newid[n.origin], named: !n.helper, pass }
        })
        .collect();
    let nkind = g.rules;
    let prods_out: Vec<ProdOut> = prods
        .iter()
        .map(|(l, r)| ProdOut {
            lhs: newid[*l],
            rhs: r
                .iter()
                .map(|s| match *s {
                    S::T(t) => Sym::T(t),
                    S::N(n) => Sym::N(newid[n]),
                })
                .collect(),
        })
        .collect();

    let split = cfg.split.iter().map(|(a, b, c)| (tid[a], tid[b], tid[c])).collect();
    let term_is_class: Vec<bool> = terms.iter().map(|s| lower::is_class(&classes, s)).collect();
    let any_word = cfg.any_word.and_then(|a| tid.get(a.name).copied());
    let term_is_word: Vec<bool> = match cfg.any_word {
        None => Vec::new(),
        Some(a) => terms
            .iter()
            .zip(&term_is_class)
            .map(
                |(s, &c)| {
                    if c { s == a.name || a.word_classes.contains(&s.as_str()) } else { is_word_shaped(s) }
                },
            )
            .collect(),
    };
    let mut t = Tables {
        terms,
        term_is_class,
        nts,
        nkind,
        prods: prods_out,
        table: Vec::new(),
        overrides: Vec::new(),
        split,
        any_word,
        term_is_word,
    };

    let mut errors = Vec::new();
    let nrhs: usize = t.prods.iter().map(|p| p.rhs.len()).sum();
    if t.nts.len() >= 0x8000 || t.prods.len() >= 0x7fff || nrhs > 0xffff {
        errors.push(Error::new("too many nonterminals or productions for 16-bit tables"));
    }
    if nkind > 256 {
        errors.push(Error::new(format!("{nkind} named rules: node kinds must fit in u8")));
    }

    // The pseudo-class stands for every Word-shaped terminal: in a row where it has an action, a
    // Word-shaped terminal takes that action too, and a different action there is a conflict.
    if let Some(any) = cfg.any_word {
        match t.any_word {
            None => {
                errors.push(Error::new(format!("the pseudo-class `{}` is not used by the grammar", any.name)))
            }
            Some(a) => {
                for name in any.word_classes {
                    if !classes.contains(name) {
                        errors
                            .push(Error::new(format!("word class `{name}` is not a declared lexical class")));
                    }
                }
                for row in &mut an.cells {
                    let from_any = row[a as usize].clone();
                    if from_any.is_empty() {
                        continue;
                    }
                    for (w, cell) in row.iter_mut().enumerate() {
                        if w != a as usize && t.term_is_word[w] {
                            cell.extend(from_any.iter().copied());
                            cell.sort_unstable();
                            cell.dedup();
                        }
                    }
                }
            }
        }
    }

    // Conflict resolution.
    let nterm = t.nterm();
    let mut table = vec![-1i16; t.nts.len() * nterm];
    let mut conflicts = an.conflicts();
    conflicts.sort_by(|a, b| {
        let key = |c: &analyze::Conflict| (g.nts[g.nts[c.nt].origin].name.clone(), c.term, newid[c.nt]);
        key(a).cmp(&key(b))
    });
    let mut used = vec![Vec::new(); cfg.overrides.len()];
    for c in &conflicts {
        let n = &g.nts[c.nt];
        let origin = &g.nts[n.origin];
        let token = &t.terms[c.term as usize];
        match cfg.overrides.iter().position(|o| o.rule == origin.name && o.token == token) {
            Some(k) => used[k].push(c.clone()),
            None => {
                let alts: Vec<String> = c.prods.iter().map(|&p| t.show_prod(p)).collect();
                errors.push(Error::at(
                    origin.line,
                    format!(
                        "LL(1) conflict not in OVERRIDES: ({}, \"{token}\") in {}: alternatives {}",
                        origin.name,
                        n.name,
                        alts.join(" | ")
                    ),
                ));
            }
        }
    }
    for (old, row) in an.cells.iter().enumerate() {
        for (term, ps) in row.iter().enumerate() {
            if let Some(&p) = ps.first() {
                table[newid[old] as usize * nterm + term] = p as i16;
            }
        }
    }
    for (k, o) in cfg.overrides.iter().enumerate() {
        match used[k].as_slice() {
            [] => errors.push(Error::new(format!(
                "stale override ({}, \"{}\"): the grammar has no such conflict; remove it from OVERRIDES",
                o.rule, o.token
            ))),
            [c] => t.overrides.push(OverrideCell {
                rule: o.rule.to_string(),
                token: o.token.to_string(),
                why: o.why.to_string(),
                origin: newid[g.nts[c.nt].origin],
                nt: newid[c.nt],
                term: c.term,
                prod: c.prods[0] as u16,
                alternatives: c.prods.iter().map(|&p| p as u16).collect(),
            }),
            many => errors.push(Error::new(format!(
                "override ({}, \"{}\") resolves {} table cells; expected exactly one",
                o.rule,
                o.token,
                many.len()
            ))),
        }
    }
    for r in cfg.required_rules {
        if !rules.iter().any(|x| x.name == *r) {
            errors.push(Error::new(format!("rule `{r}` named by contracts/parse-errors.md is missing")));
        }
    }
    for (r, tok) in cfg.required_overrides {
        if !cfg.overrides.iter().any(|o| o.rule == *r && o.token == *tok) {
            errors.push(Error::new(format!(
                "override ({r}, \"{tok}\") named by contracts/parse-errors.md is not listed"
            )));
        }
    }
    for r in cfg.pass_through {
        match g.nts.iter().position(|n| !n.helper && n.name == *r) {
            None => errors.push(Error::new(format!("pass-through rule `{r}` is missing"))),
            Some(i) if an.nullable[i] => errors.push(Error::at(
                g.nts[i].line,
                format!("pass-through rule `{r}` derives ε: its node could have no child"),
            )),
            Some(_) => {}
        }
    }
    if !errors.is_empty() {
        return Err(errors);
    }
    t.table = table;
    Ok(t)
}

/// One generated file: path relative to the repository root, and its content.
pub struct Output {
    pub path: String,
    pub content: String,
}

/// Renders every generated file. `grammar` is the grammar path shown in the headers.
pub fn render(t: &Tables, grammar: &str, prof: &Profile) -> Result<Vec<Output>, Error> {
    let max = prof.terminal_ids.max_terminals();
    if t.nterm() > max {
        return Err(Error::new(format!(
            "{} terminals, at most {max} for the {} profile: its terminal ids are {} ({})",
            t.nterm(),
            prof.name,
            prof.terminal_ids.stave_type(),
            prof.terminal_ids.why_limited(),
        )));
    }
    let names = naming::Names::new(t)?;
    let header = format!("generated by tools/ll1gen from {grammar}; do not edit");
    let mut out = emit_rust::emit(t, &names, &header, prof);
    out.extend(emit_stave::emit(t, &names, &header, prof));
    Ok(out)
}

/// Builds and renders in one step.
pub fn generate(src: &str, grammar: &str, prof: &Profile) -> Result<(Tables, Vec<Output>), Vec<Error>> {
    let t = build_tables(src, &prof.config)?;
    let out = render(&t, grammar, prof).map_err(|e| vec![e])?;
    Ok((t, out))
}

/// Identifier names for the generated code.
pub mod naming {
    use super::{Error, Tables};
    use std::collections::BTreeSet;

    const RUST_KEYWORDS: [&str; 51] = [
        "as", "async", "await", "break", "const", "continue", "crate", "dyn", "else", "enum", "extern",
        "false", "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut", "pub", "ref",
        "return", "self", "Self", "static", "struct", "super", "trait", "true", "type", "unsafe", "use",
        "where", "while", "abstract", "become", "box", "do", "final", "macro", "override", "priv", "typeof",
        "unsized", "virtual", "yield", "try",
    ];

    fn char_name(c: char) -> Option<&'static str> {
        Some(match c {
            '+' => "Plus",
            '-' => "Minus",
            '*' => "Star",
            '/' => "Slash",
            '%' => "Percent",
            '=' => "Eq",
            '<' => "Lt",
            '>' => "Gt",
            '!' => "Bang",
            '&' => "Amp",
            '|' => "Pipe",
            '^' => "Caret",
            '~' => "Tilde",
            '?' => "Question",
            '.' => "Dot",
            ',' => "Comma",
            ';' => "Semi",
            ':' => "Colon",
            '(' => "LParen",
            ')' => "RParen",
            '[' => "LBracket",
            ']' => "RBracket",
            '{' => "LBrace",
            '}' => "RBrace",
            '@' => "At",
            '#' => "Hash",
            '$' => "Dollar",
            '\\' => "Backslash",
            '\'' => "Quote",
            '"' => "DQuote",
            '`' => "Backtick",
            _ => return None,
        })
    }

    fn camel_word(w: &str) -> String {
        w.split('_')
            .filter(|p| !p.is_empty())
            .map(|p| {
                let mut cs = p.chars();
                match cs.next() {
                    Some(f) => f.to_ascii_uppercase().to_string() + cs.as_str(),
                    None => String::new(),
                }
            })
            .collect()
    }

    /// Rust variant name of a terminal: `fn` → `KwFn`, `Self` → `KwSelfType`, `+%=` →
    /// `PlusPercentEq`, `as%` → `AsPercent`, `_` → `Underscore`, `EOF` → `Eof`.
    pub fn term_variant(text: &str, is_class: bool) -> Option<String> {
        if is_class {
            return Some(if text == "EOF" { "Eof".to_string() } else { camel_word(text) });
        }
        if text == "_" {
            return Some("Underscore".to_string());
        }
        let ident = |c: char| c.is_ascii_alphanumeric() || c == '_';
        if text.chars().all(ident) && text.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_') {
            let cap = text.starts_with(|c: char| c.is_ascii_uppercase());
            return Some(format!("Kw{}{}", camel_word(text), if cap { "Type" } else { "" }));
        }
        let mut out = String::new();
        let mut word = String::new();
        for c in text.chars() {
            if ident(c) {
                word.push(c);
                continue;
            }
            if !word.is_empty() {
                out += &camel_word(&word);
                word.clear();
            }
            out += char_name(c)?;
        }
        out += &camel_word(&word);
        Some(out)
    }

    /// `CamelCase` → `CAMEL_CASE` (`XParams` → `X_PARAMS`, `KwFn` → `KW_FN`).
    pub fn screaming(camel: &str) -> String {
        let cs: Vec<char> = camel.chars().collect();
        let mut out = String::new();
        for (i, &c) in cs.iter().enumerate() {
            if c == '_' {
                if !out.ends_with('_') && !out.is_empty() {
                    out.push('_');
                }
                continue;
            }
            if i > 0 && c.is_ascii_uppercase() {
                let prev = cs[i - 1];
                let next_lower = cs.get(i + 1).is_some_and(|n| n.is_ascii_lowercase());
                if (prev.is_ascii_lowercase()
                    || prev.is_ascii_digit()
                    || (prev.is_ascii_uppercase() && next_lower))
                    && !out.ends_with('_')
                {
                    out.push('_');
                }
            }
            out.push(c.to_ascii_uppercase());
        }
        out
    }

    /// All generated identifiers, checked for validity and uniqueness.
    pub struct Names {
        /// Rust `Term` variant per terminal.
        pub term: Vec<String>,
        /// `SCREAMING` form per terminal (Stave `T_*` constants).
        pub term_const: Vec<String>,
        /// Rust `NodeKind` variant per named rule (= the rule name).
        pub kind: Vec<String>,
        /// `SCREAMING` form per named rule.
        pub kind_const: Vec<String>,
        /// Constant name per override cell, e.g. `OVERRIDE_TYPE_BASE_LT`.
        pub override_const: Vec<String>,
    }

    fn unique(what: &str, names: &[String]) -> Result<(), Error> {
        let mut seen = BTreeSet::new();
        for n in names {
            if !seen.insert(n) {
                return Err(Error::new(format!("two {what} get the generated name `{n}`")));
            }
        }
        Ok(())
    }

    impl Names {
        pub fn new(t: &Tables) -> Result<Self, Error> {
            let mut term = Vec::new();
            for (s, &c) in t.terms.iter().zip(&t.term_is_class) {
                match term_variant(s, c) {
                    Some(v) => term.push(v),
                    None => return Err(Error::new(format!("terminal \"{s}\" has no generated name"))),
                }
            }
            let term_const: Vec<String> = term.iter().map(|v| screaming(v)).collect();
            let kind: Vec<String> = t.nts[..t.nkind].iter().map(|n| n.name.clone()).collect();
            for k in &kind {
                if RUST_KEYWORDS.contains(&k.as_str()) {
                    return Err(Error::new(format!("rule name `{k}` is a Rust keyword")));
                }
            }
            let kind_const: Vec<String> = kind.iter().map(|k| screaming(k)).collect();
            let override_const: Vec<String> = t
                .overrides
                .iter()
                .map(|o| {
                    format!("OVERRIDE_{}_{}", kind_const[o.origin as usize], term_const[o.term as usize])
                })
                .collect();
            unique("terminals", &term)?;
            unique("terminals", &term_const)?;
            unique("rules", &kind_const)?;
            unique("overrides", &override_const)?;
            Ok(Names { term, term_const, kind, kind_const, override_const })
        }
    }
}
