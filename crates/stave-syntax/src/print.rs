//! The pretty-printer.
//!
//! Prints a parsed file in one canonical layout: one statement or member per line, 4-space
//! indentation inside function bodies, trait/impl/extern bodies and statement `match` arms;
//! spaces around binary operators; none inside paths, calls, indexing, type arguments, or after
//! unary operators. Comments are not kept. Dropped items (and the run before the first item) are
//! copied verbatim.
//!
//! The layout comes from the parser itself: each item is re-run through the PDA with a trace that
//! records, per token, the grammar rule whose production consumed it (so `-` in `Unary` differs
//! from `-` in `Add`, `<` in `GenericArgs` from `<` in `CmpOp`, a `Block` `{` from a struct-literal
//! `.{`). Every separator choice is checked by re-lexing the last tokens (`1 .5` must not become
//! `1.5`, `& &x` not `&&x`, `f "s"` not `f"s"`), and the whole item is re-lexed at the end: if
//! its tokens differ, the item is copied verbatim and counted in [`Printed::fallbacks`]. So the
//! output always lexes to the same token kinds, and therefore parses to the same AST.

use crate::generated::ll1_table as lt;
use crate::generated::node_kinds::NodeKind;
use crate::generated::terminals::Term;
use crate::items::ItemKind;
use crate::lex::{Token, lex_kinds, token_end};
use crate::parse::{ParsedFile, Scratch, Trace, run};

/// The printer's output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Printed {
    pub text: String,
    /// Ok items printed verbatim because the formatted text did not re-lex to the same tokens
    /// (a printer bug; tests assert it is 0), plus files with no items (copied verbatim).
    pub fallbacks: usize,
}

/// Prints `pf` (parsed from `src`). Files with no items (E0001, E0002, E0005, E0016) are copied
/// verbatim, with invalid UTF-8 replaced.
pub fn print_file(pf: &ParsedFile, src: &[u8]) -> Printed {
    if pf.stop.is_some() {
        return Printed { text: String::from_utf8_lossy(src).into_owned(), fallbacks: 1 };
    }
    let mut text = String::new();
    let mut fallbacks = 0;
    let mut scratch = Scratch::default();
    let mut prev_kind: Option<ItemKind> = None;
    for it in &pf.items {
        if let Some(pk) = prev_kind {
            text.push_str(if pk == ItemKind::Import && it.kind == ItemKind::Import { "\n" } else { "\n\n" });
        }
        prev_kind = Some(it.kind);
        let (b, e) = (it.start as usize, it.end as usize);
        let formatted = if it.ok { format_item(src, &pf.tokens, b, e, &mut scratch) } else { None };
        match formatted {
            Some(s) => text.push_str(&s),
            None => {
                if it.ok {
                    fallbacks += 1;
                }
                text.push_str(&verbatim(src, &pf.tokens, b, e));
            }
        }
    }
    if pf.items.last().is_some_and(|it| it.ok) {
        text.push('\n');
    }
    if text.starts_with('\u{FEFF}') {
        // the first item is a dropped item, copied verbatim, that starts with a run of bad bytes
        // beginning with U+FEFF (a U+FEFF that was not at offset 0 of the source): keep it off
        // offset 0, where it would read as a byte-order mark (E0017)
        text.insert(0, '\n');
    }
    Printed { text, fallbacks }
}

fn verbatim(src: &[u8], toks: &[Token], b: usize, e: usize) -> String {
    if b >= e || e > toks.len() {
        return String::new();
    }
    let s = toks[b].start as usize;
    let end = token_end(src, toks, e - 1) as usize;
    String::from_utf8_lossy(src.get(s..end).unwrap_or_default()).into_owned()
}

/// Per token of the item: the nonterminal whose production first consumed it.
struct Roles {
    first: usize,
    producer: Vec<u16>,
}

impl Trace for Roles {
    const ON: bool = true;

    fn term(&mut self, tok: usize, _term: u16, producer: u16) {
        if let Some(p) = self.producer.get_mut(tok.wrapping_sub(self.first))
            && *p == u16::MAX
        {
            *p = producer;
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Sep {
    None,
    Space,
    Newline,
}

/// One emitted token, for the re-lex checks.
#[derive(Clone)]
struct Emitted {
    text: String,
    kind: u8,
    /// Directly preceded by a `.` token in the output (lexed in after-dot mode).
    after_dot: bool,
    /// The separator printed before it (`""` or `" "` for lexing purposes).
    glued: bool,
}

fn origin(producer: u16) -> Option<NodeKind> {
    lt::NT_ORIGIN.get(producer as usize).and_then(|&o| NodeKind::from_id(o as u8))
}

fn is(k: u8, t: Term) -> bool {
    k == t as u8
}

/// The formatted item, or `None` if it cannot be printed safely.
fn format_item(src: &[u8], toks: &[Token], b: usize, e: usize, scratch: &mut Scratch) -> Option<String> {
    let mut roles = Roles { first: b, producer: vec![u16::MAX; e - b] };
    let mut nodes = Vec::new();
    let start = if is(toks[b].kind, Term::KwImport) { NodeKind::Import.nt() } else { NodeKind::Item.nt() };
    run(toks, b, e, start, &mut nodes, scratch, &mut roles).ok()?;
    let org = |i: usize| roles.producer.get(i - b).and_then(|&p| origin(p));
    // Bracket stack: (opener kind, multi-line brace).
    let mut stack: Vec<(u8, bool)> = Vec::new();
    let mut out = String::new();
    let mut last: Vec<Emitted> = Vec::new();
    let mut prev_open_ml = false;
    let mut prev_close_ml = false;
    for i in b..e {
        let ck = toks[i].kind;
        let co = org(i);
        let text = String::from_utf8_lossy(crate::lex::token_text(src, toks, i)).into_owned();
        let mut closing_ml = false;
        if (is(ck, Term::RParen) || is(ck, Term::RBracket) || is(ck, Term::RBrace))
            && let Some((_, ml)) = stack.pop()
        {
            closing_ml = ml && is(ck, Term::RBrace);
        }
        let depth = stack.iter().filter(|s| s.1).count();
        let sep = if i == b {
            Sep::None
        } else {
            let pk = toks[i - 1].kind;
            let po = org(i - 1);
            let body_level = stack.last().is_none_or(|s| s.1);
            if prev_open_ml {
                if closing_ml { Sep::Space } else { Sep::Newline }
            } else if closing_ml {
                Sep::Newline
            } else if (is(pk, Term::Semi) || prev_close_ml) && body_level {
                if is(ck, Term::KwElse) || is(ck, Term::KwTail) { Sep::Space } else { Sep::Newline }
            } else {
                style(pk, po, ck, co)
            }
        };
        let mut glued = sep == Sep::None;
        if !last.is_empty() && !relex_ok(&last, glued, &text, ck) {
            glued = !glued;
            if !relex_ok(&last, glued, &text, ck) {
                return None;
            }
        }
        if i > b {
            if glued {
                // keep it glued
            } else if sep == Sep::Newline {
                out.push('\n');
                for _ in 0..depth {
                    out.push_str("    ");
                }
            } else {
                out.push(' ');
            }
        }
        out.push_str(&text);
        let after_dot = glued && last.last().is_some_and(|p| is(p.kind, Term::Dot));
        last.push(Emitted { text, kind: ck, after_dot, glued });
        if last.len() > 2 {
            last.remove(0);
        }
        prev_open_ml = false;
        prev_close_ml = closing_ml;
        if is(ck, Term::LParen) || is(ck, Term::LBracket) || is(ck, Term::LBrace) || is(ck, Term::DotLBrace) {
            let ml = is(ck, Term::LBrace)
                && matches!(
                    co,
                    Some(
                        NodeKind::Block
                            | NodeKind::TraitDecl
                            | NodeKind::ImplDecl
                            | NodeKind::ExternItem
                            | NodeKind::Stmt
                    )
                );
            stack.push((ck, ml));
            prev_open_ml = ml;
        }
    }
    // Final check: the item text re-lexes to the item's token kinds.
    let mut got = Vec::with_capacity(e - b);
    lex_kinds(out.as_bytes(), false, &mut got);
    let same = got.len() == e - b && got.iter().zip(&toks[b..e]).all(|(g, t)| g.0 == t.kind);
    same.then_some(out)
}

/// The preferred separator between two tokens on one line.
fn style(pk: u8, po: Option<NodeKind>, ck: u8, co: Option<NodeKind>) -> Sep {
    use NodeKind as N;
    use Term as T;
    let generic = |o: Option<NodeKind>| matches!(o, Some(N::Generics | N::GenericArgs));
    let range = |o: Option<NodeKind>| matches!(o, Some(N::RangeExpr | N::Pattern));
    // Never a space before these.
    if [T::RParen, T::RBracket, T::Comma, T::Semi, T::Question, T::Colon, T::ColonColon, T::Dot, T::DotLBrace]
        .iter()
        .any(|&t| is(ck, t))
    {
        return Sep::None;
    }
    // Never a space after these.
    if [T::LParen, T::LBracket, T::Dot, T::ColonColon].iter().any(|&t| is(pk, t)) {
        return Sep::None;
    }
    // Unary operators and reference / pointer type constructors.
    let unary_tok = [T::Minus, T::Bang, T::Tilde, T::Star, T::Amp, T::AmpAmp].iter().any(|&t| is(pk, t));
    if unary_tok && matches!(po, Some(N::Unary | N::Literal | N::TypeBase | N::SelfParam)) {
        return Sep::None;
    }
    // Type arguments and generic parameters.
    if (is(ck, T::Lt) && generic(co)) || (is(pk, T::Lt) && generic(po)) {
        return Sep::None;
    }
    if (is(ck, T::Gt) || is(ck, T::GtGt) || is(ck, T::GtEq) || is(ck, T::GtGtEq)) && generic(co) {
        return Sep::None;
    }
    // Ranges.
    if ((is(ck, T::DotDot) || is(ck, T::DotDotEq)) && range(co))
        || ((is(pk, T::DotDot) || is(pk, T::DotDotEq)) && range(po))
    {
        return Sep::None;
    }
    // Call, parameter list, tuple variant / pattern, `needs(`, `unsafe(`, `fn(`.
    if is(ck, T::LParen) && co != Some(N::TailClause) {
        let operand_end =
            [T::Ident, T::KwFn, T::KwNeeds, T::KwUnsafe, T::KwSelfType, T::RParen, T::RBracket, T::Question]
                .iter()
                .any(|&t| is(pk, t))
                || (is(pk, T::Gt) && generic(po));
        if operand_end {
            return Sep::None;
        }
    }
    // Indexing.
    if is(ck, T::LBracket) && co == Some(N::Postfix) {
        return Sep::None;
    }
    Sep::Space
}

/// Whether appending `text` (kind `kind`) after the last emitted tokens, glued or after a space,
/// re-lexes to the same tokens.
fn relex_ok(last: &[Emitted], glued: bool, text: &str, kind: u8) -> bool {
    let mut s = String::new();
    let mut want: Vec<(u8, usize)> = Vec::new();
    for (k, t) in last.iter().enumerate() {
        if k > 0 && !t.glued {
            s.push(' ');
        }
        want.push((t.kind, s.len()));
        s.push_str(&t.text);
    }
    if !glued {
        s.push(' ');
    }
    want.push((kind, s.len()));
    s.push_str(text);
    let mut got = Vec::with_capacity(want.len());
    lex_kinds(s.as_bytes(), last.first().is_some_and(|t| t.after_dot), &mut got);
    got == want
}
