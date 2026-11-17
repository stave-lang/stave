//! Bracket check and item finder.
//!
//! 1. Bracket scan over `( [ { .{` / `) ] }`. An unbalanced file gets one E0016 at the first error
//!    of a left-to-right stack scan: the smallest-offset closer that is unmatched or closes an
//!    opener of another kind, else the innermost opener left unclosed at end of file. A balanced
//!    file whose depth exceeds [`MAX_DEPTH`] gets one E0005 at the first opener reaching depth
//!    `MAX_DEPTH + 1`. Either way the file has no items.
//! 2. Item cut: a token starts an item iff it is at bracket depth 0, is an item keyword
//!    (`import pub test unsafe rec fn struct enum view trait impl const static dispatch extern type
//!    allocator kernel`), and follows `;`, `}`, or the start of the file. The tokens before the first item
//!    start are a [`ItemKind::PreItem`] (always dropped, E0004). An `import` item after a
//!    non-import item is a [`ItemKind::LateImport`] (dropped, E0004 at `import`).

use stave_diag::{Diagnostic, FileId, Span};

use crate::generated::terminals::Term;
use crate::lex::Token;
use crate::make_diag;

/// Deepest accepted bracket nesting.
pub const MAX_DEPTH: usize = 10_000;

/// What an item's tokens start with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ItemKind {
    /// The tokens before the first item start: always dropped (E0004 at its first token).
    PreItem,
    /// An `import` before any non-import item.
    Import,
    /// An `import` after a non-import item: dropped (E0004 at `import`).
    LateImport,
    /// Any other item (`Item` rule of grammar.ebnf).
    Item,
}

/// One item of the cut: its kind and token range `[start, end)` (file token indices; the `EOF`
/// token belongs to no item).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ItemSpan {
    pub kind: ItemKind,
    pub start: u32,
    pub end: u32,
}

/// The items of one file, in file order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ItemTable {
    pub items: Vec<ItemSpan>,
}

fn is_item_keyword(kind: u8) -> bool {
    const KW: [Term; 18] = [
        Term::KwImport,
        Term::KwPub,
        Term::KwTest,
        Term::KwUnsafe,
        Term::KwRec,
        Term::KwFn,
        Term::KwStruct,
        Term::KwEnum,
        Term::KwView,
        Term::KwTrait,
        Term::KwImpl,
        Term::KwConst,
        Term::KwStatic,
        Term::KwDispatch,
        Term::KwExtern,
        Term::KwType,
        Term::KwAllocator,
        Term::KwKernel,
    ];
    KW.iter().any(|&k| k as u8 == kind)
}

const L_PAREN: u8 = Term::LParen as u8;
const L_BRACKET: u8 = Term::LBracket as u8;
const L_BRACE: u8 = Term::LBrace as u8;
const DOT_L_BRACE: u8 = Term::DotLBrace as u8;
const R_PAREN: u8 = Term::RParen as u8;
const R_BRACKET: u8 = Term::RBracket as u8;
const R_BRACE: u8 = Term::RBrace as u8;

/// The closer of an opener kind.
#[inline]
fn closer_of(kind: u8) -> Option<u8> {
    match kind {
        L_PAREN => Some(R_PAREN),
        L_BRACKET => Some(R_BRACKET),
        L_BRACE | DOT_L_BRACE => Some(R_BRACE),
        _ => None,
    }
}

#[inline]
fn is_closer(kind: u8) -> bool {
    matches!(kind, R_PAREN | R_BRACKET | R_BRACE)
}

fn bracket_len(kind: u8) -> u32 {
    if kind == DOT_L_BRACE { 2 } else { 1 }
}

fn bracket_span(file: FileId, t: Token) -> Span {
    Span::new(file, t.start, bracket_len(t.kind))
}

/// Why a file has no items.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ItemsError {
    /// E0016: unbalanced brackets.
    Unbalanced(Option<Box<Diagnostic>>),
    /// E0005: nesting deeper than [`MAX_DEPTH`].
    TooDeep(Option<Box<Diagnostic>>),
}

impl ItemsError {
    /// The diagnostic (`None` only if the registry entry could not be rendered, a compiler bug).
    pub fn diagnostic(&self) -> Option<&Diagnostic> {
        match self {
            ItemsError::Unbalanced(d) | ItemsError::TooDeep(d) => d.as_deref(),
        }
    }

    pub fn into_diagnostic(self) -> Option<Diagnostic> {
        match self {
            ItemsError::Unbalanced(d) | ItemsError::TooDeep(d) => d.map(|b| *b),
        }
    }
}

/// Checks the brackets of a file's tokens and cuts it into items. `tokens`
/// is the output of [`crate::lex`] (ending in `EOF`). Error tokens are not brackets.
pub fn items(tokens: &[Token], file: FileId) -> Result<ItemTable, ItemsError> {
    check_brackets(tokens, file)?;
    Ok(cut(tokens))
}

fn check_brackets(tokens: &[Token], file: FileId) -> Result<(), ItemsError> {
    let mut open: Vec<Token> = Vec::new();
    let mut too_deep: Option<Token> = None;
    for &t in tokens {
        if closer_of(t.kind).is_some() {
            open.push(t);
            if open.len() > MAX_DEPTH && too_deep.is_none() {
                too_deep = Some(t);
            }
        } else if is_closer(t.kind) {
            let closer = t.kind;
            let closer_name = crate::lex::kind_name(closer);
            match open.last() {
                None => {
                    let d = make_diag(
                        "E0016",
                        Some("unmatched"),
                        &[("closer", closer_name)],
                        bracket_span(file, t),
                        &[],
                    );
                    return Err(ItemsError::Unbalanced(d.map(Box::new)));
                }
                Some(&o) if closer_of(o.kind) != Some(closer) => {
                    let d = make_diag(
                        "E0016",
                        Some("mismatched"),
                        &[("closer", closer_name), ("opener", crate::lex::kind_name(o.kind))],
                        bracket_span(file, t),
                        &[("opener", bracket_span(file, o))],
                    );
                    return Err(ItemsError::Unbalanced(d.map(Box::new)));
                }
                Some(_) => {
                    open.pop();
                }
            }
        }
    }
    if let Some(&o) = open.last() {
        let d = make_diag(
            "E0016",
            Some("unclosed"),
            &[("opener", crate::lex::kind_name(o.kind))],
            bracket_span(file, o),
            &[],
        );
        return Err(ItemsError::Unbalanced(d.map(Box::new)));
    }
    if let Some(o) = too_deep {
        let d = make_diag("E0005", None, &[], bracket_span(file, o), &[]);
        return Err(ItemsError::TooDeep(d.map(Box::new)));
    }
    Ok(())
}

/// The item cut of a balanced file.
fn cut(tokens: &[Token]) -> ItemTable {
    let n = tokens.iter().rposition(|t| t.kind == Term::Eof as u8).unwrap_or(tokens.len());
    let n = n.min(u32::MAX as usize);
    let mut starts: Vec<u32> = Vec::new();
    let mut depth: usize = 0;
    for (i, t) in tokens[..n].iter().enumerate() {
        let after_end = i == 0 || {
            let p = tokens[i - 1].kind;
            p == Term::Semi as u8 || p == R_BRACE
        };
        if depth == 0 && after_end && is_item_keyword(t.kind) {
            starts.push(i as u32);
        }
        if closer_of(t.kind).is_some() {
            depth += 1;
        } else if is_closer(t.kind) {
            depth = depth.saturating_sub(1);
        }
    }
    let mut items = Vec::with_capacity(starts.len() + 1);
    let first = starts.first().copied().unwrap_or(n as u32);
    if first > 0 {
        items.push(ItemSpan { kind: ItemKind::PreItem, start: 0, end: first });
    }
    let mut seen_item = false;
    for (k, &s) in starts.iter().enumerate() {
        let end = starts.get(k + 1).copied().unwrap_or(n as u32);
        let kind = if tokens[s as usize].kind == Term::KwImport as u8 {
            if seen_item { ItemKind::LateImport } else { ItemKind::Import }
        } else {
            seen_item = true;
            ItemKind::Item
        };
        items.push(ItemSpan { kind, start: s, end });
    }
    ItemTable { items }
}
