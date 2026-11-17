//! The pretty-printer (`stave fmt`).
//!
//! Prints a parsed file in the canonical layout. Tokens and comments are never added, removed, or
//! reordered: only the whitespace between them is chosen. On a line the spacing is fixed (spaces
//! around binary operators; none inside paths, calls, indexing, type arguments, or after unary
//! operators), indentation is 4 spaces, and there is no line width limit: the printer never wraps
//! a line by itself. Where lines break:
//!
//! - Every item starts a line.
//! - A block (fn body, statement block, `match` statement, trait/impl/extern body) holds one
//!   statement or member per line, `}` on its own line. It stays on one line (`{ return x; }`)
//!   only when the source has it on one line and it holds at most one statement or member, and so
//!   do the blocks inside it. The blocks of one `if`/`else` chain, and of a `for` and its `tail`,
//!   are broken together. `{` never starts a line.
//! - `else` and `tail` follow a broken block's `}` on the same line. After a one-line block they
//!   keep the source's line break (`if a { x(); }` / `else { y(); }` on two lines).
//! - A list (`( )`, `[ ]`, `.{ }`, struct/enum/dispatch body, value `match` arms) is broken after
//!   its opener and before its closer iff the source has a line break directly after the opener
//!   (or a comment directly before the closer); its elements then sit one level deeper.
//! - Any other line break of the source stays where the separator is a space (except before `{`),
//!   and before the `.` of a field or method step. Such a line is a continuation, one level deeper
//!   than the statement, member, or list element it belongs to; the line after a broken list's
//!   `,` and the line after an item's or member's `unsafe("..")` are not continuations.
//! - An empty pair of brackets is `{}`, `()`, `[]`, `.{}`.
//! - A comment on its own line keeps its own line, indented like the code that follows it (before
//!   a closer: like the elements of that block or list). A comment after code stays on that
//!   line; such comments on consecutive lines start in one column, one space after the longest
//!   code of the run. Trailing whitespace in a comment is removed.
//! - A shebang line (`#!` at offset 0) is kept as the first line, like a comment.
//! - Blank lines: one is kept where the source has one or more before an item, a statement, a
//!   member, a line of a broken list, or a comment; none after an opener, before a closer, or
//!   inside a statement. None at the start of the file; the file ends with one newline.
//!
//! Items that did not parse are copied verbatim (first token through last token).
//!
//! The layout comes from the parser itself: each item is re-run through the PDA with a trace that
//! records, per token, the grammar rule whose production consumed it (so `-` in `Unary` differs
//! from `-` in `Add`, `<` in `GenericArgs` from `<` in `CmpOp`, a `Block` `{` from a struct-literal
//! `.{`) and the node that owns it. Every separator choice is checked by re-lexing the last tokens
//! (`1 .5` must not become `1.5`, `& &x` not `&&x`, `f "s"` not `f"s"`), and the whole item is
//! re-lexed at the end: if its tokens differ, the item is copied verbatim and counted in
//! [`Printed::fallbacks`]. So the output always lexes to the same token kinds, and therefore parses
//! to the same AST.

use stave_diag::{Diagnostic, FileId};

use crate::generated::ll1_table as lt;
use crate::generated::node_kinds::NodeKind;
use crate::generated::terminals::Term;
use crate::lex::{Token, lex_kinds, shebang_end, token_end, token_text};
use crate::parse::{ParsedFile, Scratch, Trace, parse_file, run};

/// Spaces per indentation level.
const INDENT: usize = 4;

/// The printer's output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Printed {
    pub text: String,
    /// Ok items printed verbatim because the formatted text did not re-lex to the same tokens
    /// (a printer bug; tests assert it is 0), plus files with no items (copied verbatim).
    pub fallbacks: usize,
}

/// Why [`format_source`] gives no text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FormatError {
    /// The file has lexical or syntax diagnostics: it is not formatted.
    Diagnostics(Vec<Diagnostic>),
    /// The formatted text failed the self-check (a printer bug): what differs.
    Internal(&'static str),
}

/// The canonical layout of `src`, checked: the result parses without diagnostics to the same
/// tokens (kind and text), comments, and AST as `src`.
pub fn format_source(src: &[u8], file: FileId) -> Result<String, FormatError> {
    let pf = parse_file(src, file);
    if !pf.diagnostics.is_empty() || pf.stop.is_some() {
        return Err(FormatError::Diagnostics(pf.diagnostics));
    }
    let printed = print_file(&pf, src);
    if printed.fallbacks != 0 {
        return Err(FormatError::Internal("an item could not be laid out"));
    }
    let out = printed.text.as_bytes();
    let pf2 = parse_file(out, file);
    if !pf2.diagnostics.is_empty() || pf2.stop.is_some() {
        return Err(FormatError::Internal("the formatted text has diagnostics"));
    }
    let same_tokens = pf.tokens.len() == pf2.tokens.len()
        && (0..pf.tokens.len()).all(|i| {
            pf.tokens[i].kind == pf2.tokens[i].kind
                && token_text(src, &pf.tokens, i) == token_text(out, &pf2.tokens, i)
        });
    if !same_tokens {
        return Err(FormatError::Internal("the formatted text has other tokens"));
    }
    if !pf.ast.nodes().eq(pf2.ast.nodes()) {
        return Err(FormatError::Internal("the formatted text has another AST"));
    }
    let (g, g2) = (Gaps::scan(src, &pf.tokens), Gaps::scan(out, &pf2.tokens));
    let same_comments = g.first == g2.first
        && g.comments.len() == g2.comments.len()
        && g.comments.iter().zip(&g2.comments).all(|(a, b)| a.text(src) == b.text(out));
    if !same_comments {
        return Err(FormatError::Internal("the formatted text has other comments"));
    }
    Ok(printed.text)
}

/// A `//` comment between two tokens, or the shebang line.
#[derive(Clone, Copy, Debug)]
struct Comment {
    start: u32,
    /// End, without trailing whitespace.
    end: u32,
    /// Newlines between the previous token or comment and this comment (0: it follows code on
    /// the same line).
    nl: u32,
}

impl Comment {
    fn text(self, src: &[u8]) -> &[u8] {
        src.get(self.start as usize..self.end as usize).unwrap_or_default()
    }

    fn push_to(self, src: &[u8], out: &mut String) {
        out.push_str(&String::from_utf8_lossy(self.text(src)));
    }
}

/// What the source holds between consecutive tokens: comments and newlines.
struct Gaps {
    comments: Vec<Comment>,
    /// Per token `i` (the `EOF` token included), `first[i]..first[i + 1]`: the comments between
    /// token `i - 1` (or the start of the file) and token `i`.
    first: Vec<u32>,
    /// Per token: newlines between the last of those comments (else the previous token) and it.
    nl: Vec<u32>,
}

impl Gaps {
    fn scan(src: &[u8], toks: &[Token]) -> Gaps {
        let mut g = Gaps { comments: Vec::new(), first: Vec::with_capacity(toks.len() + 1), nl: Vec::new() };
        let mut from = 0usize;
        for (i, t) in toks.iter().enumerate() {
            g.first.push(g.comments.len() as u32);
            let to = (t.start as usize).min(src.len());
            let mut nl = 0;
            let mut j = from.min(to);
            if i == 0 {
                // the shebang line is kept like a comment that starts the file
                let stop = shebang_end(src).min(to);
                if stop > 0 {
                    let mut end = stop;
                    while end > 0 && matches!(src[end - 1], b' ' | b'\t' | b'\r') {
                        end -= 1;
                    }
                    g.comments.push(Comment { start: 0, end: end as u32, nl: 0 });
                    j = stop;
                }
            }
            while j < to {
                if src[j] == b'\n' {
                    nl += 1;
                    j += 1;
                } else if src[j] == b'/' && src.get(j + 1) == Some(&b'/') {
                    let stop = src[j..to].iter().position(|&c| c == b'\n').map_or(to, |p| j + p);
                    let mut end = stop;
                    while end > j && matches!(src[end - 1], b' ' | b'\t' | b'\r') {
                        end -= 1;
                    }
                    g.comments.push(Comment { start: j as u32, end: end as u32, nl });
                    nl = 0;
                    j = stop;
                } else {
                    j += 1;
                }
            }
            g.nl.push(nl);
            from = token_end(src, toks, i) as usize;
        }
        g.first.push(g.comments.len() as u32);
        g
    }

    /// The comments before token `i`.
    fn before(&self, i: usize) -> &[Comment] {
        let lo = self.first.get(i).copied().unwrap_or(0) as usize;
        let hi = self.first.get(i + 1).copied().unwrap_or(0) as usize;
        self.comments.get(lo..hi).unwrap_or_default()
    }

    /// Newlines between the last comment before token `i` (else the previous token) and it.
    fn nl(&self, i: usize) -> u32 {
        self.nl.get(i).copied().unwrap_or(0)
    }

    /// All newlines between token `i - 1` and token `i`.
    fn newlines(&self, i: usize) -> u32 {
        self.before(i).iter().map(|c| c.nl).sum::<u32>() + self.nl(i)
    }
}

/// Prints `pf` (parsed from `src`). Files with no items (E0001, E0002, E0005, E0016) are copied
/// verbatim, with invalid UTF-8 replaced.
pub fn print_file(pf: &ParsedFile, src: &[u8]) -> Printed {
    if pf.stop.is_some() {
        return Printed { text: String::from_utf8_lossy(src).into_owned(), fallbacks: 1 };
    }
    let gaps = Gaps::scan(src, &pf.tokens);
    let mut text = String::new();
    // Offsets in `text` of the space before each comment that follows code on its line.
    let mut marks: Vec<usize> = Vec::new();
    let mut fallbacks = 0;
    let mut scratch = Scratch::default();
    for it in &pf.items {
        let (b, e) = (it.start as usize, it.end as usize);
        if b >= e || e > pf.tokens.len() {
            continue;
        }
        top_level_comments(&mut text, &mut marks, &gaps, src, b);
        if !text.is_empty() {
            text.push('\n');
            if gaps.nl(b) > 1 {
                text.push('\n');
            }
        }
        let formatted = if it.ok { format_item(src, &pf.tokens, &gaps, b, e, &mut scratch) } else { None };
        match formatted {
            Some((s, item_marks)) => {
                marks.extend(item_marks.iter().map(|m| m + text.len()));
                text.push_str(&s);
            }
            None => {
                if it.ok {
                    fallbacks += 1;
                }
                text.push_str(&verbatim(src, &pf.tokens, b, e));
            }
        }
    }
    let eof = pf.tokens.len().saturating_sub(1);
    top_level_comments(&mut text, &mut marks, &gaps, src, eof);
    let ends_line = pf.items.last().is_none_or(|it| it.ok) || !gaps.before(eof).is_empty();
    if ends_line && !text.is_empty() {
        text.push('\n');
    }
    let mut text = align_comments(&text, &marks);
    if text.starts_with('\u{FEFF}') {
        // the first item is a dropped item, copied verbatim, that starts with a run of bad bytes
        // beginning with U+FEFF (a U+FEFF that was not at offset 0 of the source): keep it off
        // offset 0, where it would read as a byte-order mark (E0017)
        text.insert(0, '\n');
    }
    if shebang_end(src) == 0 && text.starts_with("#!") {
        // likewise a dropped item that starts with `#!` (not at offset 0 of the source): at
        // offset 0 it would read as a shebang line
        text.insert(0, '\n');
    }
    Printed { text, fallbacks }
}

/// Appends the comments before token `i`, a token at item level (or `EOF`).
fn top_level_comments(text: &mut String, marks: &mut Vec<usize>, gaps: &Gaps, src: &[u8], i: usize) {
    for c in gaps.before(i) {
        if text.is_empty() {
            // the first line of the file
        } else if c.nl == 0 {
            marks.push(text.len());
            text.push(' ');
        } else {
            text.push('\n');
            if c.nl > 1 {
                text.push('\n');
            }
        }
        c.push_to(src, text);
    }
}

fn verbatim(src: &[u8], toks: &[Token], b: usize, e: usize) -> String {
    if b >= e || e > toks.len() {
        return String::new();
    }
    let s = toks[b].start as usize;
    let end = token_end(src, toks, e - 1) as usize;
    String::from_utf8_lossy(src.get(s..end).unwrap_or_default()).into_owned()
}

/// A named node of the traced parse.
struct RoleNode {
    kind: u8,
    /// Index of the parent node, `u32::MAX` for the root.
    parent: u32,
    /// Direct children that are statements or members (`Stmt`, `StmtArm`, `TraitFn`, `ExternFn`,
    /// `FnDecl`).
    elems: u32,
}

/// Per token of the item: the nonterminal whose production first consumed it, and the innermost
/// named node open at that point.
struct Roles {
    first: usize,
    producer: Vec<u16>,
    owner: Vec<u32>,
    nodes: Vec<RoleNode>,
    open: Vec<u32>,
}

impl Trace for Roles {
    const ON: bool = true;

    fn open(&mut self, kind: u8, _tok: usize) {
        let parent = self.open.last().copied().unwrap_or(u32::MAX);
        let element = matches!(
            NodeKind::from_id(kind),
            Some(
                NodeKind::Stmt
                    | NodeKind::StmtArm
                    | NodeKind::TraitFn
                    | NodeKind::ExternFn
                    | NodeKind::FnDecl
            )
        );
        if element && let Some(p) = self.nodes.get_mut(parent as usize) {
            p.elems += 1;
        }
        self.open.push(self.nodes.len() as u32);
        self.nodes.push(RoleNode { kind, parent, elems: 0 });
    }

    fn close(&mut self) {
        self.open.pop();
    }

    fn term(&mut self, tok: usize, _term: u16, producer: u16) {
        let k = tok.wrapping_sub(self.first);
        if let Some(p) = self.producer.get_mut(k)
            && *p == u16::MAX
        {
            *p = producer;
            self.owner[k] = self.open.last().copied().unwrap_or(u32::MAX);
        }
    }
}

impl Roles {
    /// The node that owns token `i`, and its parent's kind.
    fn owner(&self, i: usize) -> Option<(&RoleNode, Option<NodeKind>)> {
        let n = self.nodes.get(*self.owner.get(i.wrapping_sub(self.first))? as usize)?;
        let parent = self.nodes.get(n.parent as usize).and_then(|p| NodeKind::from_id(p.kind));
        Some((n, parent))
    }

    /// The rule that produced token `i`.
    fn origin(&self, i: usize) -> Option<NodeKind> {
        let p = *self.producer.get(i.wrapping_sub(self.first))?;
        lt::NT_ORIGIN.get(p as usize).and_then(|&o| NodeKind::from_id(o as u8))
    }

    fn kind(&self, node: u32) -> Option<NodeKind> {
        self.nodes.get(node as usize).and_then(|n| NodeKind::from_id(n.kind))
    }

    fn parent(&self, node: u32) -> u32 {
        self.nodes.get(node as usize).map_or(u32::MAX, |n| n.parent)
    }

    /// For the `{` (token `i`) of a block of an `if`/`else` chain or of a `for` with its `tail`:
    /// the statement's node. The blocks of one statement are laid out alike.
    fn statement(&self, i: usize) -> Option<u32> {
        let n = *self.owner.get(i.wrapping_sub(self.first))?;
        if self.kind(n) != Some(NodeKind::Block) {
            return None;
        }
        let mut p = self.parent(n);
        match self.kind(p)? {
            NodeKind::IfStmt => {
                while self.kind(self.parent(p)) == Some(NodeKind::IfStmt) {
                    p = self.parent(p);
                }
                Some(p)
            }
            NodeKind::ForStmt => Some(p),
            NodeKind::TailClause => Some(self.parent(p)),
            _ => None,
        }
    }

    /// Token `i` is the `)` of an `unsafe("..")` that prefixes an item, a method, or a trait
    /// method (not a statement or an operand).
    fn ends_unsafe_prefix(&self, toks: &[Token], i: usize) -> bool {
        is(toks[i].kind, Term::RParen)
            && self.owner(i).is_some_and(|(n, parent)| {
                NodeKind::from_id(n.kind) == Some(NodeKind::Unsafe)
                    && !matches!(parent, Some(NodeKind::Unary | NodeKind::Stmt))
            })
    }
}

/// How a bracket pair is laid out.
#[derive(Clone, Copy, Default)]
struct Bracket {
    /// Broken: a line break after the opener and before the closer.
    ml: bool,
    /// Holds statements or members (else a list).
    block: bool,
    /// A closure body: the expression goes on after its `}`.
    expr: bool,
}

/// An open bracket while printing.
#[derive(Clone, Copy)]
struct Open {
    br: Bracket,
    /// Indentation of the closer when broken; the elements sit one level deeper.
    base: usize,
    /// Indentation of a statement, member, or list element directly inside: one level deeper
    /// than `base` when broken, else as around the bracket.
    elem: usize,
}

fn is_opener(k: u8) -> bool {
    is(k, Term::LParen) || is(k, Term::LBracket) || is(k, Term::LBrace) || is(k, Term::DotLBrace)
}

fn is_closer(k: u8) -> bool {
    is(k, Term::RParen) || is(k, Term::RBracket) || is(k, Term::RBrace)
}

/// The layout of every bracket pair of the item, indexed by the opener's token (item-relative).
fn brackets(toks: &[Token], gaps: &Gaps, roles: &Roles, b: usize, e: usize) -> Option<Vec<Bracket>> {
    let mut out = vec![Bracket::default(); e - b];
    // (opener, a bracket inside is broken, newlines inside)
    let mut stack: Vec<(usize, bool, u32)> = Vec::new();
    // Blocks of `if`/`else` chains and `for`/`tail` statements: (statement, opener); the
    // statements with a broken block.
    let mut chained: Vec<(u32, usize)> = Vec::new();
    let mut broken: Vec<u32> = Vec::new();
    for i in b..e {
        let k = toks[i].kind;
        if i > b
            && let Some(top) = stack.last_mut()
        {
            top.2 += gaps.newlines(i);
        }
        if is_opener(k) {
            stack.push((i, false, 0));
        } else if is_closer(k) {
            let (o, inner_ml, nl) = stack.pop()?;
            let (node, parent) = roles.owner(o)?;
            let kind = NodeKind::from_id(node.kind);
            let block = is(toks[o].kind, Term::LBrace)
                && matches!(
                    kind,
                    Some(
                        NodeKind::Block
                            | NodeKind::TraitDecl
                            | NodeKind::ImplDecl
                            | NodeKind::ExternItem
                            | NodeKind::Stmt
                    )
                );
            let expr = kind == Some(NodeKind::Block) && parent == Some(NodeKind::Primary);
            let ml = if i == o + 1 {
                !gaps.before(i).is_empty()
            } else if block {
                nl > 0 || inner_ml || node.elems > 1
            } else {
                gaps.newlines(o + 1) > 0 || !gaps.before(i).is_empty()
            };
            out[o - b] = Bracket { ml, block, expr };
            if let Some(top) = stack.last_mut() {
                top.1 |= ml || inner_ml;
                top.2 += nl;
            }
            if let Some(s) = roles.statement(o) {
                chained.push((s, o));
                if ml {
                    broken.push(s);
                }
            }
        }
    }
    // A bracket around a statement that gets broken here already holds a broken block of it.
    broken.sort_unstable();
    for (s, o) in chained {
        let empty = toks.get(o + 1).is_some_and(|t| is_closer(t.kind));
        if !empty && broken.binary_search(&s).is_ok() {
            out[o - b].ml = true;
        }
    }
    stack.is_empty().then_some(out)
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

fn is(k: u8, t: Term) -> bool {
    k == t as u8
}

fn push_line(out: &mut String, blank: bool, indent: usize) {
    out.push('\n');
    if blank {
        out.push('\n');
    }
    for _ in 0..indent {
        out.push(' ');
    }
}

/// The formatted item and the offsets in it of the space before each comment that follows code,
/// or `None` if the item cannot be printed safely.
fn format_item(
    src: &[u8],
    toks: &[Token],
    gaps: &Gaps,
    b: usize,
    e: usize,
    scratch: &mut Scratch,
) -> Option<(String, Vec<usize>)> {
    let n = e - b;
    let mut roles = Roles {
        first: b,
        producer: vec![u16::MAX; n],
        owner: vec![u32::MAX; n],
        nodes: Vec::new(),
        open: Vec::new(),
    };
    let mut nodes = Vec::new();
    let start = if is(toks[b].kind, Term::KwImport) { NodeKind::Import.nt() } else { NodeKind::Item.nt() };
    run(toks, b, e, start, &mut nodes, scratch, &mut roles).ok()?;
    let layout = brackets(toks, gaps, &roles, b, e)?;
    let mut stack: Vec<Open> = Vec::new();
    let mut out = String::new();
    let mut last: Vec<Emitted> = Vec::new();
    // Indentation of the current output line.
    let mut line = 0;
    // The previous token opened the innermost bracket.
    let mut prev_open = false;
    // The previous token ended a statement or member of the innermost bracket (a block).
    let mut prev_end = false;
    // The previous token closed a one-line block.
    let mut prev_flat = false;
    // Offsets in `out` of the space before each comment that follows code on its line.
    let mut marks = Vec::new();
    for i in b..e {
        let ck = toks[i].kind;
        let co = roles.origin(i);
        let text = String::from_utf8_lossy(token_text(src, toks, i)).into_owned();
        let closing = if is_closer(ck) { stack.pop() } else { None };
        let top = stack.last().copied();
        // Where a statement, member, or list element starts in the innermost broken bracket.
        let elem = top.map_or(0, |o| o.elem);
        let mut glued = true;
        if i > b {
            let (pk, po) = (toks[i - 1].kind, roles.origin(i - 1));
            let comments = gaps.before(i);
            // The separator, and the indentation if the token starts a line.
            let (mut sep, indent) = if let Some(c) = closing {
                if c.br.ml {
                    (Sep::Newline, c.base)
                } else if prev_open {
                    (Sep::None, elem + INDENT)
                } else if c.br.block {
                    (Sep::Space, elem + INDENT)
                } else {
                    (style(pk, po, ck, co), elem + INDENT)
                }
            } else if prev_open {
                match top {
                    Some(o) if o.br.ml => (Sep::Newline, o.base + INDENT),
                    _ => (style(pk, po, ck, co), elem + INDENT),
                }
            } else if prev_end {
                // `else` and `tail` stay on the line of a broken block's `}`; after a one-line
                // block they keep the source's line break.
                let stays = if is(ck, Term::KwElse) || is(ck, Term::KwTail) {
                    !(prev_flat && gaps.nl(i) > 0)
                } else {
                    !top.is_some_and(|o| o.br.ml)
                };
                (if stays { Sep::Space } else { Sep::Newline }, elem)
            } else {
                let st = style(pk, po, ck, co);
                let starts_element = (is(pk, Term::Comma) && top.is_some_and(|o| o.br.ml && !o.br.block))
                    || roles.ends_unsafe_prefix(toks, i - 1);
                let may_break = match st {
                    Sep::Space => !is(ck, Term::LBrace),
                    _ => is(ck, Term::Dot) && co == Some(NodeKind::Postfix),
                };
                let sep = if may_break && gaps.nl(i) > 0 { Sep::Newline } else { st };
                (sep, if starts_element { elem } else { elem + INDENT })
            };
            let before_closer = closing.is_some_and(|c| c.br.ml);
            for (k, c) in comments.iter().enumerate() {
                if c.nl == 0 {
                    marks.push(out.len());
                    out.push(' ');
                } else {
                    let at = closing.filter(|c| c.br.ml).map_or(indent, |c| c.base + INDENT);
                    push_line(&mut out, c.nl > 1 && !(k == 0 && prev_open), at);
                }
                c.push_to(src, &mut out);
            }
            if !comments.is_empty() {
                sep = Sep::Newline;
            }
            glued = sep == Sep::None;
            if !relex_ok(&last, glued, &text, ck) {
                glued = !glued;
                if !comments.is_empty() || !relex_ok(&last, glued, &text, ck) {
                    return None;
                }
            }
            if glued {
                // nothing between the tokens
            } else if sep == Sep::Newline {
                // A blank line stays before a statement, member, or list line, not inside one.
                let blank =
                    gaps.nl(i) > 1 && indent == elem && !before_closer && !(prev_open && comments.is_empty());
                push_line(&mut out, blank, indent);
                line = indent;
            } else {
                out.push(' ');
            }
        }
        out.push_str(&text);
        let after_dot = i > b && glued && last.last().is_some_and(|p| is(p.kind, Term::Dot));
        last.push(Emitted { text, kind: ck, after_dot, glued });
        if last.len() > 2 {
            last.remove(0);
        }
        prev_open = false;
        prev_end = false;
        prev_flat = closing.is_some_and(|c| c.br.block && !c.br.ml);
        let in_block = stack.last().is_some_and(|o| o.br.block);
        if is_opener(ck) {
            let br = layout[i - b];
            let base = if br.block && !br.expr { elem } else { line };
            stack.push(Open { br, base, elem: if br.ml { base + INDENT } else { elem } });
            prev_open = true;
        } else if is(ck, Term::Semi) {
            prev_end = in_block;
        } else if let Some(c) = closing {
            prev_end = in_block && c.br.block && !c.br.expr;
        }
    }
    // Final check: the item text re-lexes to the item's token kinds.
    let mut got = Vec::with_capacity(n);
    lex_kinds(out.as_bytes(), false, &mut got);
    let same = got.len() == n && got.iter().zip(&toks[b..e]).all(|(g, t)| g.0 == t.kind);
    same.then_some((out, marks))
}

/// Pads the space at each of `marks` (ascending offsets in `text`, each before a comment that
/// follows code) so that such comments on consecutive lines start in the same column: one space
/// after the longest code of the run.
fn align_comments(text: &str, marks: &[usize]) -> String {
    // (mark, line number, width of the code before the mark)
    let mut at: Vec<(usize, usize, usize)> = Vec::with_capacity(marks.len());
    let (mut pos, mut line, mut line_start) = (0, 0, 0);
    for &m in marks {
        for (k, c) in text.as_bytes()[pos..m].iter().enumerate() {
            if *c == b'\n' {
                line += 1;
                line_start = pos + k + 1;
            }
        }
        pos = m;
        at.push((m, line, text[line_start..m].chars().count()));
    }
    let mut out = String::with_capacity(text.len() + marks.len());
    let (mut copied, mut k) = (0, 0);
    while k < at.len() {
        let mut end = k + 1;
        while end < at.len() && at[end].1 == at[end - 1].1 + 1 {
            end += 1;
        }
        let width = at[k..end].iter().map(|a| a.2).max().unwrap_or(0);
        for &(m, _, w) in &at[k..end] {
            out.push_str(&text[copied..m]);
            for _ in w..width {
                out.push(' ');
            }
            copied = m;
        }
        k = end;
    }
    out.push_str(&text[copied..]);
    out
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
    // `unsafe("..") (*p).f`: the operand is not a call of the obligation.
    if is(pk, T::RParen) && po == Some(N::Unsafe) {
        return Sep::Space;
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
    // Call, parameter list, tuple variant / pattern, `needs(`, `without(`, `requires(`,
    // `ensures(`, `unsafe(`, `fn(`.
    if is(ck, T::LParen) && co != Some(N::TailClause) {
        let operand_end = [
            T::Ident,
            T::KwFn,
            T::KwNeeds,
            T::KwWithout,
            T::KwRequires,
            T::KwEnsures,
            T::KwUnsafe,
            T::KwSelfType,
            T::RParen,
            T::RBracket,
            T::Question,
        ]
        .iter()
        .any(|&t| is(pk, t))
            || ((is(pk, T::Gt) || is(pk, T::GtGt)) && generic(po));
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
