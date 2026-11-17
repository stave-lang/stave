//! Per-item syntax access for the type checker: tokens, the AST, resolutions, bracket matching,
//! node end tokens, and spans.
//!
//! The AST stores only nonterminal nodes with their first token, so operators, names, and the
//! end of a node are read from the token stream. [`Brackets`] matches every opener (`( [ { .{`)
//! of an item to its closer once, so a node's end token is found in O(1) per step
//! ([`ItemCx::end_tok`], iterative).

use stave_diag::{FileId, Span};
use stave_syntax::{ItemAst, NodeKind, Term, Token};

use crate::ids::{ItemId, Sym};
use crate::resolve::{Deferred, ItemRes, Use};
use crate::scan;
use crate::{File, Program};

/// Matching closers of one item's tokens.
#[derive(Clone, Debug, Default)]
pub struct Brackets {
    /// First token of the item.
    pub base: u32,
    /// Per token (from `base`): the matching closer, or `u32::MAX`.
    pub close: Vec<u32>,
}

impl Brackets {
    pub fn new(toks: &[Token], start: u32, end: u32) -> Brackets {
        let n = end.saturating_sub(start) as usize;
        let mut close = vec![u32::MAX; n];
        let mut stack: Vec<u32> = Vec::new();
        for i in start..end.min(toks.len() as u32) {
            let k = toks[i as usize].kind;
            if k == Term::LParen as u8
                || k == Term::LBracket as u8
                || k == Term::LBrace as u8
                || k == Term::DotLBrace as u8
            {
                stack.push(i);
            } else if (k == Term::RParen as u8 || k == Term::RBracket as u8 || k == Term::RBrace as u8)
                && let Some(o) = stack.pop()
            {
                close[(o - start) as usize] = i;
            }
        }
        Brackets { base: start, close }
    }

    /// The closer of the opener at token `t` (`t` itself if none).
    #[inline]
    pub fn close_of(&self, t: u32) -> u32 {
        match t.checked_sub(self.base).and_then(|i| self.close.get(i as usize)) {
            Some(&c) if c != u32::MAX => c,
            _ => t,
        }
    }
}

/// A node's children in source order. Up to [`Kids::INLINE`] are stored in place: most nodes have
/// one or two, and the walks ask for them at every node.
#[derive(Clone, Debug, Default)]
pub struct Kids {
    len: u32,
    inline: [u32; Kids::INLINE],
    /// All the children when they do not fit in place (else empty).
    spill: Vec<u32>,
}

impl Kids {
    pub const INLINE: usize = 8;

    /// The children of the iterator, which yields them last first.
    fn from_rev(rev: impl Iterator<Item = u32>) -> Kids {
        let mut k = Kids::default();
        for c in rev {
            if !k.spill.is_empty() {
                k.spill.push(c);
            } else if (k.len as usize) < Kids::INLINE {
                k.inline[k.len as usize] = c;
                k.len += 1;
            } else {
                k.spill.reserve(2 * Kids::INLINE);
                k.spill.extend_from_slice(&k.inline);
                k.spill.push(c);
            }
        }
        if k.spill.is_empty() {
            k.inline[..k.len as usize].reverse();
        } else {
            k.spill.reverse();
        }
        k
    }

    #[inline]
    pub fn as_slice(&self) -> &[u32] {
        if self.spill.is_empty() { &self.inline[..self.len as usize] } else { &self.spill }
    }
}

impl std::ops::Deref for Kids {
    type Target = [u32];

    #[inline]
    fn deref(&self) -> &[u32] {
        self.as_slice()
    }
}

impl<'k> IntoIterator for &'k Kids {
    type Item = &'k u32;
    type IntoIter = std::slice::Iter<'k, u32>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

impl IntoIterator for Kids {
    type Item = u32;
    type IntoIter = KidsIter;

    fn into_iter(self) -> KidsIter {
        let end = self.as_slice().len();
        KidsIter { kids: self, at: 0, end }
    }
}

/// [`Kids`] by value.
#[derive(Clone, Debug)]
pub struct KidsIter {
    kids: Kids,
    at: usize,
    end: usize,
}

impl Iterator for KidsIter {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        if self.at >= self.end {
            return None;
        }
        self.at += 1;
        self.kids.as_slice().get(self.at - 1).copied()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.end - self.at;
        (n, Some(n))
    }
}

impl DoubleEndedIterator for KidsIter {
    fn next_back(&mut self) -> Option<u32> {
        if self.at >= self.end {
            return None;
        }
        self.end -= 1;
        self.kids.as_slice().get(self.end).copied()
    }
}

impl ExactSizeIterator for KidsIter {}

/// One item's syntax.
#[derive(Clone, Copy)]
pub struct ItemCx<'p> {
    pub prog: &'p Program,
    pub item: ItemId,
    pub file: &'p File,
    pub ast: ItemAst<'p>,
    pub toks: &'p [Token],
    pub res: &'p ItemRes,
    pub br: &'p Brackets,
}

impl<'p> ItemCx<'p> {
    pub fn new(prog: &'p Program, item: ItemId, br: &'p Brackets) -> Option<ItemCx<'p>> {
        let it = prog.item(item)?;
        let file = prog.file(it.file)?;
        let res = prog.item_res(item)?;
        Some(ItemCx {
            prog,
            item,
            file,
            ast: file.parsed.item_ast(it.index as usize),
            toks: &file.parsed.tokens,
            res,
            br,
        })
    }

    #[inline]
    pub fn fid(&self) -> FileId {
        self.file.id
    }

    #[inline]
    pub fn kind(&self, n: u32) -> Option<NodeKind> {
        self.ast.kind(n)
    }

    /// First token of node `n`.
    #[inline]
    pub fn a(&self, n: u32) -> u32 {
        self.ast.get(n).map_or(0, |x| x.a)
    }

    #[inline]
    pub fn tk(&self, t: u32) -> u8 {
        scan::kind(self.toks, t as usize)
    }

    #[inline]
    pub fn is(&self, t: u32, term: Term) -> bool {
        self.tk(t) == term as u8
    }

    #[inline]
    pub fn sym(&self, t: u32) -> Sym {
        self.file.syms.get(t as usize).copied().unwrap_or(Sym::NONE)
    }

    pub fn name(&self, s: Sym) -> &'p str {
        self.prog.strings.get(s)
    }

    /// The byte offset of token `t`: [`ItemCx::tok_span`]'s offset, without finding the token's
    /// end (which re-lexes it).
    #[inline]
    pub fn tok_start(&self, t: u32) -> u32 {
        self.toks.get(t as usize).map_or(0, |x| x.start)
    }

    pub fn tok_span(&self, t: u32) -> Span {
        scan::tok_span(self.file.id, &self.file.bytes, self.toks, t as usize)
    }

    pub fn range_span(&self, a: u32, b: u32) -> Span {
        scan::range_span(self.file.id, &self.file.bytes, self.toks, a as usize, b as usize)
    }

    pub fn tok_text(&self, t: u32) -> &'p [u8] {
        stave_syntax::token_text(&self.file.bytes, self.toks, t as usize)
    }

    pub fn tok_str(&self, t: u32) -> &'p str {
        std::str::from_utf8(self.tok_text(t)).unwrap_or("")
    }

    /// The source text of a span (one file).
    pub fn span_text(&self, s: Span) -> &'p str {
        let lo = (s.offset as usize).min(self.file.bytes.len());
        let hi = (s.offset as usize + s.len as usize).min(self.file.bytes.len());
        std::str::from_utf8(&self.file.bytes[lo..hi]).unwrap_or("")
    }

    pub fn close(&self, t: u32) -> u32 {
        self.br.close_of(t)
    }

    /// Children of `n` in order, into `out` (cleared first).
    pub fn kids(&self, n: u32, out: &mut Vec<u32>) {
        out.clear();
        out.extend(self.ast.children_rev(n));
        out.reverse();
    }

    pub fn kids_vec(&self, n: u32) -> Kids {
        Kids::from_rev(self.ast.children_rev(n))
    }

    /// The child of `n` if it has exactly one. The nodes are in postorder: the last child is
    /// `n - 1`, and it is the only one when its subtree starts where `n`'s descendants do.
    #[inline]
    pub fn only_child(&self, n: u32) -> Option<u32> {
        let nodes = self.ast.nodes;
        let lo = n.saturating_sub(nodes.get(n as usize)?.b);
        if n <= lo {
            return None;
        }
        let c = n - 1;
        (c.saturating_sub(nodes.get(c as usize)?.b) <= lo).then_some(c)
    }

    pub fn child(&self, n: u32, k: NodeKind) -> Option<u32> {
        self.ast.child(n, k)
    }

    /// The first child of `n`.
    pub fn first_child(&self, n: u32) -> Option<u32> {
        self.ast.children_rev(n).last()
    }

    /// The last child of `n`.
    pub fn last_child(&self, n: u32) -> Option<u32> {
        self.ast.children_rev(n).next()
    }

    /// Whether `n` is an expression node (any of them can stand in an expression position).
    pub fn is_expr(&self, n: u32) -> bool {
        self.kind(n).is_some_and(crate::typeck::is_expr_kind)
    }

    /// The first expression child of `n`.
    pub fn expr_child(&self, n: u32) -> Option<u32> {
        self.ast.children_rev(n).filter(|&c| self.is_expr(c)).last()
    }

    /// The head `Primary` of a postfix chain: a `Postfix` node's first child, or a bare
    /// `Primary` (a chain without steps) itself.
    pub fn chain_head(&self, n: u32) -> Option<u32> {
        match self.kind(n)? {
            NodeKind::Postfix => self.first_child(n),
            NodeKind::Primary => Some(n),
            _ => None,
        }
    }

    /// Whether the `Primary` `p` starts with a name (an identifier or `Self`): the head of a
    /// path whose steps resolution recorded.
    pub fn is_path_head(&self, p: u32) -> bool {
        let a = self.a(p);
        self.is(a, Term::Ident) || self.is(a, Term::KwSelfType)
    }

    /// The uses keyed at `n`.
    pub fn uses(&self, n: u32) -> &'p [Use] {
        self.res.uses_at(n)
    }

    /// The use of the identifier at token `t` keyed at `n`.
    pub fn use_at(&self, n: u32, t: u32) -> Option<&'p Use> {
        self.uses(n).iter().find(|u| u.token == t)
    }

    /// The use of the identifier at byte offset `off` keyed at `n` (f-string holes).
    pub fn use_at_offset(&self, n: u32, off: u32) -> Option<&'p Use> {
        self.uses(n).iter().find(|u| u.offset == off)
    }

    pub fn deferred_at(&self, n: u32, t: u32) -> Option<&'p Deferred> {
        let d = &self.res.deferred;
        let lo = d.partition_point(|x| x.node < n);
        d[lo..].iter().take_while(|x| x.node == n).find(|x| x.token == t)
    }

    /// The end token (inclusive) of an identifier chain starting at `t`.
    pub fn chain_end(&self, t: u32) -> u32 {
        let mut p = t;
        while self.is(p + 1, Term::Dot) && self.is(p + 2, Term::Ident) {
            p += 2;
        }
        p
    }

    /// The end (exclusive) of a generic argument list opening at `lt`: `(next, partial)`.
    pub fn angle_end(&self, lt: u32) -> (u32, bool) {
        let (e, p) = scan::angle_end(self.toks, lt as usize);
        (e as u32, p)
    }

    /// The last token (inclusive) of node `n`. Iterative over trailing children.
    pub fn end_tok(&self, n: u32) -> u32 {
        let mut n = n;
        for _ in 0..1_000_000 {
            let a = self.a(n);
            let Some(k) = self.kind(n) else { return a };
            match k {
                NodeKind::Primary => return self.primary_end(n),
                NodeKind::Postfix => return self.postfix_end(n),
                NodeKind::Type => {
                    if self.is(a, Term::LBracket) {
                        return self.close(a);
                    }
                    match self.first_child(n) {
                        Some(c) => n = c,
                        None => return a,
                    }
                }
                NodeKind::TypeBase => {
                    let t = self.tk(a);
                    if t == Term::LParen as u8 {
                        return self.close(a);
                    } else if t == Term::KwFn as u8 {
                        let close = self.close(a + 1);
                        if let Some(nd) = self.child(n, NodeKind::Without) {
                            return self.close(self.a(nd) + 1);
                        }
                        if let Some(nd) = self.child(n, NodeKind::Needs) {
                            return self.close(self.a(nd) + 1);
                        }
                        if self.is(close + 1, Term::MinusGt) {
                            match self.last_child(n) {
                                Some(c) => n = c,
                                None => return close,
                            }
                            continue;
                        }
                        return close;
                    } else if t == Term::Ident as u8 {
                        let e = self.chain_end(a);
                        if self.is(e + 1, Term::Lt) {
                            let (end, _) = self.angle_end(e + 1);
                            return end.saturating_sub(1);
                        }
                        return e;
                    } else if t == Term::KwSelfType as u8 {
                        return a;
                    }
                    match self.last_child(n) {
                        Some(c) => n = c,
                        None => return a,
                    }
                }
                NodeKind::MatchExpr => {
                    // `match` Expr `{` arms `}`
                    let scrut = self.first_child(n).unwrap_or(n);
                    let e = if scrut == n { a } else { self.end_tok(scrut) };
                    return self.close(e + 1);
                }
                NodeKind::Block | NodeKind::StructLit => return self.close(a),
                NodeKind::GenericArgs => return self.angle_end(a).0.saturating_sub(1),
                NodeKind::Literal => {
                    let mut e = a;
                    if self.is(e, Term::Minus) {
                        e += 1;
                    }
                    return e;
                }
                NodeKind::Pattern => {
                    let t = self.tk(a);
                    if t == Term::LParen as u8 {
                        return self.close(a);
                    }
                    if t == Term::Ident as u8 {
                        let e = self.chain_end(a);
                        if self.is(e + 1, Term::LParen) || self.is(e + 1, Term::DotLBrace) {
                            return self.close(e + 1);
                        }
                        return e;
                    }
                    if t == Term::KwMut as u8 {
                        return a + 1;
                    }
                    match self.last_child(n) {
                        Some(c) => n = c,
                        None => return a,
                    }
                }
                _ => match self.last_child(n) {
                    Some(c) => n = c,
                    None => return a,
                },
            }
        }
        self.a(n)
    }

    /// The last token of a `Primary`.
    pub fn primary_end(&self, n: u32) -> u32 {
        let a = self.a(n);
        let t = self.tk(a);
        if t == Term::LParen as u8 || t == Term::LBracket as u8 {
            return self.close(a);
        }
        if t == Term::KwFn as u8 {
            return match self.child(n, NodeKind::Block) {
                Some(b) => self.close(self.a(b)),
                None => a,
            };
        }
        if t == Term::Ident as u8 && self.is(a + 1, Term::ColonColon) {
            let (e, _) = self.angle_end(a + 2);
            return e.saturating_sub(1);
        }
        a
    }

    /// The last token of a postfix chain: a `Postfix` (its last suffix) or a bare `Primary`.
    pub fn postfix_end(&self, n: u32) -> u32 {
        if self.kind(n) == Some(NodeKind::Primary) {
            return self.primary_end(n);
        }
        let Some(prim) = self.first_child(n) else { return self.a(n) };
        let mut p = self.primary_end(prim) + 1;
        loop {
            let t = self.tk(p);
            if t == Term::Dot as u8 {
                if self.is(p + 1, Term::Ident) {
                    p += 2;
                    if self.is(p, Term::ColonColon) {
                        let (e, partial) = self.angle_end(p + 1);
                        if partial {
                            return e.saturating_sub(1);
                        }
                        p = e;
                    }
                } else if self.is(p + 1, Term::IntLit) {
                    p += 2;
                } else {
                    return p - 1;
                }
            } else if t == Term::DotLBrace as u8 || t == Term::LParen as u8 || t == Term::LBracket as u8 {
                p = self.close(p) + 1;
            } else if t == Term::Question as u8 {
                p += 1;
            } else {
                return p - 1;
            }
        }
    }

    /// The span of node `n` (first token through its end token).
    pub fn node_span(&self, n: u32) -> Span {
        self.range_span(self.a(n), self.end_tok(n))
    }

    /// The source text of node `n`.
    pub fn node_text(&self, n: u32) -> &'p str {
        self.span_text(self.node_span(n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kids_in_place_and_spilled() {
        for n in [0u32, 1, 2, Kids::INLINE as u32, Kids::INLINE as u32 + 1, 40] {
            let want: Vec<u32> = (0..n).collect();
            let k = Kids::from_rev(want.iter().rev().copied());
            assert_eq!(k.as_slice(), &want[..], "{n}");
            assert_eq!(k.len(), n as usize);
            assert_eq!(k.clone().into_iter().collect::<Vec<_>>(), want);
            let mut back: Vec<u32> = k.clone().into_iter().rev().collect();
            back.reverse();
            assert_eq!(back, want);
            assert_eq!((&k).into_iter().copied().collect::<Vec<_>>(), want);
            assert_eq!(k.into_iter().len(), n as usize);
        }
    }

    #[test]
    fn brackets_match() {
        let lexed = stave_syntax::lex(b"f(a[1], .{ b }) { }", FileId(0));
        let b = Brackets::new(&lexed.tokens, 0, lexed.tokens.len() as u32);
        // f ( a [ 1 ] , .{ b } ) { } EOF
        assert_eq!(b.close_of(1), 10);
        assert_eq!(b.close_of(3), 5);
        assert_eq!(b.close_of(7), 9);
        assert_eq!(b.close_of(11), 12);
        assert_eq!(b.close_of(0), 0);
    }
}
