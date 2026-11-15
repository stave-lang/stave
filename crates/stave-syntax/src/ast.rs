//! The AST: per item, a flat postorder array of named-rule nodes.
//!
//! One node per expansion of a named grammar rule (every non-helper nonterminal of grammar.ebnf;
//! helpers from `? * +` and groups make no node). Node indices restart at 0 for each item.
//! `a` is the file token index of the node's first token (for a node that matched no token: the
//! next token); `b` is the number of nodes in its subtree excluding itself, so the descendants of
//! node `n` are `n - b .. n` and its last child is `n - 1`. The parser leaves `extra` empty; later
//! stages use it for CSR side data.

use crate::generated::node_kinds::NodeKind;

/// One AST node (12 bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Node {
    /// `NodeKind` id.
    pub kind: u8,
    /// File token index of the first token (or of the next token, for a node matching none).
    pub a: u32,
    /// Subtree size excluding the node itself.
    pub b: u32,
}

impl Node {
    /// The node kind. Every node the parser makes has a valid kind; a corrupt id maps to `File`.
    pub fn kind(self) -> NodeKind {
        NodeKind::from_id(self.kind).unwrap_or(NodeKind::File)
    }
}

/// The AST of one file: the nodes of every parsed (ok) item, item after item, plus `extra`.
///
/// The nodes live in a few buffers (one per group of consecutive items parsed together), so
/// parallel parsing needs no final copy; the nodes of one item are always contiguous in one
/// buffer ([`crate::ItemInfo`] `chunk` and `nodes`). Use [`crate::ParsedFile::item_ast`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Ast {
    pub chunks: Vec<Vec<Node>>,
    pub extra: Vec<u32>,
}

impl Ast {
    /// Total number of nodes.
    pub fn node_count(&self) -> usize {
        self.chunks.iter().map(Vec::len).sum()
    }

    /// Every node, item after item.
    pub fn nodes(&self) -> impl Iterator<Item = &Node> {
        self.chunks.iter().flatten()
    }
}

/// The nodes of one item, indexed from 0 (postorder). The root is the last node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ItemAst<'a> {
    pub nodes: &'a [Node],
}

impl<'a> ItemAst<'a> {
    pub fn len(self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(self) -> bool {
        self.nodes.is_empty()
    }

    /// The root node index (`Item` or `Import`), `None` for a dropped item.
    pub fn root(self) -> Option<u32> {
        self.nodes.len().checked_sub(1).map(|n| n as u32)
    }

    /// Node `n`, if in range.
    pub fn get(self, n: u32) -> Option<Node> {
        self.nodes.get(n as usize).copied()
    }

    /// The kind of node `n`.
    pub fn kind(self, n: u32) -> Option<NodeKind> {
        self.get(n).map(Node::kind)
    }

    /// The index range of node `n`'s descendants (`n - b .. n`).
    pub fn descendants(self, n: u32) -> std::ops::Range<u32> {
        match self.get(n) {
            Some(node) => n.saturating_sub(node.b)..n,
            None => n..n,
        }
    }

    /// The direct children of `n`, last child first (stepping back over subtree sizes).
    pub fn children_rev(self, n: u32) -> ChildrenRev<'a> {
        let lo = self.descendants(n).start;
        ChildrenRev { nodes: self.nodes, next: n, lo }
    }

    /// The direct children of `n`, in source order.
    pub fn children(self, n: u32) -> Vec<u32> {
        let mut v: Vec<u32> = self.children_rev(n).collect();
        v.reverse();
        v
    }

    /// The first direct child of `n` whose kind is `kind`, in source order.
    pub fn child(self, n: u32, kind: NodeKind) -> Option<u32> {
        self.children_rev(n).filter(|&c| self.kind(c) == Some(kind)).last()
    }

    /// The parent of every node (`u32::MAX` for the root), computed in one backward pass.
    pub fn parents(self) -> Vec<u32> {
        let mut parent = vec![u32::MAX; self.nodes.len()];
        for n in (0..self.nodes.len() as u32).rev() {
            for c in self.children_rev(n) {
                parent[c as usize] = n;
            }
        }
        parent
    }
}

/// Iterator over a node's direct children, last first.
#[derive(Clone, Debug)]
pub struct ChildrenRev<'a> {
    nodes: &'a [Node],
    /// Index just above the next child.
    next: u32,
    /// Lowest descendant index.
    lo: u32,
}

impl Iterator for ChildrenRev<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        if self.next <= self.lo {
            return None;
        }
        let c = self.next - 1;
        let size = self.nodes.get(c as usize)?.b;
        self.next = c.saturating_sub(size);
        Some(c)
    }
}
