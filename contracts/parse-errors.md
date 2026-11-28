# Contract: parse-error codes and the E0004 report (v0.3)

At an item's first syntax error, the parser reports E0004 unless one rule below matches.
stage0 and stage1 implement exactly this table, so they pick the same code and span. The rules are
evaluated once, after the error, in the order listed; the first match wins. The item is dropped as
for E0004.

Inputs at the error, all available in both parsers:
- **top**: the symbol on top of the PDA stack: a terminal, or a nonterminal with its **origin
  rule** (helper nonterminals from `? * +` and groups record the grammar.ebnf rule they come from;
  `tools/ll1gen` emits the origin ids into both generated tables);
- **la**: the lookahead token (where the error is), **prev**: the token before it, and the item's
  token array (rules may look a fixed number of tokens around la);
- **d2**: the stack of open TypeArgs entered through the D2 override `(TypeBase, "<")` directly
  after a cast type (`as`-family), each with its `<` token. An entry is popped when its `>` is
  consumed.

| # | Code | Condition | Primary span |
|---|---|---|---|
| 1 | E0013 | top is a terminal of rule `Unsafe` (`(` or `StrLit`): `unsafe` not followed by `("..."` | the `unsafe` token |
| 2 | E0011 | d2 is not empty (the error is inside TypeArgs opened by D2 after a cast type, e.g. `x as usize < y;`) | the `<` of the innermost d2 entry |
| 3 | E0014 | (a) la is `{`, prev is an `Ident`, and the tokens after la are `}` or `Ident` followed by `:` `,` or `}`; or (b) la is `:`, prev is an `Ident`, the token before prev is `{`, and the token before that `{` is an `Ident` (the block opened by `if x == S { a: 1 }`) | the `{` |
| 4 | E0012 | la is a CmpOp (`== != < <= > >=`) and top's origin rule is `AndExpr` (the `("&&" CmpExpr)*` helper). With a CmpOp lookahead every helper below CmpExpr reduces to ε, so this state means a CmpExpr that already has its CmpOp is followed by another (`a < b < c`) | la |
| – | E0004 | otherwise | la (expected set: below) |

**Rules 5–13: Rust forms.** Evaluated after rule 4, in this order, only when la is a whole token
(not the rest of a split token). "Ident `w`" means an `Ident` token spelled `w` (`where`, `dyn`
are not keywords). "The open node" is the innermost named rule being parsed at the error.

| # | Code | Condition | Primary span |
|---|---|---|---|
| 5 | E0019 | la is `let` and prev is `if` or `while` | prev |
| 6 | E0020 | la is `else`, top is a `;` terminal pushed by rule `Stmt`, and the open node (that `Stmt`) starts with `let` | la |
| 7 | E0021 | la is `}` and top is a nonterminal whose origin rule is `ExprStmt` (its `(AssignOp Expr)?` helper: the statement has no `AssignOp` and no `;`) | the open node's (the statement's) first token |
| 8 | E0022 | la is Ident `where` and la lies before the item's first depth-0 `{` (the item header) | la |
| 9 | E0023 | (a) prev is `::` and la is not `<`; or (b) la is `::` and the token after la is not `<` | the `::` |
| 10 | E0025 | la is `mut`, the token after la is an `Ident` or `self`, and top's origin rule is `Param`, `Params`, `XParams`, or a rule holding the parameter list (`FnDecl`, `TraitFn`, `KernelDecl`, `DispatchDecl`, `ExternItem`, `ExternFn`; `Primary` when prev is `(` after `fn`: a closure) | la |
| 11 | E0026 | la is `(` (case: tuple) or `;` (case `unit`), prev is the `Ident` after `struct`, and top's origin rule is `AdtDecl` | la |
| 12 | E0027 | la is `!` and top's origin rule is `Type`, `TypeBase`, or `RetClause` (case `ret`: prev is `->`) | la |
| 13 | E0028 | (a) la is `impl` and top's origin rule is `Type` or `TypeBase`; or (b) prev is Ident `dyn` and la is an `Ident` | the `impl` / `dyn` token |

E0024 (`'a`) and E0029 (`/*`) are lexical error tokens (no E0004 in that item, see below). E0024
is case `ref` when the last byte before the lifetime, skipping spaces, tabs, and line ends, is `&`.
Each rule has a conformance test `tests/conformance/syntax/rustism_E00xx.stv`. The origin rules
named here are in `REQUIRED_RULES` of `tools/ll1gen` (generation fails if one is missing).

**Empty obligation**: an item that parses without error but
contains `unsafe` `(` StrLit where the literal is exactly `""` gets E0013 (case `empty`) at the
first such `unsafe` token, and is dropped like an item with a syntax error. This covers every
unsafe site kind (block, expression, fn, method, trait, impl), since all are rule `Unsafe`.

**Returned fn type clause**: an item that parses without error and has no empty obligation but
has a fn type (a `TypeBase` starting with `fn`) with a `Needs` or `Without` child, where the
tokens before that `fn`, skipping `&`, `&&`, `mut`, `const`, and `*`, end with `->`, gets E0030 at
its first clause keyword (hole `kw`: `needs` or `without`); of several such fn types
the one whose first clause comes first in the item. The item is dropped like one with a syntax
error. Grammar D5 reads the clause as the fn type's, but a reader can take it for the clause of
the fn (or fn type) that `->` belongs to, so the type must be parenthesized either way. Roles:
`open`, an empty span at the start of the `fn` token, and `close`, an empty span at the end of
the `)` that closes the fn type's last clause; the fix inserts `(` and `)` there, the reading D5
gives.

## E0004: expected set and found token

Both parsers print E0004 as `unexpected <found>, expected one of: <expected>` (diagnostics.toml),
with these inputs:

- **Expected set**: if top is a terminal, exactly that terminal. If top is a nonterminal, every
  terminal whose cell in top's table row is not an error, including the ε cells the generator
  filled from FOLLOW. The set is listed in terminal order (the generated terminal ids, byte order
  of the spellings) and joined by `, `. Split actions (grammar header) are not expanded:
  the set names table terminals only.
- **Found token**: la, spelled by kind (the terminal text, or the class name `Ident`, `IntLit`,
  ..., `EOF`), and after a split action the remaining part (a `>>` whose first `>` was consumed is
  reported as `>` at that `>`'s byte). An item is parsed against a virtual `EOF` at its cut: when
  the item ends early (the stack still expects more at the cut), the found token is the real next
  token, i.e. the next item's first token (an item keyword) or the file's `EOF`, at its span.
- **Tokens before the first item**: one E0004 at the first token, with the expected set
  of `File`'s row (every terminal that can start the file).
- **A late import** (an `import` after a non-import item): E0004 at the `import` token, with the
  expected set of `File`'s row minus `import`.
- An item that contains an error token (E0003, E0006, E0015, E0018, E0024, E0029) gets no E0004.

Notes:
- Rule 2 also fires for a genuine type-argument error after a cast (`x as Vec<u8 ;`). The E0011
  fix, "write `(x as T) < y`, or close the type arguments", covers both.
- Rule 3 (a) also covers patterns (`match x { S { a } => .. }`) and empty literals (`S {}`).
- Each rule has a conformance test. `tools/ll1gen --check` fails if
  the origin rules named here (`Unsafe`, `AndExpr`) or the override `(TypeBase, "<")` disappear
  from the grammar.
