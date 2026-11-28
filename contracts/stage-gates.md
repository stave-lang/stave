# Contract: stage dumps and equivalence gates (stage0 ↔ stage1)

Both compilers implement `stave build --dump=STAGE[,STAGE...] --dump-dir DIR`. Each stage writes one
file, `DIR/STAGE.txt`. Line-oriented UTF-8, `\n`-terminated, no trailing spaces. Records are listed
in the **order given here**, which never depends on threads or scheduling. CI
compares stage1 against stage0 with `cmp`.

Common fields: `F` = file index (rank of (package, path)), `O` = byte offset in the file,
`S` = string id (rank of (hash, bytes)), `D` = decl id, `T` = type id, `N` = AST node index
(per item, postorder), `I` = item index (rank of (F, start offset)), `L` = loan id (rank of
(fn decl id, creation offset)).

| STAGE | One line per | Format | Order |
|---|---|---|---|
| `tokens` | token | `F O KIND` | F, O |
| `items` | item | `I F O_start O_end ok\|dropped` | I |
| `ast` | named node | `I N KIND a b`: `a` = file token index of the node's first token (for a node that matched no token: the next token); `b` = number of nodes in its subtree excluding itself | I, N |
| `strings` | interned string | `S hash32 "escaped-bytes"`: hash32 = xxHash32 (seed 0) of the bytes, 8 lowercase hex digits | S |
| `names` | resolving Ident use | `I N -> decl D` or `I N -> local I N2` | I, N |
| `sigs` | decl | `d<D> KIND MODULE PARENT NAME I N INDEX FLAGS SIG` and, once, the type table `T key` of the types the decl lines name | T table, then D |
| `kinds` | `sigs` table type, impl, (type, trait) with a candidate impl | `t<T> FLAGS depth D SIZE`, `impl d<D> trait .. head .. gen N methods[..]`, `apply t<T> d<D> [d<D> ..]` | T, then (trait, head, I), then (T, trait D) |
| `types` | expression node | `I N T` and, once, the type table `T key` | T table, then I, N |
| `mir` | body, then its types, locals, loans, blocks, statements, loops, drops, findings | `body d<D>[.<k>] I N STATE`, then the body's lines (below) | D, k |
| `borrow` | access | `I N access-kind verdict loan` | I, N, access-kind |
| `lir` | IR text line | the IR text at level `lowered` ([checked-ir.md](checked-ir.md): after lowering and the IR verifier, before the removal of proven checks) | section order (checked-ir.md "Sections"), then instance id, block, inst |
| `ir` | IR text line | the IR text at level `checked` (checked-ir.md: after the removal of the checks that have a witness, before the release passes) | as `lir` |
| `opt` | IR text line | the IR text at level `opt` (checked-ir.md: after the release passes; release builds only) | as `lir` |
| `diags` | diagnostic | the JSON line of `diagnostics.schema.json` | canonical diagnostic order (`diagnostics.schema.json`) |
| `stats` | stage | `STAGE ms items rounds fallbacks peak_bytes` (informational, never compared) | pipeline order |

**Field rules** (both compilers):
- `tokens` `KIND`: a named terminal of grammar.ebnf by its name (`Ident`, `IntLit`, `FloatLit`,
  `StrLit`, `FmtLit`, `CharLit`); a keyword or punctuation terminal by its text (`fn`, `.{`, `as%`,
  `...`); a lexical error token is `ERROR`. An f-string is one `FmtLit` token. Comments,
  whitespace, and EOF have no line. A file with E0001 or E0002 lists only the tokens before the
  stop point.
- `items` `O_end` = the byte offset just after the item's last token. The token run before the
  first item start is an item, always `dropped`. A file with E0001, E0002, E0005, or E0016 has no
  items.
- `diags`: stage1 writes the diagnostics of the stages it has, in stage0's lines. Its gate compares
  them with stage0's lines of the codes those stages report (`Stage::of_code`, driver
  `pipeline.rs`); the exit status (1 with any diagnostic, else 0) is compared where all of stage0's
  diagnostics are of those codes. E0005 counts only as the bracket depth form (the item cut); its
  expression and type forms come from later stages.
- `ast` `KIND`: the grammar rule name of the node. A node of a pass-through rule (the expression
  levels `Expr` .. `Postfix`, grammar.ebnf header) with exactly one child node and no token of its
  own does not exist: its child takes its place, in `N` numbering and in every other dump.
- `strings` escaping: printed between double quotes. Bytes 0x20–0x7E print as themselves, except
  `\` → `\\` and `"` → `\"`. 0x0A → `\n`, 0x09 → `\t`. Every other byte → `\xHH` (2 lowercase hex
  digits).
- `names`: one line for every Ident use that name resolution resolves: locals, generic params,
  items, modules, variants, and dotted-chain steps up to the first value. Exactly:
  - `-> decl D` for a use resolved to a declaration (item, member, module, or the error decl of a
    name declared only by a dropped item or bound by a failed import); `-> local I N2` for a local
    binding **and for a generic parameter** (N2 = its `GenericParam` node).
  - N2 of a local is its own binding node: `Param`, `SelfParam`, `Pattern` (`x`, `mut x`),
    `FieldPat` (shorthand `S.{ a }`), or `TailBind`. Each binding node binds one name, so
    `tail masked (v, m)` gives `v` and `m` two different `TailBind` nodes (grammar v0.6).
  - The use's node N: the chain's `Primary` (expressions; also every use in an f-string hole,
    keyed by the `Primary` of the `FmtLit`, one line per hole identifier that resolves), the
    `TypeBase` (types), `BoundItem`, `Pattern`, `FieldPat`, `Name` (dispatch arms), `Origin`
    (`from x`), the `Import` item root, or the `AllocatorDecl`.
  - Import and allocator paths: a line per segment that names a module. A first segment that
    is a package name (the own package or a dependency) names that package's root module
    (`src/main.stv`, `src/lib.stv`) and has no line when there is none (built-in packages); a
    segment that names only a directory has no line.
  - No line: `self` and `Self` (keywords), built-in type names (`u32`, `str`, `never`, `f32x8`)
    and built-in fns (`size_of`, `align_of`), field and method names (resolved by type checking),
    unresolved uses (E0101, E0107), and uses inside dropped items.
  - Lines with the same (I, N) are ordered by the identifier's byte offset.
- `sigs`: the decl table and every signature fact declaration collection and signature building
  decide, never a body fact. First the type table, then one line per decl (every D, module and
  error decls included), by D. Every T prints as `t<T>`, every D as `d<D>`.
  - Type table: every T a decl line names plus every T their keys reference, transitively, as
    `t<T> key`, key as in `types`. T here is the rank among these types in the canonical type-key
    order (a total order on keys, so the rank of a subset is defined), dense from `t0`; a
    key both tables hold may have another T in each. The error type has no line.
  - `KIND`: `module fn struct enum trait const static alias impl field variant method
    trait-method error`. `MODULE`: the module decl of the declaring module (a module: itself).
    `PARENT`: the decl of a member's struct, enum, variant (record fields), impl, or trait, else
    `-`. `NAME`: S, or `-` (impl). `I N`: the item and the decl's node in it (`FnDecl`,
    `AdtDecl`, `Field`, `ImplDecl`, ...), or `- -` (modules, error decls). `INDEX`: the order
    among its siblings (field, variant, method), else 0. `FLAGS`: the flags it has, joined by
    `,` in this order, or `-`: `pub unsafe view test kernel dispatch extern-import extern-export
    trait-impl tuple record has-self mut rec`.
  - `SIG` by kind, each part after one space:
    - module: `file F path[S ..]` (its module path).
    - fn, method, trait-method: `gen[GP, ..] params[TY ..] ret TY self -|value|ref|ref-mut
      abi stave|c|intrinsic`, then ` unsafe` (written `unsafe`, or an extern import) and
      ` variadic` when they hold, then `needs[FEATURE ..] deny[EFFECT ..] from[RO, ..]
      pfrom[PO, ..]`. `gen`: its own generic params; a method's impl or trait params come first
      in the flat numbering (`param i`) and print on the parent's line. `params`: the receiver
      first. `ret`: `tuple` (unit) without `->`. `needs`: the features as `fntype` prints them
      ([checked-ir.md](checked-ir.md)), empty after E0709. `deny`: the `without` effects in the
      order `alloc trap rec lock unsafe io block park`. `RO`: a return `from` entry, in
      source order, `p` or `copy p` (p = the param's index in `params`; a name that is no param
      has no entry). `PO`: a parameter `from` entry, `a<b` or `a<copy b` (`a: &mut V from b`, a
      and b param indices), sorted by (a, b, copy), unique; an entry with E0514, E0515, E0520,
      or E0521, or on a param whose type has an error, has none.
    - struct, enum, trait: `gen[GP, ..]`. impl: `gen[GP, ..] self TY trait d<D> GARG ..` (no
      args: `trait d<D>`), an inherent impl or an unresolved trait `trait -`. const, static,
      field: `ty TY`. alias: `gen[GP, ..] target TY`. variant: `payload[TY ..]` for a tuple
      variant, nothing for a unit or record variant (its fields are field decls). error: nothing.
    - `GP`: `S N`, the name and the `GenericParam` node in the decl's item, then ` const TY` (a
      const param: its declared type), ` view` (`T: view`), and ` bound d<D> GARG ..` per trait
      bound, in source order. `TY`: `t<T>`, or `-` for the error type. `GARG`: a `TY`, or a
      const as in a key (value, `constparam i`, `_`).
  - A use of a type alias prints its target type. Example: `fn first(&self) -> &T from self` in
    `impl<T: Area, const N: usize> Buf<T, N>` (d55, item 12) prints `d56 method d0 d55 20 12 30
    0 pub,has-self gen[] params[t13] ret t16 self ref abi stave needs[] deny[] from[0] pfrom[]`
    with `t13 ref t33`, `t33 adt d25 t40 constparam 1`, `t16 ref t40`, `t40 param 0`.
  - Const evaluation: an array length, and a const generic argument that is no int literal or
    const param (a const item, a const-expr, `size_of` / `align_of`), is evaluated as signature
    building does, and the key holds the value: `[u8; LEN * 2]` with `LEN = 4` is `array t0 8`,
    the same T as `[u8; 8]`. A length that fails to evaluate (a diagnostic) makes the array the
    error type; a const argument that fails prints `_` in the key. Const and static values
    are not printed.
  - Errors: a written type with an error component is the error type (`-`), so no key names
    it; were one to, the component would print `t<n>`, n = the number of table lines (the
    error type ranks last). A name declared only by a dropped item is an `error` decl.
- `kinds`: the kind, depth, and layout of every type of the `sigs` table, the impl table, and
  which impls apply to those types: what the trait, kind, and layout rules decide from the
  signatures alone, never a body fact. `t<T>` is the `sigs` T (the same types, the same numbers;
  the table is not printed again), `d<D>` a decl. Every answer is taken in the empty environment:
  a generic param has no bound and is not `view`; `Self` likewise (it is `sized`, and Send and
  Sync hold for it by the structural rule over no parts). Three sections:
  - `t<T> FLAGS depth DEPTH SIZE`, one per table type, by T.
    - `FLAGS`: the ones that hold, joined by `,` in this order, or `-`: `copy` (built-in Copy or
      an `impl Copy`), `view`, `send`, `sync` (an explicit impl decides for its ADT, else
      structural), `sized` (not `str` or a slice), `flat`, `frozen` (derived), `drop-impl` (an
      ADT type whose `Drop` impl, declared in the ADT's package, applies; with params: the ADT
      has one; an impl the checker rejects, E0905 E0906, counts alike), `drop` (needs drop glue: a `drop-impl` type or a non-Copy generic param reached
      by value through fields, payloads, tuple and array elements), `lock` (a
      `std.sync.MutexGuard` reached by value through fields, payloads, tuple and array elements,
      or through the pointee of an ADT's raw-pointer field).
    - `DEPTH`: the type depth of E0005 case `type`: 1 for a type without components, else one
      above its deepest component; a struct or enum one above its deepest type argument and
      field (after substitution); a part reached through a reference, pointer, slice, or fn
      type adds nothing. `over`: past the limit (1000). `-`: the type reaches, through parts held
      by value, a type that holds itself by value.
    - `SIZE`: `size S align A` in bytes, then ` fields[O ..]` for a struct or tuple (each
      field's offset, in declaration order) or ` tag 4 variants[[O ..] ..]` for an enum (a `u32`
      tag at offset 0 whose value is the variant's `INDEX`; per variant by `INDEX`, its fields'
      offsets; no niche: every enum has the tag). Rules: ints, floats, `bool` (1), `char` (4)
      naturally aligned; `never` (0, 1); a reference or raw pointer (8, 8), (16, 8) to `str` or
      a slice; a fn value (16, 8); an array `len` elements; a vector or mask lanes × lane bytes,
      aligned to its size; a tuple, struct, or range: fields in order, each at the next offset
      aligned to it, the size rounded up to the largest alignment; an enum: tag, then each
      variant's fields from offset 4, the largest variant rounded up. No layout: the first part
      without one, in that order (variants, fields, elements), decides: `size param` (a generic
      param, `Self`, an array of const-param length), `size unsized` (`str`, a slice, a sizeless
      vector or mask), `size none` (a type that holds itself by value, one nested past the depth
      limit, an ADT without a definition); `size overflow`: the size does not fit in 64 bits.
      The same on every target (each has 8-byte pointers): the dump takes no target. A
      `size_of::<X>()` / `align_of::<X>()` in a `sigs` array length or const argument is X's `S`
      / `A` here.
  - `impl d<D> trait d<D>|- head HEAD gen N methods[d<D> ..]`, one per impl decl: its trait
    (`-`: inherent, or an unresolved trait), the head of its self type (`-`: none, as for a
    generic param, reference, tuple, array, fn type, or the error type; `d<D>`: an ADT; else the
    built-in type's spelling, `u32`, `f32x8`, `mask32x8`, `vec`), its generic param count, its
    methods in declaration order. Ordered by trait (inherent first, then by D), then head (`-`,
    ADTs by D, built-in types by spelling bytes), then item `I`.
  - `apply t<T> d<D> [d<D> ..]`, for each table type and trait (by T, then trait D) that has a
    candidate, an impl of the trait whose head is the type's or none: the candidates that apply
    (the self type matches, every bound of the impl's params holds, and no param without `view`
    gets a view), in item order. Trait queries are answered on demand, so this is the finite
    set that stands for them: impl selection takes the first applying impl whose trait arguments
    match, so for a trait without params the first listed is the impl and `[]` is none. Two
    impls that both apply overlap (E0903) and are both listed. Whether a type is Copy, Send, Sync,
    Flat, or Frozen is its flag (an impl of Flat or Frozen, E1408, is listed here and ignored
    there); the operator impls of scalars are built in, not impls.
  - Errors: the tables print as stage0 holds them. An impl whose self type is the error type has
    head `-` and applies to nothing; an unresolved trait makes the impl inherent (`trait -`).
    The impl checks over this table report E0414 (case `drop`), E0415, E0416, E0419, E0420,
    E0421, E0422, E0810, E0903, E0905, E0906, E0907, E1408 (gated through `diags`).
  - Example (`kinds_layout`): `enum Shape { Dot, Circle(u8, f64), Rect { lo: Pair, hi: u32 } }`
    with `Pair { a: u8, b: u64, c: u16 }` (t46) prints `t45 send,sync,sized,flat,frozen depth 3
    size 40 align 8 tag 4 variants[[] [4 8] [8 32]]` and `t46 copy,send,sync,sized,flat,frozen
    depth 2 size 24 align 8 fields[0 8 16]`; `impl<T: Area> Area for Wrap<T>` (d83, `Area` d66,
    `Wrap` d50) prints `impl d83 trait d66 head d50 gen 1 methods[d84]`, and `apply t42 d66
    [d83]` for `Wrap<Pair>` (Pair implements Area), `apply t41 d66 []` for `Wrap<Res>`.
- `types`: one `I N T` line for every expression node, in items not dropped. A node that denotes
  no value (a module or type name, e.g. `io` in `io.print(..)`) or has the error type prints `-` as
  T. The type table lists every T a line names plus every T their keys reference, transitively,
  ordered by T, as `T key` with `key` printed exactly as the checked-IR `TypeKey`
  ([checked-ir.md](checked-ir.md), for example `int s 32`, `ref mut t4`). T is the rank among
  these types in the canonical type-key order, dense from `t0` (as for `sigs`), and prints as
  `t<T>` in both line kinds.
- `mir`: the borrow checker's CFG (stave-borrow `cfg`) of every fn and method body with a
  signature, one section per body, as the builder made it, before any check. Every number in a
  section is the rank of a key the section's own text fixes, never an order a builder makes
  things in, so two builders with different data structures print the same bytes for the same
  CFG. A section depends only on its body.
  - Header `body d<D>[.<k>] I N STATE`. D the decl, I the item, N the `FnDecl` node; a closure
    or `spawn` body adds `.<k>`, the `k` of its IR function `@<inst>.<k>` (checked-ir.md
    "Closures": the index of its `fn` / `spawn` token among the closure bodies of the fn,
    nested ones included), and N is its `Primary` / `Stmt` node. Sections by D; a fn's nested
    bodies follow it by k. `STATE`: `ok`, `error` (a type error: not borrow checked),
    `unsupported "<why>"` (a form the builder cannot model: the program is not accepted),
    `internal "<why>"` (the builder disagrees with the type checker). Only an `ok` section
    has more lines, and only an `ok` fn has nested sections. Const and static initializers
    have no section, nor has a body whose item has no bracket table or no body block (none
    in a program without errors).
    - `internal` wins over `unsupported`. Of several reasons of one kind the first wins in
      the order the builder makes the CFG statements (evaluation order: the `requires`
      clauses and `old(e)` values at entry, then the body, the receiver first, operands left
      to right, a call's arguments before the call, the `ensures` clauses at each return).
    - `unsupported` reasons: `` `for` over a type without `next` ``, `` `for` over a type that
      is not a `for` source ``, `pattern nesting too deep`, `expression nesting too deep`
      (the type checker's nesting limit, counted as stave-borrow `cfg` counts it), `` `old`
      outside an `ensures` clause ``, and `nested body` for a fn one of whose closure or
      `spawn` bodies is unsupported.
    - `internal` reasons name the disagreement (a value without a type, a temporary or
      statement the temporaries table does not hold, a capture the capture analysis holds
      otherwise, a pattern shape stave-sema `drops` rejects); a nested body's reason is its
      fn's. The text is stage0's diagnosis of its own bug: a second compiler prints the
      state, and the gate compares an `internal` header up to the reason.
  - Numbering. Blocks `b<j>`: the preorder of a depth-first walk from the entry that takes the
    successors in order, then the blocks the entry does not reach (code after a loop without
    `break`, a join every way into which exits): the checks read their calls, loans, and
    moves too (the lock-rank scan takes every call token, the liveness every use). They are
    numbered by further walks of the same kind through the blocks not yet numbered, each from
    the block with the least key among those left that no block left leads to (among all
    left, when only cycles are left). The key of a block, taken when the walk starts: the list
    of its statement nodes, then the list of its successors' numbers (`-` for one not yet
    numbered), each compared as a list (`-` after every number, a prefix first). Two blocks
    with equal keys differ at most in what their statements do at the same nodes; of such
    the builder's first is taken (in the corpus every such pair prints alike). Locals
    `l<j>`: the
    params by index, then the captures by binding node, then every other local in the order the
    lines below first name it (blocks in order, then the `find` lines); a local no line names is
    not printed. Loans `L<j>`: in the order the block lines first name them. Naming a loan for
    the first time names, right after it, the local of its place and then its `whole` loan
    (with that loan's place local, and so on), though the block line shows neither: an
    `activate` line, for one, numbers the locals its loans borrow. Types `t<j>`: as in
    `types`, the rank in the canonical type-key order among the types the section names and
    the components of their keys, dense from `t0`; the error type and no type print `-` (as a
    key component the error type prints `t<n>`, n the number of table lines, as in `sigs`).
    Wherever a field may be `-` (no node, token, or statement), `-` sorts after every number.
  - `t<j> KEY`, by j; KEY as in `types`.
  - `local l<j> ROLE N TY STMT END MADE FLAGS`, by j. ROLE: `param<i>`, `capture`, `bind` (a
    pattern binding), `temp`, `env` (a closure literal's environment, N the literal), `join` (a
    `scope` statement's join value, N the statement), `static` (the storage of a `static` the
    body names, N `-`), `result` (the returned value the `ensures` clauses read, N the
    first `Ensures` node, the binding node of `result`), `anon` (a
    local of no node: a chunk loop's block value, a returned value copied out of a part). N: the
    binding node, a temporary's expression node, or `-`. STMT, END, MADE: a temporary's
    innermost open statement node (`-`: none), what ends it (`stmt cond guard branch arm
    clause old` of stave-sema `temps`, `loop` for a `for` iterator, `-`), and the token range
    `A:B` of the value it holds when that is not its node's (`-`); every other local prints
    `- - -`. FLAGS joined by `,` in this order, or `-`: `mut capture-mut guard matched
    static-root`.
  - `loan L<j> N PLACE WHOLE FLAGS`, by j: the creation node, the borrowed place, the loan it
    reborrows whole (`L<k>` or `-`), FLAGS `mut reserved capture cast` or `-`.
  - `block b<j> TERM`, by j, TERM `goto b<j>`, `branch b<j> ..` (successors in order),
    `return`, `unreachable`; then one line per statement, in order, two spaces then
    `N KIND ..`, N the statement's node or `-`: `assign PLACE init|set RV`, `activate l<j>
    L<reserved> L<active>`, `read OPND`, `drop PLACE`, `dead l<j>`, `return OPND`.
    PLACE: `l<j>` then per projection `.*` `.*m` `.*r` (deref of `&`, `&mut`, raw), `.f<i>`
    (field), `.v<v>:<i>` (field i of variant v), `.[]` (an element), `.c<i>` (an SoA column),
    then `:t<j>` (`:-`: no type), the place's type, when it has a projection or a type other
    than its local's: the checks read it (a drop or a write through a dereference of a type
    whose drop reaches references accesses what it reaches).
    OPND: `copy PLACE N`, `move PLACE N`, `const N` (N the operand's node or `-`). RV: `use
    OPND`, `ref PLACE L<j>`, `ref-mut PLACE L<j>`, `agg[OPND, ..]`, `pure[OPND, ..]`,
    `call[OPND, ..] tok T flows F stores[S, ..]`: T the call's file token index (as `ast`
    `a`), F `-` (no origins), `all` (every argument's), or `[i, copy i, ..]` (per named param:
    the argument index); S `a<b` / `a<copy b`, then ` view` when the target points to a view.
  - `loop b<head> b<exit> [b<j> ..]`, by (head, exit): each loop, with its exit and its
    blocks, sorted. The blocks of a loop are syntactic:
    its head and every block its condition, body, and (`for`) step make, those of the forms
    and of the exits (`return`, `?`, `break`, `continue`) inside it included; its exit block
    (where the code after the loop goes on) is not. So a block only an exit inside the loop
    reaches is one, though no path leads from it back to the head. The checks read this set:
    an edge to the head from one of its blocks starts the next iteration.
  - `drop POINT OWNER HEAD RUNS`, one per row of the body's drop schedule (stave-sema
    `drops`), sorted by (POINT, OWNER, HEAD) in the order the forms are listed here, then by
    their numbers. POINT: `end`, `return T`, `try T`, `jump T` (T the exit's file token index),
    `bind N`, `assign N`. OWNER: `local N`, `temp S A:B` (S the statement node or `-`, A:B the
    value's first and last token, `-` for none), `part i`, `old`. HEAD: `any`, `d<D>`,
    `tuple<n>`, `array`. RUNS: `always`, `never`, `flag` (the init dataflow's answer), `-` (no
    path from the entry reaches it). Then ` due` for a row with `-` that the code must decide
    all the same (its drop statement is in a block the entry reaches, as for the old value
    of some assignments), ` undue` for a row with an answer it need not decide. Then ` holder
    OWNER` when the owner whose flag a tested drop tests is not the row's (an assignment's old
    value: the local assigned).
  - `find KIND F O ..`, by (KIND, F, O), then by the line's text with every local no line
    before the `find` lines names written `l?` and no place type (of findings equal so, the
    builder's first); the locals the lines name first are numbered in this order. One line
    per finding (two equal findings print two equal lines). What the builder found and the
    checks
    report, at the start `F O` of its span: `move F O PLACE WHY` (E0504; WHY `behind`,
    `capture F O`, `drop PLACE`, `static`, `part PLACE reborrow|pattern|replace`), `guard F O
    move|assign|mut-borrow PLACE` or `guard F O try` (E0505), `temp-write F O` (E0503 case
    `temporary`).
  - `deep t<j> ..`: the dropped types whose drop may run a destructor on a value holding
    references, by j; absent when there is none.
  - Not printed, and not gated here: what only chooses or words a diagnostic, the spans of
    statements, operands, loans, and locals, the move sites and the operands that move with
    later operands, the exit spans, a non-temporary's open statement and the words naming
    the statements an exit leaves, the text of a `pattern` fix and of a written temporary
    (the E0501 / E0502 / E0503 / E0504 / E0512 case, label, and hole, and which of several
    loans or moves is named). `diags` gates them, and `borrow` the reported loan. The order
    of the drops at one point is the order of their `drop` statements in the blocks; the
    lowering's use of it is gated by `lir`. A body's `from` returns and param stores are
    signature facts (`sigs`).
- `borrow`: one line per access of the borrow checker in every body that was borrow
  checked (bodies with a type error are skipped).
  - `N` is the node of the access: the place expression for `read`, `write`, `move`; for an
    implicit `drop` or `storage-dead` at the end of a scope, the binding node of the local (a
    temporary: its expression node). A temporary that some loan borrows has a `storage-dead`
    where its statement ends and where a `break` / `continue` leaves that statement, each at
    the temporary's expression node.
  - `access-kind` is one of `read write move drop storage-dead`; on one node they are listed in
    that order.
  - `verdict` is `ok` or the diagnostic code reported for this access (`E0501`, `E0502`, `E0503`,
    `E0512`, `E0513`).
  - `loan` is the conflicting loan `L` reported (smallest creation offset), or `-`.
- `lir`, `ir`, `opt`: the gate form of the IR text (checked-ir.md "Levels and forms"): the emit
  form with the stack-section lines filtered out. Each file is absent (not written) when the build has error diagnostics; `opt.txt`
  is written only by a `--release` build (cli.md). `lir` gates the lowering alone, `ir` adds the
  bounds prover and the witness checker, `opt` the release passes stage0 has: from the same
  `ir` text, stage1's `ir opt --passes=base` text equals stage0's `opt` text. The full pipeline,
  inlining included, is gated by the gates of [opt-passes.md](opt-passes.md).
- `stats` `peak_bytes`: the stage's device-memory high-water mark from stage1's own arena counters.
  stage0 writes `-`.

**Behavior gate (valid programs)**: the executables produced by both compilers are run on
the conformance inputs (CPU language, GPU kernels, `gpu-sm86` programs) and the randomized
differential suite. stdout, stderr, files, and
exit codes must be identical. Machine code is not compared.

**Incremental gate**: after an edit, a resident rebuild equals a clean build of the same tree
in behavior, every dump above (except `stats`), and the executable's symbol table (each symbol's
name, type, binding, section, and size; addresses excluded). Executable bytes are not compared:
in-place slot patching changes the layout. stage0 uses the same rule (`stave build --watch`).

**Stage gates**: a stage1 stage passes when

1. its dump and every earlier dump equal stage0's on the conformance suites, the fuzz
   corpora, and the generated 100k, 1M, and 10M corpora;
2. its median time on the benchmark machine (warm daemon, after `stave daemon evict`) is within the stage's
   time budget on the 1M corpus;
3. 100 consecutive runs give byte-identical dumps.

Later stages are built only on passed gates.

**Third check**: stage1's own source built for x86-64, which selects the sequential `prims`
backend through a per-target dependency, must give the same dumps. A mismatch
between that build and stage1 on the GPU is a GPU-runtime or `prims` bug. A mismatch between that
build and stage0 is a stage-logic bug.
