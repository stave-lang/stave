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
| `types` | expression node | `I N T` and, once, the type table `T key` | T table, then I, N |
| `borrow` | access | `I N access-kind verdict loan` | I, N, access-kind |
| `ir` | checked IR line | the canonical checked-IR printer ([checked-ir.md](checked-ir.md); after type, borrow, and mono; before optimization) | instance id, block, inst |
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
- `types`: one `I N T` line for every expression node, in items not dropped. A node that denotes
  no value (a module or type name, e.g. `io` in `io.print(..)`) or has the error type prints `-` as
  T. The type table lists every T a line names plus every T their keys reference, transitively,
  ordered by T, as `T key` with `key` printed exactly as the checked-IR `TypeKey`
  ([checked-ir.md](checked-ir.md), for example `int s 32`, `ref mut t4`). T prints as `t<T>` in
  both line kinds.
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
- `ir`: `ir.txt` is absent (not written) when the build has error diagnostics.
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
