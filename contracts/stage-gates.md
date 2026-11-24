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
    order (the order of the `types` table, a total order on keys, so the rank of a subset is
    defined), dense from `t0`; it differs from the `types` T of the same key. The error type has
    no line.
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
      order `alloc trap recursive lock unsafe io block park`. `RO`: a return `from` entry, in
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
      has one), `drop` (needs drop glue: a `drop-impl` type or a non-Copy generic param reached
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
