# Contract: conformance test annotations (v0.4)

`stave test` runs every `tests/conformance/**.stv` file and every package directory (a directory with
a `stave.toml`) as one test. Expectations are comments in the sources. The GPU targets and stage1 use
the same annotations.

## Syntax

An annotation is a line comment starting `//~ ` (in `.stv` files) or `#~ ` (in `stave.toml`),
followed by a keyword and its arguments. Unknown keywords, or arguments of the wrong form, make the
test fail as malformed. `//~^ ` (or `#~^ `) shifts a located annotation one line up per `^`, for
lines that cannot hold a comment (e.g. E0015, where `//` would be inside the unclosed literal).

- E0003 (a literal reaching end of file) cannot be annotated in a file: any comment after the
  literal's start is inside it, and nothing follows it. It is covered by unit tests
  (`crates/stave-syntax/tests/lex.rs`).
- An empty (0-byte) `.stv` test file cannot hold an annotation: it runs as a `PARSE-ONLY`
  positive (it must lex and parse with no diagnostic; a build would need `fn main`, E0108).
- Test files are read as bytes. A file with invalid UTF-8 (E0001) is decoded lossily for the
  annotation scan (line structure is kept); the compiler gets the original bytes.

| Annotation | Meaning |
|---|---|
| `//~ ERROR E0201` | an `error` diagnostic with this code whose primary span starts on this line of this file |
| `//~ WARNING E1205` | the same for a `warning` diagnostic |
| `//~ NOLOC ERROR E1302` (or `NOLOC WARNING`) | a diagnostic with no primary span (anywhere in the file; by convention at the top) |
| `#~ ERROR E1002` (or `WARNING`), in `stave.toml` | a diagnostic whose primary span starts on this line of the manifest (E1002, E1203) |
| `//~ STDOUT <file>` | run test: stdout must equal `<file>` byte for byte (path relative to the directory of the file holding the annotation) |
| `//~ STDERR-CONTAINS "<text>"` | run test: stderr contains `<text>` (escapes `\\ \" \n \t`). Repeatable; all must match |
| `//~ EXIT <n>` | run test: the exit status is `n` (default 0) |
| `//~ TARGET <t>` | build for target `t` (`--target` value). Repeatable: the test runs once per target. Default: the host baseline |
| `//~ ARGS <flags>` | extra `stave` flags for the build, e.g. `--release`, `--compiler cpu`, `--features avx2` |
| `//~ CHECK-ONLY` | run `stave check` only (no codegen, no run) |
| `//~ PARSE-ONLY` | only lex and parse: semantic stages do not run, so only lexical and syntax diagnostics are compared. For syntax tests whose items are not (yet) semantically valid (e.g. `kernel fn`, `allocator` naming no module, unresolved names). Stage0 runs `stave-syntax` in process; there is no `stave` command for it |
| `//~ RESOURCE-LIMIT` | the program fails on a resource limit (heap, stack); `stave test` ignores it, and the cross-target differential tool (CPU vs `gpu-sm86`) compares only the panic kind and exit status |
| `//~ INTERP-REPORT <class>` | run test of a program with unsafe code of its own: run in the reference interpreter with borrow tracking ([cli.md](cli.md) "Borrow tracking"), it ends with a report of this class (`use-after-free`, `write-under-shared`, ...) instead of its run expectations; `stave test` ignores it. A program whose own packages have no unsafe site must run clean there, so the annotation on it is a failure of the interpreter gate |

Package tests: located annotations may appear in any file of the package; the directives
(`STDOUT STDERR-CONTAINS EXIT TARGET ARGS CHECK-ONLY PARSE-ONLY RESOURCE-LIMIT INTERP-REPORT`) only in the root
module (`src/main.stv` or `src/lib.stv`). A package directory with `expected.stdout` and/or `expected.exit` (one decimal
number) gets the same meaning as `STDOUT expected.stdout` / `EXIT n`.

## Test kinds and pass rules

- **Negative**: at least one `ERROR` annotation. The test builds (`stave build`, `stave check` under
  `CHECK-ONLY`, lex + parse only under `PARSE-ONLY`, whose status is 1 iff there is an error
  diagnostic) with `--diagnostics json`, expects exit 1, and passes iff the multiset of
  (severity, code, file, line) of located diagnostics, plus (severity, code) of locationless ones,
  equals the annotated multiset exactly. `note` diagnostics are ignored.
- **Positive**: no `ERROR` annotation. The build must exit 0 and emit exactly the annotated
  `WARNING`s (none if unannotated). With `STDOUT`, `STDERR-CONTAINS`, `EXIT`, or `expected.*`
  files it is a **run test**: the binary runs and every stated expectation must hold.
- A target the host cannot run (without `--runner`) is E1001 for run tests: reported as "not run",
  never "passed". Check-only and negative tests are not affected.
- Every code in an annotation must exist in the registry: the union of
  [diagnostic-codes.md](diagnostic-codes.md), [gpu-diagnostics.md](gpu-diagnostics.md) (E11xx),
  and [gpu-program-diagnostics.md](gpu-program-diagnostics.md) (E12xx). Otherwise the test is
  malformed.
- `stave test` reports the counts of negative and positive tests.

## Runner conventions

- **Discovery**: below the suite root, files and directories whose name starts with `.` or `_`
  are skipped (`_selftest/` holds the runner's own fixtures). A directory with a `stave.toml` is one
  package test; discovery does not descend into it (its subdirectories are its path
  dependencies). Tests run in name order (path relative to the root, `/`-separated).
- **Files of a package test**: every `.stv` file and every `stave.toml` below the package
  directory, hidden names excluded. A subdirectory with its own `stave.toml` is another package (a
  path dependency): the test reads its files only for located `ERROR` and `WARNING` annotations,
  the diagnostics this build reports in them. Its directives (`CHECK-ONLY` of its own library
  root, ..) and malformed annotations belong to the test of that package alone, and so do its
  `test fn`s.
- **Paths**: a diagnostic's `file` is compared relative to the package directory (package test)
  or to the directory holding the single file. A `STDOUT <file>` path is relative to the directory
  of the file holding the annotation; an absolute path is malformed.
- **Malformed** (the test fails as malformed, never runs): an unknown keyword or wrong-form
  argument; trailing text after an annotation; a non-repeatable directive given twice (`TARGET`
  and `STDERR-CONTAINS` repeat; the same `TARGET` twice is malformed); both `expected.stdout` and
  `STDOUT`, or both `expected.exit` and `EXIT`; an `expected.exit` that is not one decimal number
  0..=255; a negative test with run expectations; `INTERP-REPORT` on a test that is not a positive
  run test; `CHECK-ONLY` or `PARSE-ONLY` with run
  expectations; both `CHECK-ONLY` and `PARSE-ONLY`; a directive outside the root module of a
  package (a nested package's files do not count); a code not in the registry.
- **Not run** (E1001): never fails the suite and is never counted as passed. The suite passes iff
  no test failed and none is malformed; the summary lists passed, failed, malformed, and not-run
  counts.
- `RESOURCE-LIMIT` has no effect in `stave test` (it is read only by the cross-target differential tool).
  `INTERP-REPORT` neither (it is read only by the interpreter gate).

Directory conventions of other suites (e.g. the GPU kernel suite's `run-nodev/`, which sets
`CUDA_VISIBLE_DEVICES=""`) add environment, not annotations.
