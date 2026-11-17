//! The lowering tables against contracts/simd-ops.md: every row of the operation table and of the
//! chunk-loop table is mapped below to (ops, form, types); for each tier the class the contract
//! cell gives each type must be the table's, and a cell part that needs a feature (`N (+avx512vbmi)`)
//! must match the table's cell for that feature. Every table op must come from some contract row.
//!
//! Cell grammar: parts separated by `, `, each a class (N M U, `else M`, `E0717` = no class) with
//! qualifiers: lane widths (`8/16`, `8–32`, `≤32`, `16/32/64`), typed widths (`i32/i64`, `u8`:
//! by width, or by width and sign in rows marked signed; `f32`, `f64`: float lanes), `int` / `fp`, a feature in parentheses
//! (`(+f16c)`; a feature in the tier baseline counts as baseline), and notes (`(a)`, `(movmsk)`,
//! `(≈3)`, `(≤16 B)`, ...). A type takes the first baseline part whose qualifiers match it; a part
//! without type qualifiers matches every type still unmatched, and a type no part names takes the
//! last baseline part.

use std::collections::BTreeSet;

use stave_target::lowering::{self, Class, Tier};

const TIERS: [Tier; 5] = [Tier::V2, Tier::V3, Tier::V4, Tier::Neon, Tier::Sve];

const FLOAT: &[&str] = &["f32", "f64"];
const SINT: &[&str] = &["i8", "i16", "i32", "i64"];
const UINT: &[&str] = &["u8", "u16", "u32", "u64"];
const INT: &[&str] = &["i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64"];
const ANY: &[&str] = &["i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f16", "bf16", "f32", "f64"];
const ANY8_16: &[&str] = &["i8", "u8", "i16", "u16", "f16", "bf16"];
const ANY32_64: &[&str] = &["i32", "u32", "f32", "i64", "u64", "f64"];
const MASK: &[&str] = &["m8", "m16", "m32", "m64"];

/// One contract row: its first cell, then (ops, form, types) groups, and whether typed qualifiers
/// (`u32/i64`) name a sign.
struct Row {
    label: &'static str,
    groups: &'static [(&'static [&'static str], &'static str, &'static [&'static str])],
    signed: bool,
}

const fn row(
    label: &'static str,
    groups: &'static [(&'static [&'static str], &'static str, &'static [&'static str])],
) -> Row {
    Row { label, groups, signed: false }
}

/// A row whose typed qualifiers (`u32/i64`) name a sign.
const fn signed_row(
    label: &'static str,
    groups: &'static [(&'static [&'static str], &'static str, &'static [&'static str])],
) -> Row {
    Row { label, groups, signed: true }
}

const WIDE16: &[&str] = &["i16", "u16", "i32", "u32", "i64", "u64"];
const WIDEN1: &[&str] =
    &["i8>i16", "i16>i32", "i32>i64", "u8>u16", "u16>u32", "u32>u64", "u8>i16", "u16>i32", "u32>i64"];
const NARROW1: &[&str] = &[
    "i16>i8", "i32>i16", "i64>i32", "u16>u8", "u32>u16", "u64>u32", "i16>u8", "i32>u16", "i64>u32", "u16>i8",
    "u32>i16", "u64>i32",
];

const OPS: &[Row] = &[
    row("float add/sub/mul/div", &[(&["vec.add", "vec.sub", "vec.mul", "vec.div"], "", FLOAT)]),
    row("float sqrt", &[(&["vec.sqrt"], "", FLOAT)]),
    row("float neg/abs", &[(&["vec.neg", "vec.abs"], "", FLOAT)]),
    row("float copysign", &[(&["vec.copysign"], "", FLOAT)]),
    row("float floor/ceil/round/trunc", &[(&["vec.floor", "vec.ceil", "vec.round", "vec.trunc"], "", FLOAT)]),
    row("int add/sub wrapping", &[(&["vec.add.wrap", "vec.sub.wrap"], "", INT)]),
    row("int neg wrapping / abs", &[(&["vec.abs_wrap"], "", SINT)]),
    row("int mul wrapping i16/i32", &[(&["vec.mul.wrap"], "", &["i16", "u16", "i32", "u32"])]),
    row("int mul wrapping i8", &[(&["vec.mul.wrap"], "", &["i8", "u8"])]),
    row("int mul wrapping i64", &[(&["vec.mul.wrap"], "", &["i64", "u64"])]),
    row("int mul saturating", &[(&["vec.mul.sat"], "", INT)]),
    row("int mul high half", &[(&["vec.mul_hi"], "", INT)]),
    row("sat add/sub i8/i16", &[(&["vec.add.sat", "vec.sub.sat"], "", &["i8", "u8", "i16", "u16"])]),
    row("sat add i32/i64", &[(&["vec.add.sat"], "", &["i32", "u32", "i64", "u64"])]),
    signed_row("sat sub i32/i64", &[(&["vec.sub.sat"], "", &["i32", "u32", "i64", "u64"])]),
    row("checked add/sub/mul", &[(&["vec.checked_add", "vec.checked_sub", "vec.checked_mul"], "", INT)]),
    row(
        "bitwise and/or/xor/not, andnot",
        &[(&["vec.and", "vec.or", "vec.xor", "vec.not", "vec.andnot"], "", INT)],
    ),
    row("shift left by const amount", &[(&["vec.shl", "vec.shl.wrap"], "const", INT)]),
    signed_row("shift right by const amount", &[(&["vec.shr"], "const", INT)]),
    row("shift left by vector amount (16/32/64)", &[(&["vec.shl", "vec.shl.wrap"], "vector", WIDE16)]),
    signed_row("shift right by vector amount (16/32/64)", &[(&["vec.shr"], "vector", WIDE16)]),
    row("shift by vector amount (8)", &[(&["vec.shl", "vec.shl.wrap", "vec.shr"], "vector", &["i8", "u8"])]),
    row("int div by const splat", &[(&["vec.div_const"], "", INT)]),
    row("int div by vector", &[(&["vec.div_lanes"], "", INT)]),
    row("fma", &[(&["vec.fma"], "", FLOAT)]),
    row("min/max (IEEE minimum, NaN propagates)", &[(&["vec.min", "vec.max"], "", FLOAT)]),
    row("min_num/max_num (NaN = missing)", &[(&["vec.min_num", "vec.max_num"], "", FLOAT)]),
    row("min/max x86 semantics (named, order-dependent)", &[(&["vec.min_x86", "vec.max_x86"], "", FLOAT)]),
    row("int min/max", &[(&["vec.min", "vec.max"], "", INT)]),
    row(
        "compare → mask, signed and float",
        &[
            (&["vec.eq", "vec.ne", "vec.lt", "vec.le", "vec.gt", "vec.ge"], "", SINT),
            (&["vec.eq", "vec.ne", "vec.lt", "vec.le", "vec.gt", "vec.ge"], "", FLOAT),
            (&["vec.eq", "vec.ne"], "", UINT),
        ],
    ),
    row("compare → mask, unsigned", &[(&["vec.lt", "vec.le", "vec.gt", "vec.ge"], "", UINT)]),
    row("mask and/or/xor/not", &[(&["mask.and", "mask.or", "mask.xor", "mask.not"], "", MASK)]),
    row("mask constants", &[(&["mask.all", "mask.none"], "", MASK)]),
    row("mask to vector", &[(&["mask.to_vec"], "", MASK)]),
    row("mask rewiden", &[(&["mask.rewiden"], "", MASK)]),
    row("mask to_bits", &[(&["mask.to_bits"], "", MASK)]),
    row("select", &[(&["mask.select"], "", ANY)]),
    row("load/store unaligned", &[(&["vec.load", "vec.store"], "", ANY)]),
    row("load/store aligned", &[(&["vec.load_aligned", "vec.store_aligned"], "", ANY)]),
    row("masked load/store 32/64-bit", &[(&["vec.load_masked", "vec.store_masked"], "", ANY32_64)]),
    row("masked load/store 8/16-bit", &[(&["vec.load_masked", "vec.store_masked"], "", ANY8_16)]),
    row("first-fault load", &[(&["vec.load_first_fault"], "", ANY)]),
    row("gather 32/64", &[(&["vec.gather"], "", ANY32_64)]),
    row("scatter 32/64", &[(&["vec.scatter"], "", ANY32_64)]),
    row("compress 32/64", &[(&["vec.compress"], "", ANY32_64)]),
    row("compress 8/16", &[(&["vec.compress"], "", ANY8_16)]),
    row("expand", &[(&["vec.expand"], "", ANY)]),
    row("float reduce add (fixed tree)", &[(&["vec.reduce_add"], "", FLOAT)]),
    row("int reduce add (wrapping)", &[(&["vec.reduce_add_wrap"], "", INT)]),
    row("reduce and/or", &[(&["vec.reduce_and", "vec.reduce_or"], "", INT)]),
    row(
        "reduce min/max (fixed tree)",
        &[
            (&["vec.reduce_min", "vec.reduce_max"], "", INT),
            (&["vec.reduce_min", "vec.reduce_max"], "", FLOAT),
        ],
    ),
    row("reduce add ordered (float)", &[(&["vec.reduce_add_ordered"], "", FLOAT)]),
    row("prefix scan add", &[(&["vec.scan_add_wrap"], "", INT), (&["vec.scan_add"], "", FLOAT)]),
    row("permute (const indices)", &[(&["vec.permute"], "", ANY)]),
    row("table lookup bytes (out-of-range → 0)", &[(&["vec.lookup"], "", &["i8", "u8"])]),
    row("popcount", &[(&["vec.popcount"], "", INT)]),
    row("leading zeros", &[(&["vec.clz"], "", INT)]),
    row(
        "widen",
        &[(&["vec.widen_lo", "vec.widen_hi"], "", &["i8", "i16", "i32", "u8", "u16", "u32", "f32"])],
    ),
    row("narrow truncating", &[(&["vec.narrow_wrap"], "", &["i16", "i32", "i64", "u16", "u32", "u64"])]),
    signed_row(
        "narrow saturating",
        &[(&["vec.narrow_sat"], "", &["i16", "i32", "i64", "u16", "u32", "u64"])],
    ),
    row("int widen one step", &[(&["vec.cast", "vec.cast.wrap"], "", WIDEN1)]),
    row("int narrow one step, wrapping", &[(&["vec.cast.wrap"], "", NARROW1)]),
    signed_row(
        "int → float exact (e.g. i32 → f64)",
        &[(&["vec.cast"], "", &["i8>f32", "u8>f32", "i16>f32", "u16>f32", "i32>f64", "u32>f64"])],
    ),
    signed_row(
        "int → float rounding (i32 → f32, i64 → f64)",
        &[(&["vec.cast.round"], "", &["i32>f32", "u32>f32", "i64>f64", "u64>f64"])],
    ),
    row(
        "f32 ↔ f64, float to its own type",
        &[(&["vec.cast"], "", &["f32>f64"]), (&["vec.cast.round"], "", &["f64>f32", "f32>f32", "f64>f64"])],
    ),
    signed_row(
        "float → int saturating",
        &[(&["vec.cast.sat"], "", &["f32>i32", "f32>u32", "f64>i64", "f64>u64"])],
    ),
    row("float → int checked", &[(&["vec.cast.try"], "", &["f32>i32", "f32>u32", "f64>i64", "f64>u64"])]),
    row("f16 ↔ f32 convert", &[(&["vec.cast"], "", &["f16>f32"]), (&["vec.cast.round"], "", &["f32>f16"])]),
    row("splat", &[(&["vec.splat"], "", ANY)]),
    row("bitcast / split / join", &[(&["vec.bitcast", "vec.split", "vec.join"], "", ANY)]),
    row("tail mask", &[(&["mask.first_n"], "", MASK)]),
];

const LOOPS: &[Row] = &[
    row("`for v: TxN in s` body (full chunks)", &[(&["loop.chunk"], "", ANY)]),
    row("`tail masked (v, m)`, 32/64-bit lanes", &[(&["loop.tail_masked"], "", ANY32_64)]),
    row("`tail masked (v, m)`, 8/16-bit lanes", &[(&["loop.tail_masked"], "", ANY8_16)]),
    row(
        "`tail masked` with a `&mut` binding: write-back of active lanes only",
        &[(&["loop.tail_masked_mut"], "", ANY)],
    ),
    row("`tail scalar (x)`", &[(&["loop.tail_scalar"], "", ANY)]),
    row("`tail exact;`", &[(&["loop.tail_exact"], "", ANY)]),
];

/// Memory-hint ops: the contract says class N everywhere; their rows are checked by name.
const HINTS: &[(&str, &str)] = &[
    ("`prefetch_read_l1`", "mem.prefetch_read_l1"),
    ("`prefetch_read_l2`", "mem.prefetch_read_l2"),
    ("`prefetch_read_l3`", "mem.prefetch_read_l3"),
    ("`prefetch_read_nta`", "mem.prefetch_read_nta"),
    ("`prefetch_write_l1`", "mem.prefetch_write_l1"),
    ("`prefetch_write_l2`", "mem.prefetch_write_l2"),
    ("`prefetch_write_l3`", "mem.prefetch_write_l3"),
    ("`prefetch_write_nta`", "mem.prefetch_write_nta"),
    ("`v.store_stream(s, i)` float lanes", "vec.store_stream"),
    ("`v.store_stream(s, i)` int lanes", "vec.store_stream"),
    ("`x.store_stream(s, i)` 32/64-bit int", "mem.store_stream"),
    ("`x.store_stream(s, i)` f32/f64", "mem.store_stream"),
    ("`x.store_stream(s, i)` 8/16-bit", "mem.store_stream"),
    ("`fence_stream()`", "mem.fence_stream"),
];

fn contract() -> String {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../contracts/simd-ops.md");
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

/// Splits a table line into trimmed cells; a `|` inside backticks is part of its cell.
fn cells(line: &str) -> Vec<String> {
    let inner = line.trim().trim_start_matches('|').trim_end_matches('|');
    let mut out = vec![String::new()];
    let mut code = false;
    for c in inner.chars() {
        match c {
            '`' => {
                code = !code;
                out.last_mut().unwrap().push(c);
            }
            '|' if !code => out.push(String::new()),
            _ => out.last_mut().unwrap().push(c),
        }
    }
    out.iter().map(|s| s.trim().to_string()).collect()
}

/// The body rows of the table that follows `heading`, as cells.
fn table(text: &str, heading: &str) -> Vec<Vec<String>> {
    let start = text.find(heading).unwrap_or_else(|| panic!("no `{heading}` in simd-ops.md"));
    text[start..]
        .lines()
        .skip_while(|l| !l.starts_with('|'))
        .take_while(|l| l.starts_with('|'))
        .skip(2)
        .map(cells)
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Want {
    Class(Class),
    /// `E0717`: the contract gives no class.
    Any,
}

/// What a type is, for qualifier matching: lane bits, float, signed (the integer side of a cast).
struct TypeInfo {
    bits: u32,
    float: bool,
    signed: bool,
}

fn info(key: &str) -> TypeInfo {
    let lane = match key.split_once('>') {
        Some((s, d)) if s.starts_with('f') || s.starts_with("bf") => {
            if d.starts_with('f') || d.starts_with("bf") { s } else { d }
        }
        Some((s, _)) => s,
        None => key,
    };
    let bits = lane.trim_start_matches(|c: char| c.is_ascii_alphabetic()).parse().unwrap();
    TypeInfo { bits, float: lane.starts_with('f') || lane.starts_with("bf"), signed: lane.starts_with('i') }
}

type Test = Box<dyn Fn(&TypeInfo) -> bool>;

/// One qualifier test; `None` for a note.
fn qualifier(word: &str, signed_row: bool) -> Option<Test> {
    let w = word.trim_matches(|c| c == '(' || c == ')' || c == ',');
    if w == "int" {
        return Some(Box::new(|t| !t.float));
    }
    if w == "fp" {
        return Some(Box::new(|t| t.float));
    }
    if let Some(n) = w.strip_prefix('≤').and_then(|n| n.parse::<u32>().ok()) {
        return Some(Box::new(move |t| t.bits <= n));
    }
    if let Some((a, b)) = w.split_once('–')
        && let (Ok(a), Ok(b)) = (a.parse::<u32>(), b.parse::<u32>())
    {
        return Some(Box::new(move |t| (a..=b).contains(&t.bits)));
    }
    let parts: Vec<&str> = w.split('/').collect();
    if parts.iter().all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit())) {
        let ws: Vec<u32> = parts.iter().map(|p| p.parse().unwrap()).collect();
        return Some(Box::new(move |t| ws.contains(&t.bits)));
    }
    let typed = |p: &&str| {
        (p.starts_with('i') || p.starts_with('u') || p.starts_with('f'))
            && p.len() > 1
            && p[1..].bytes().all(|b| b.is_ascii_digit())
    };
    if parts.iter().all(typed) {
        let ts: Vec<(u8, u32)> = parts.iter().map(|p| (p.as_bytes()[0], p[1..].parse().unwrap())).collect();
        return Some(Box::new(move |t| {
            ts.iter().any(|&(c, b)| {
                b == t.bits
                    && match c {
                        b'f' => t.float,
                        _ => !t.float && (!signed_row || (c == b'i') == t.signed),
                    }
            })
        }));
    }
    None
}

struct Part {
    want: Want,
    /// No type qualifier: matches every type.
    any_type: bool,
    test: Test,
    feature: Option<String>,
}

fn parse_cell(cell: &str, signed_row: bool) -> Vec<Part> {
    let mut parts = Vec::new();
    for raw in cell.split(", ") {
        let mut words = raw.split_whitespace().peekable();
        if words.peek() == Some(&"else") {
            words.next();
        }
        let want = match words.next() {
            Some("N") => Want::Class(Class::N),
            Some("M") => Want::Class(Class::M),
            Some("U") => Want::Class(Class::U),
            Some("E0717") => Want::Any,
            other => panic!("cell `{cell}`: no class in `{raw}` ({other:?})"),
        };
        let mut tests = Vec::new();
        let mut feature = None;
        let mut paren = String::new();
        for w in words {
            if !paren.is_empty() || w.starts_with('(') {
                paren.push_str(w);
                paren.push(' ');
                if !w.ends_with(')') {
                    continue;
                }
                let inner = paren.trim().trim_start_matches('(').trim_end_matches(')').to_string();
                paren.clear();
                if let Some(f) = inner.strip_prefix('+') {
                    feature = Some(f.to_string());
                } else if let Some(t) = qualifier(&inner, signed_row) {
                    tests.push(t);
                }
            } else if let Some(t) = qualifier(w, signed_row) {
                tests.push(t);
            } else if w != "wider" {
                panic!("cell `{cell}`: unknown qualifier `{w}`");
            }
        }
        let any_type = tests.is_empty();
        parts.push(Part { want, any_type, test: Box::new(move |t| tests.iter().any(|f| f(t))), feature });
    }
    parts
}

/// The contract class of `key` for the baseline parts.
fn baseline_want(parts: &[&Part], key: &str) -> Want {
    let t = info(key);
    parts
        .iter()
        .find(|p| !p.any_type && (p.test)(&t))
        .or_else(|| parts.iter().find(|p| p.any_type))
        .or(parts.last())
        .map(|p| p.want)
        .unwrap_or_else(|| panic!("no baseline part for {key}"))
}

fn check_rows(kind: &str, rows: &[Vec<String>], map: &[Row], first_tier: usize, failures: &mut Vec<String>) {
    let labels: BTreeSet<&str> = rows.iter().map(|r| r[0].as_str()).collect();
    for r in map {
        assert!(labels.contains(r.label), "{kind}: no contract row `{}` (update the mapping)", r.label);
    }
    for cells in rows {
        let label = cells[0].as_str();
        let r =
            map.iter().find(|r| r.label == label).unwrap_or_else(|| panic!("{kind}: unmapped row `{label}`"));
        for (ti, &tier) in TIERS.iter().enumerate() {
            let cell = &cells[first_tier + ti];
            let parts = parse_cell(cell, r.signed);
            let in_base = |f: &str| {
                stave_target::Feature::from_name(f).is_some_and(|f| tier.triple().baseline().contains(f))
            };
            let base: Vec<&Part> =
                parts.iter().filter(|p| p.feature.as_deref().is_none_or(in_base)).collect();
            for &(ops, form, keys) in r.groups {
                for op in ops {
                    for key in keys {
                        let Some(got) = lowering::lookup(op, key, form, tier) else {
                            failures.push(format!("{label}: {op} {key} {form}: no {} entry", tier.name()));
                            continue;
                        };
                        let want = baseline_want(&base, key);
                        if want != Want::Any && want != Want::Class(got.class) {
                            failures.push(format!(
                                "{label}: {op} {key} {form} on {}: contract `{cell}`, table {}",
                                tier.name(),
                                got.class.name()
                            ));
                        }
                        for p in parts.iter().filter(|p| p.feature.as_deref().is_some_and(|f| !in_base(f))) {
                            if !p.any_type && !(p.test)(&info(key)) {
                                continue;
                            }
                            let f = p.feature.as_deref().unwrap();
                            let feat = stave_target::Feature::from_name(f)
                                .unwrap_or_else(|| panic!("{label}: unknown feature {f}"));
                            let with = tier.triple().baseline().with(feat);
                            let got = lowering::lookup_with(op, key, form, tier, with).unwrap();
                            if Want::Class(got.class) != p.want || got.feature != Some(f) {
                                failures.push(format!(
                                    "{label}: {op} {key} on {}+{f}: contract `{cell}`, table {} ({:?})",
                                    tier.name(),
                                    got.class.name(),
                                    got.feature
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn operation_and_chunk_loop_tables_match_the_contract() {
    let text = contract();
    let mut failures = Vec::new();
    let ops = table(&text, "## Operation table");
    assert!(ops.len() >= OPS.len(), "operation table: {} rows", ops.len());
    check_rows("operation table", &ops, OPS, 2, &mut failures);
    let loops = table(&text, "## Chunk loops and tails");
    check_rows("chunk-loop table", &loops, LOOPS, 1, &mut failures);
    assert!(failures.is_empty(), "{} mismatches:\n{}", failures.len(), failures.join("\n"));
}

#[test]
fn memory_hints_are_native_everywhere() {
    let text = contract();
    let rows = table(&text, "## Memory hints");
    let labels: Vec<&str> = rows.iter().map(|r| r[0].as_str()).collect();
    for (label, op) in HINTS {
        assert!(labels.contains(label), "no memory-hint row {label}");
        let mut found = false;
        for tier in TIERS {
            for e in tier.entries().iter().filter(|e| e.op == *op) {
                found = true;
                let c = e.baseline(tier).unwrap();
                assert_eq!(c.class, Class::N, "{op} {} on {}", e.ty, tier.name());
            }
        }
        assert!(found, "no table entry for {op}");
    }
    assert_eq!(labels.len(), HINTS.len(), "unmapped memory-hint rows: {labels:?}");
}

#[test]
fn every_table_op_comes_from_a_contract_row() {
    let mut known: BTreeSet<&str> = HINTS.iter().map(|h| h.1).collect();
    for r in OPS.iter().chain(LOOPS) {
        for (ops, _, _) in r.groups {
            known.extend(ops.iter().copied());
        }
    }
    for tier in TIERS {
        for e in tier.entries() {
            assert!(known.contains(e.op), "{}: {} is in no contract row", tier.name(), e.op);
        }
    }
}
