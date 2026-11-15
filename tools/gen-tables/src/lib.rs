//! gen-tables: checks the declarative SIMD lowering tables (`crates/stave-target/tables/*.toml`)
//! and generates their Rust form (`crates/stave-target/src/generated/*.rs`).
//!
//! The format is described in the header of each table. Checks: every entry has one cell per tier
//! of its architecture, a class N, M or U, a recipe unless U, known type tokens, placeholders that
//! apply to every type of the entry, and no (op, form, type) covered twice. The generator expands
//! type groups to concrete keys, expands placeholders per key, and counts the cost.

use std::collections::BTreeMap;
use std::fmt::Write;
use std::path::{Path, PathBuf};

use serde::Deserialize;

/// One target architecture: its table, its generated file, its tiers in order.
pub struct Arch {
    pub table: &'static str,
    pub out: &'static str,
    pub tiers: &'static [&'static str],
    /// Size letters for 8, 16, 32, 64-bit lanes (`*`).
    pub sizes: [&'static str; 4],
}

pub const ARCHES: [Arch; 2] = [
    Arch {
        table: "crates/stave-target/tables/x86.toml",
        out: "crates/stave-target/src/generated/x86.rs",
        tiers: &["v2", "v3", "v4"],
        sizes: ["b", "w", "d", "q"],
    },
    Arch {
        table: "crates/stave-target/tables/a64.toml",
        out: "crates/stave-target/src/generated/a64.rs",
        tiers: &["neon", "sve"],
        sizes: ["b", "h", "w", "d"],
    },
];

/// Finds the repository root: the first ancestor of this crate's manifest directory that holds the
/// x86 table, whatever the current directory.
pub fn find_root() -> Option<PathBuf> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .find(|p| p.join(ARCHES[0].table).is_file())
        .map(Path::to_path_buf)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTable {
    op: Vec<RawEntry>,
}

#[derive(Deserialize)]
struct RawEntry {
    op: String,
    ty: Vec<String>,
    #[serde(default)]
    form: String,
    #[serde(flatten)]
    cells: BTreeMap<String, RawCell>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawCell {
    class: String,
    recipe: Option<String>,
    note: Option<String>,
    #[serde(default)]
    unverified: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub op: String,
    /// The type tokens as written, joined by `/`.
    pub ty: String,
    pub form: String,
    /// Concrete keys: lane types, masks, casts.
    pub keys: Vec<String>,
    pub cells: Vec<Cell>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cell {
    pub tier: String,
    pub feature: Option<String>,
    pub class: char,
    pub cost: u32,
    /// As written (empty for U).
    pub recipe: String,
    /// Expanded per key, in key order.
    pub recipes: Vec<String>,
    pub note: String,
    pub unverified: bool,
}

const LANES: [&str; 12] = ["i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f16", "bf16", "f32", "f64"];
const MASKS: [&str; 4] = ["m8", "m16", "m32", "m64"];
const FORMS: [&str; 3] = ["", "const", "vector"];

/// The lane types a lane token stands for.
fn lane_group(tok: &str) -> Option<Vec<&'static str>> {
    if let Some(&l) = LANES.iter().find(|&&l| l == tok) {
        return Some(vec![l]);
    }
    let width =
        |w: &str| LANES.iter().copied().filter(|l| lane_bits(l) == w.parse().ok()).collect::<Vec<_>>();
    let v = match tok {
        "int" => LANES[..8].to_vec(),
        "sint" => LANES[..4].to_vec(),
        "uint" => LANES[4..8].to_vec(),
        "float" => vec!["f32", "f64"],
        "any" => LANES.to_vec(),
        _ => match (tok.strip_prefix("int"), tok.strip_prefix("any")) {
            (Some(w), _) => width(w).into_iter().filter(|l| !is_float(l)).collect(),
            (_, Some(w)) => width(w),
            _ => Vec::new(),
        },
    };
    if v.is_empty() { None } else { Some(v) }
}

fn lane_bits(l: &str) -> Option<u32> {
    let digits = l.trim_start_matches(|c: char| c.is_ascii_alphabetic());
    digits.parse().ok()
}

fn is_float(l: &str) -> bool {
    l.starts_with('f') || l.starts_with("bf")
}

/// The concrete keys of one `ty` token.
fn keys_of(tok: &str) -> Result<Vec<String>, String> {
    if let Some((s, d)) = tok.split_once('>') {
        let src = lane_group(s).ok_or_else(|| format!("unknown cast source `{s}`"))?;
        let dst = lane_group(d).ok_or_else(|| format!("unknown cast target `{d}`"))?;
        return Ok(src.iter().flat_map(|s| dst.iter().map(move |d| format!("{s}>{d}"))).collect());
    }
    if tok == "mask" {
        return Ok(MASKS.iter().map(|m| m.to_string()).collect());
    }
    if MASKS.contains(&tok) {
        return Ok(vec![tok.to_string()]);
    }
    lane_group(tok)
        .map(|v| v.into_iter().map(String::from).collect())
        .ok_or_else(|| format!("unknown type `{tok}`"))
}

/// The lane a placeholder reads for `key`: the integer side of a cast for `* # ?`, the float side
/// for `%`; the key itself otherwise.
fn placeholder_lane(key: &str, float_side: bool) -> &str {
    match key.split_once('>') {
        Some((s, d)) => {
            let pick_src = if float_side { is_float(s) || !is_float(d) } else { !is_float(s) || is_float(d) };
            if pick_src { s } else { d }
        }
        None => key,
    }
}

fn expand(recipe: &str, key: &str, arch: &Arch) -> Result<String, String> {
    let mut out = String::with_capacity(recipe.len());
    for c in recipe.chars() {
        match c {
            '*' | '#' | '?' | '%' => {
                let lane = placeholder_lane(key, c == '%');
                let mask = lane.starts_with('m');
                let bits = lane_bits(lane).filter(|b| [8, 16, 32, 64].contains(b));
                let bad = || format!("placeholder `{c}` does not apply to `{key}`");
                match c {
                    '*' => {
                        let i = bits.ok_or_else(bad)?.trailing_zeros() - 3;
                        out.push_str(arch.sizes[i as usize]);
                    }
                    '#' => {
                        let b = bits.ok_or_else(bad)?;
                        let _ = write!(out, "{b}");
                    }
                    '?' if !mask && !is_float(lane) => {
                        out.push(if lane.starts_with('u') { 'u' } else { 's' })
                    }
                    '%' => match lane {
                        "f16" => out.push('h'),
                        "f32" => out.push('s'),
                        "f64" => out.push('d'),
                        _ => return Err(bad()),
                    },
                    _ => return Err(bad()),
                }
            }
            _ => out.push(c),
        }
    }
    Ok(out)
}

/// Instructions in a recipe: `; `-separated items, `xK` repeating one K times, `none` for none.
pub fn cost(recipe: &str) -> Result<u32, String> {
    if recipe == "none" {
        return Ok(0);
    }
    let mut n = 0;
    for item in recipe.split(';') {
        let item = item.trim();
        if item.is_empty() {
            return Err(format!("empty instruction in `{recipe}`"));
        }
        n += match item.rsplit_once(" x") {
            Some((head, k)) if !head.is_empty() && !k.is_empty() && k.bytes().all(|b| b.is_ascii_digit()) => {
                match k.parse::<u32>() {
                    Ok(k) if k >= 2 => k,
                    _ => return Err(format!("bad repeat count in `{item}`")),
                }
            }
            _ => 1,
        };
    }
    Ok(n)
}

/// Parses and checks one table.
pub fn parse(arch: &Arch, text: &str) -> Result<Vec<Entry>, String> {
    let raw: RawTable = toml::from_str(text).map_err(|e| format!("{}: {e}", arch.table))?;
    let mut errors = Vec::new();
    let mut entries = Vec::new();
    let mut seen: BTreeMap<(String, String, String), usize> = BTreeMap::new();
    for (i, e) in raw.op.into_iter().enumerate() {
        let at = format!("{}: entry {} ({})", arch.table, i + 1, e.op);
        match entry(arch, e) {
            Ok(entry) => {
                for k in &entry.keys {
                    let id = (entry.op.clone(), entry.form.clone(), k.clone());
                    if let Some(prev) = seen.insert(id, i + 1) {
                        errors.push(format!("{at}: `{k}` is already covered by entry {prev}"));
                    }
                }
                entries.push(entry);
            }
            Err(msg) => errors.push(format!("{at}: {msg}")),
        }
    }
    if errors.is_empty() { Ok(entries) } else { Err(errors.join("\n")) }
}

fn entry(arch: &Arch, e: RawEntry) -> Result<Entry, String> {
    let valid_op = e.op.split('.').count() >= 2
        && e.op.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'.' || b == b'_');
    if !valid_op {
        return Err("op must be a dotted lowercase name".into());
    }
    if !FORMS.contains(&e.form.as_str()) {
        return Err(format!("unknown form `{}` (const, vector)", e.form));
    }
    if e.ty.is_empty() {
        return Err("empty ty".into());
    }
    let mut keys = Vec::new();
    for t in &e.ty {
        for k in keys_of(t)? {
            if keys.contains(&k) {
                return Err(format!("`{k}` is listed twice"));
            }
            keys.push(k);
        }
    }
    let mut cells = Vec::new();
    for name in e.cells.keys() {
        let (tier, feature) = match name.split_once('+') {
            Some((t, f)) => (t, Some(f)),
            None => (name.as_str(), None),
        };
        if !arch.tiers.contains(&tier) {
            return Err(format!("unknown tier `{name}` (tiers: {})", arch.tiers.join(" ")));
        }
        if feature
            .is_some_and(|f| f.is_empty() || !f.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit()))
        {
            return Err(format!("bad feature in `{name}`"));
        }
    }
    for &tier in arch.tiers {
        if !e.cells.contains_key(tier) {
            return Err(format!("no `{tier}` cell"));
        }
        let mut names: Vec<&String> = e
            .cells
            .keys()
            .filter(|n| n.as_str() == tier || n.strip_prefix(tier).is_some_and(|r| r.starts_with('+')))
            .collect();
        names.sort_by_key(|n| (n.len() != tier.len(), n.as_str()));
        for name in names {
            let c = &e.cells[name];
            let feature = name.split_once('+').map(|(_, f)| f.to_string());
            cells.push(cell(arch, tier, feature, c, &keys).map_err(|m| format!("{name}: {m}"))?);
        }
    }
    Ok(Entry { op: e.op, ty: e.ty.join("/"), form: e.form, keys, cells })
}

fn cell(
    arch: &Arch,
    tier: &str,
    feature: Option<String>,
    c: &RawCell,
    keys: &[String],
) -> Result<Cell, String> {
    let class = match c.class.as_str() {
        "N" => 'N',
        "M" => 'M',
        "U" => 'U',
        other => return Err(format!("class `{other}` (N, M, U)")),
    };
    let (recipe, cost, recipes) = match (class, &c.recipe) {
        ('U', None) => (String::new(), 0, vec![String::new(); keys.len()]),
        ('U', Some(_)) => return Err("a U cell has no recipe".into()),
        (_, None) => return Err("no recipe".into()),
        (_, Some(r)) => {
            let n = cost(r)?;
            let rs = keys.iter().map(|k| expand(r, k, arch)).collect::<Result<Vec<_>, _>>()?;
            (r.clone(), n, rs)
        }
    };
    Ok(Cell {
        tier: tier.to_string(),
        feature,
        class,
        cost,
        recipe,
        recipes,
        note: c.note.clone().unwrap_or_default(),
        unverified: c.unverified,
    })
}

fn tier_variant(tier: &str) -> String {
    let mut s = String::new();
    let mut up = true;
    for ch in tier.chars() {
        if up {
            s.extend(ch.to_uppercase());
        } else {
            s.push(ch);
        }
        up = false;
    }
    s
}

/// The generated Rust file for one table.
pub fn emit(arch: &Arch, entries: &[Entry]) -> String {
    let mut o = String::new();
    let _ = writeln!(o, "// generated by tools/gen-tables from {}; do not edit\n", arch.table);
    o.push_str("use super::{Cell, Class, Entry, Tier};\n\n");
    o.push_str("pub(super) static ENTRIES: &[Entry] = &[\n");
    let list = |v: &[String]| v.iter().map(|s| format!("{s:?}")).collect::<Vec<_>>().join(", ");
    for e in entries {
        o.push_str("    Entry {\n");
        let _ = writeln!(o, "        op: {:?},", e.op);
        let _ = writeln!(o, "        ty: {:?},", e.ty);
        let _ = writeln!(o, "        form: {:?},", e.form);
        let _ = writeln!(o, "        keys: &[{}],", list(&e.keys));
        o.push_str("        cells: &[\n");
        for c in &e.cells {
            let feature = match &c.feature {
                Some(f) => format!("Some({f:?})"),
                None => "None".into(),
            };
            let _ = writeln!(
                o,
                "            Cell {{ tier: Tier::{}, feature: {feature}, class: Class::{}, cost: {}, recipe: {:?}, recipes: &[{}], note: {:?}, unverified: {} }},",
                tier_variant(&c.tier),
                c.class,
                c.cost,
                c.recipe,
                list(&c.recipes),
                c.note,
                c.unverified
            );
        }
        o.push_str("        ],\n    },\n");
    }
    o.push_str("];\n");
    o
}

/// Every generated file under `root`, as (path relative to root, content).
pub fn generate(root: &Path) -> Result<Vec<(&'static str, String)>, String> {
    let mut out = Vec::new();
    let mut errors = Vec::new();
    for arch in &ARCHES {
        let text = match std::fs::read_to_string(root.join(arch.table)) {
            Ok(t) => t,
            Err(e) => {
                errors.push(format!("{}: {e}", arch.table));
                continue;
            }
        };
        match parse(arch, &text) {
            Ok(entries) => out.push((arch.out, emit(arch, &entries))),
            Err(e) => errors.push(e),
        }
    }
    if errors.is_empty() { Ok(out) } else { Err(errors.join("\n")) }
}

#[cfg(test)]
mod tests {
    use super::*;

    const X86: &Arch = &ARCHES[0];
    const A64: &Arch = &ARCHES[1];

    #[test]
    fn costs() {
        assert_eq!(cost("none"), Ok(0));
        assert_eq!(cost("paddd"), Ok(1));
        assert_eq!(cost("pshufd; psrlq x2; pblendw"), Ok(4));
        assert_eq!(cost("vaddp% {k} x4; prfm pldl1keep"), Ok(5));
        assert_eq!(cost("roundp% 9"), Ok(1));
        assert!(cost("a;; b").is_err());
        assert!(cost("a x1").is_err());
    }

    #[test]
    fn keys_and_placeholders() {
        assert_eq!(keys_of("int16").unwrap(), ["i16", "u16"]);
        assert_eq!(keys_of("any16").unwrap(), ["i16", "u16", "f16", "bf16"]);
        assert_eq!(keys_of("int8>f32").unwrap(), ["i8>f32", "u8>f32"]);
        assert_eq!(keys_of("mask").unwrap(), MASKS);
        assert!(keys_of("int12").is_err() && keys_of("vec").is_err());
        assert_eq!(expand("vpmin?*", "u16", X86).unwrap(), "vpminuw");
        assert_eq!(expand("addp%", "f64", X86).unwrap(), "addpd");
        assert_eq!(expand("vmovdqu# {k}", "f32", X86).unwrap(), "vmovdqu32 {k}");
        assert_eq!(expand("ld1*", "i16", A64).unwrap(), "ld1h");
        assert_eq!(expand("pmovsx*d; cvtdq2ps", "i8>f32", X86).unwrap(), "pmovsxbd; cvtdq2ps");
        assert_eq!(expand("fcvtz?", "f64>u64", A64).unwrap(), "fcvtzu");
        assert!(expand("addp%", "i32", X86).is_err());
        assert!(expand("pmin?b", "f32", X86).is_err());
    }

    #[test]
    fn table_errors() {
        let one = |body: &str| parse(X86, &format!("[[op]]\nop = \"vec.add\"\n{body}"));
        let ok = "ty = [\"f32\"]\nv2 = { class = \"N\", recipe = \"addps\" }\nv3 = { class = \"N\", recipe = \"a\" }\nv4 = { class = \"N\", recipe = \"a\" }\n";
        let e = one(ok).unwrap();
        assert_eq!((e[0].cells.len(), e[0].cells[0].cost), (3, 1));
        assert!(one(&ok.replace("v4 =", "neon =")).unwrap_err().contains("unknown tier"));
        assert!(
            one(&ok.replace("v4 = { class = \"N\", recipe = \"a\" }\n", ""))
                .unwrap_err()
                .contains("no `v4` cell")
        );
        assert!(
            one(&ok.replace("class = \"N\", recipe = \"addps\"", "class = \"U\", recipe = \"x\"")).is_err()
        );
        assert!(
            one(&ok.replace("class = \"N\", recipe = \"addps\"", "class = \"Q\", recipe = \"x\"")).is_err()
        );
        let variant = format!("{ok}\"v4+avx512vbmi\" = {{ class = \"N\", recipe = \"vpermb\" }}\n");
        let e = one(&variant).unwrap();
        assert_eq!(e[0].cells[3].feature.as_deref(), Some("avx512vbmi"));
        let twice =
            format!("[[op]]\nop = \"vec.add\"\n{ok}[[op]]\nop = \"vec.add\"\n{}", ok.replace("f32", "float"));
        assert!(parse(X86, &twice).unwrap_err().contains("already covered"));
    }
}
