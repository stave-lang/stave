//! Identifier vocabulary. No word is a Stave keyword, a primitive type, a std name, or `io`/`std`.

/// Nouns: field names, variant names (capitalised), type names, module names.
pub const NOUNS: [&str; 96] = [
    "alpha", "beta", "gamma", "delta", "omega", "sigma", "kappa", "theta", "lambda", "zeta", "iota", "rho",
    "count", "total", "width", "height", "depth", "index", "offset", "limit", "score", "value", "weight",
    "price", "level", "rank", "speed", "angle", "size", "mass", "heat", "flow", "rate", "load", "gain",
    "base", "step", "seed", "hash", "mask", "flag", "tick", "slot", "page", "node", "edge", "leaf", "root",
    "tree", "path", "port", "link", "item", "unit", "cell", "grid", "tile", "zone", "area", "part", "band",
    "peak", "mode", "phase", "pulse", "frame", "chunk", "block", "batch", "queue", "stack", "heap", "pool",
    "ring", "wave", "beam", "spark", "stone", "river", "cloud", "storm", "field", "orbit", "comet", "lumen",
    "joule", "tensor", "vertex", "pixel", "voxel", "sample", "token", "glyph", "cursor", "anchor", "bucket",
];

/// Verbs: fn name prefixes.
pub const VERBS: [&str; 32] = [
    "calc", "eval", "scan", "fold", "blend", "pack", "split", "merge", "probe", "shift", "scale", "clamp",
    "route", "advance", "tally", "sort", "pick", "lookup", "measure", "rotate", "encode", "decode", "check",
    "update", "reduce", "apply", "build", "fetch", "store", "mix", "trace", "warp",
];

/// Comment words.
pub const NOTES: [&str; 16] = [
    "keeps", "the", "running", "sum", "in", "range", "for", "each", "step", "of", "a", "small", "fixed",
    "window", "over", "inputs",
];

/// Capitalised `w` appended to `out`.
pub fn push_pascal(out: &mut String, w: &str) {
    let mut cs = w.chars();
    if let Some(c) = cs.next() {
        out.push(c.to_ascii_uppercase());
        out.push_str(cs.as_str());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RESERVED: [&str; 62] = [
        "fn",
        "let",
        "mut",
        "const",
        "struct",
        "enum",
        "trait",
        "impl",
        "for",
        "in",
        "while",
        "loop",
        "if",
        "then",
        "else",
        "match",
        "return",
        "break",
        "continue",
        "pub",
        "import",
        "as",
        "from",
        "needs",
        "dispatch",
        "unsafe",
        "extern",
        "self",
        "true",
        "false",
        "tail",
        "masked",
        "scalar",
        "exact",
        "align",
        "copy",
        "type",
        "view",
        "scope",
        "spawn",
        "strict",
        "emulate",
        "test",
        "allocator",
        "kernel",
        "static",
        "requires",
        "ensures",
        "rec",
        "without",
        "invariant",
        "decreases",
        "ghost",
        "pure",
        "spec",
        "forall",
        "exists",
        "derive",
        "never",
        "io",
        "std",
        "main",
    ];

    #[test]
    fn vocabulary_is_clean_and_unique() {
        let mut all: Vec<&str> = NOUNS.iter().chain(VERBS.iter()).copied().collect();
        for w in &all {
            assert!(!RESERVED.contains(w), "{w}");
            assert!(w.bytes().all(|b| b.is_ascii_lowercase()), "{w}");
        }
        all.sort_unstable();
        let n = all.len();
        all.dedup();
        assert_eq!(n, all.len(), "duplicate word");
    }
}
