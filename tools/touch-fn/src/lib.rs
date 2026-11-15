//! touch-fn: edits one fn body of a Stave package without changing any signature, for the warm
//! incremental rebuild budget.
//!
//! The chosen body (seeded, over all fn bodies of all `.stv` files in path order) gets the
//! statement `let touch_<n>: u32 = <v>;` as its first statement. When that body already starts
//! with such a statement, only its value changes, so repeated touches do not move any line.
//! The statement is valid in every fn body and changes no type, borrow or signature.

use std::path::{Path, PathBuf};

/// A fn body: `{` and its matching `}` (byte offsets).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FnBody {
    pub open: usize,
    pub close: usize,
}

/// The edit that was made.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Touch {
    pub file: PathBuf,
    /// 1-based line of the inserted or rewritten statement.
    pub line: usize,
    /// True when an earlier touch statement was rewritten in place.
    pub in_place: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tok {
    Ident,
    Punct(u8),
    Other,
}

/// Tokens with byte spans; comments, strings, f-strings and chars are single `Other` tokens.
fn tokens(src: &[u8]) -> Vec<(Tok, usize, usize)> {
    let n = src.len();
    let at = |j: usize| src.get(j).copied().unwrap_or(0);
    let idc = |b: u8| b.is_ascii_alphanumeric() || b == b'_' || b >= 0x80;
    let mut out = Vec::new();
    let mut i = 0;
    let quoted = |mut j: usize, q: u8| {
        while j < n && src[j] != q && src[j] != b'\n' {
            j += if src[j] == b'\\' { 2 } else { 1 };
        }
        (j + 1).min(n)
    };
    while i < n {
        let c = src[i];
        let s = i;
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if c == b'/' && at(i + 1) == b'/' {
            while i < n && src[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if c == b'f' && at(i + 1) == b'"' {
            i = quoted(i + 2, b'"');
            out.push((Tok::Other, s, i));
        } else if c == b'"' || c == b'\'' {
            i = quoted(i + 1, c);
            out.push((Tok::Other, s, i));
        } else if c.is_ascii_digit() {
            while i < n && (idc(src[i]) || (src[i] == b'.' && at(i + 1).is_ascii_digit())) {
                i += 1;
            }
            out.push((Tok::Other, s, i));
        } else if idc(c) {
            while i < n && idc(src[i]) {
                i += 1;
            }
            out.push((Tok::Ident, s, i));
        } else {
            i += 1;
            out.push((Tok::Punct(c), s, i));
        }
    }
    out
}

/// Every fn body of a file: top-level fns (also `test`, `extern "C"`, `unsafe`, `kernel` fns)
/// and methods of impl blocks. Trait signatures, extern imports, closures, fn types and
/// `dispatch fn` tables are not bodies.
pub fn fn_bodies(src: &str) -> Vec<FnBody> {
    let b = src.as_bytes();
    let toks = tokens(b);
    let word = |k: usize| -> &str {
        match toks.get(k) {
            Some(&(Tok::Ident, s, e)) => &src[s..e],
            _ => "",
        }
    };
    let mut bodies = Vec::new();
    // Stack of open braces: true = impl block body.
    let mut stack: Vec<bool> = Vec::new();
    let mut head_impl = false;
    let mut k = 0;
    while k < toks.len() {
        match toks[k].0 {
            Tok::Ident if word(k) == "fn" => {
                let at_item = stack.is_empty() || stack == [true];
                let named = matches!(toks.get(k + 1), Some(&(Tok::Ident, _, _)));
                let dispatch = k > 0 && word(k - 1) == "dispatch";
                if at_item
                    && named
                    && !dispatch
                    && let Some((open_k, close_k)) = body_of(&toks, k + 2)
                {
                    bodies.push(FnBody { open: toks[open_k].1, close: toks[close_k].1 });
                    k = close_k + 1;
                    if stack.is_empty() {
                        head_impl = false;
                    }
                    continue;
                }
            }
            Tok::Ident if stack.is_empty() && word(k) == "impl" => head_impl = true,
            Tok::Punct(b'{') => {
                stack.push(stack.is_empty() && head_impl);
                head_impl = false;
            }
            Tok::Punct(b'}') => {
                stack.pop();
            }
            Tok::Punct(b';') if stack.is_empty() => head_impl = false,
            _ => {}
        }
        k += 1;
    }
    bodies
}

/// From the token after the fn name: the body's `{` and matching `}` token indices, or None
/// for a signature ending in `;`.
fn body_of(toks: &[(Tok, usize, usize)], mut k: usize) -> Option<(usize, usize)> {
    let mut paren = 0i32;
    let mut open = None;
    while k < toks.len() {
        match toks[k].0 {
            Tok::Punct(b'(') | Tok::Punct(b'[') => paren += 1,
            Tok::Punct(b')') | Tok::Punct(b']') => paren -= 1,
            Tok::Punct(b';') if paren == 0 => return None,
            Tok::Punct(b'{') if paren == 0 => {
                let prev = if k > 0 { toks[k - 1].0 } else { Tok::Other };
                if prev != Tok::Punct(b'<') && prev != Tok::Punct(b',') {
                    open = Some(k);
                    break;
                }
                // `{const-expr}` generic argument: skip to its `}`.
                let mut d = 0;
                while k < toks.len() {
                    match toks[k].0 {
                        Tok::Punct(b'{') => d += 1,
                        Tok::Punct(b'}') => {
                            d -= 1;
                            if d == 0 {
                                break;
                            }
                        }
                        _ => {}
                    }
                    k += 1;
                }
            }
            _ => {}
        }
        k += 1;
    }
    let open = open?;
    let mut d = 0;
    for (j, t) in toks.iter().enumerate().skip(open) {
        match t.0 {
            Tok::Punct(b'{') => d += 1,
            Tok::Punct(b'}') => {
                d -= 1;
                if d == 0 {
                    return Some((open, j));
                }
            }
            _ => {}
        }
    }
    None
}

/// `.stv` files under `dir` (its `src/` when present), sorted by path.
pub fn stave_files(dir: &Path) -> Result<Vec<PathBuf>, String> {
    fn walk(d: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
        let rd = std::fs::read_dir(d).map_err(|e| format!("{}: {e}", d.display()))?;
        for e in rd {
            let p = e.map_err(|e| format!("{}: {e}", d.display()))?.path();
            if p.is_dir() {
                walk(&p, out)?;
            } else if p.extension().is_some_and(|x| x == "stv") {
                out.push(p);
            }
        }
        Ok(())
    }
    let src = dir.join("src");
    let root = if src.is_dir() { src } else { dir.to_path_buf() };
    let mut out = Vec::new();
    walk(&root, &mut out)?;
    out.sort();
    Ok(out)
}

fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Value of an existing `let touch_<n>: u32 = <v>;` right after `open`: (value start, end).
fn existing_touch(src: &str, open: usize) -> Option<(usize, usize)> {
    let rest = &src[open + 1..];
    let lead = rest.len() - rest.trim_start().len();
    let t = &rest[lead..];
    let after = t.strip_prefix("let touch_")?;
    let digits = after.bytes().take_while(u8::is_ascii_digit).count();
    let after = after[digits..].strip_prefix(": u32 = ")?;
    let vlen = after.bytes().take_while(u8::is_ascii_digit).count();
    if digits == 0 || vlen == 0 || !after[vlen..].starts_with(';') {
        return None;
    }
    let vs = open + 1 + lead + "let touch_".len() + digits + ": u32 = ".len();
    Some((vs, vs + vlen))
}

/// Smallest `n` with no identifier `touch_<n>` in `src`.
fn fresh_index(src: &str) -> u64 {
    let mut max: Option<u64> = None;
    let mut rest = src;
    while let Some(p) = rest.find("touch_") {
        let after = &rest[p + 6..];
        let d: String = after.chars().take_while(char::is_ascii_digit).collect();
        if let Ok(v) = d.parse::<u64>() {
            max = Some(max.map_or(v, |m| m.max(v)));
        }
        rest = after;
    }
    max.map_or(0, |m| m + 1)
}

/// Edits `src` at `body`; returns the new text, the 1-based line edited and whether in place.
pub fn touch_source(src: &str, body: FnBody, seed: u64) -> (String, usize, bool) {
    let line_of = |pos: usize| src[..pos].bytes().filter(|&c| c == b'\n').count() + 1;
    if let Some((vs, ve)) = existing_touch(src, body.open) {
        let v: u64 = src[vs..ve].parse().unwrap_or(0);
        let nv = (v + 1 + mix64(seed) % 7) % 1_000_000;
        let mut out = String::with_capacity(src.len() + 8);
        out.push_str(&src[..vs]);
        out.push_str(&nv.to_string());
        out.push_str(&src[ve..]);
        return (out, line_of(vs), true);
    }
    let n = fresh_index(src);
    let v = mix64(seed ^ 0x5EED) % 1000;
    let stmt = format!("let touch_{n}: u32 = {v};");
    let after = &src[body.open + 1..];
    let ws = after.len() - after.trim_start_matches([' ', '\t']).len();
    let mut out = String::with_capacity(src.len() + stmt.len() + 16);
    if after[ws..].starts_with('\n') || after[ws..].starts_with("\r\n") {
        // Multi-line body: a new line, indented like the first body line (or the fn plus 4).
        let nl = body.open + 1 + after[ws..].find('\n').map_or(0, |p| ws + p);
        let next = &src[nl + 1..];
        let mut indent: String = next.chars().take_while(|c| *c == ' ' || *c == '\t').collect();
        if next[indent.len()..].starts_with('}') {
            let ls = src[..body.open].rfind('\n').map_or(0, |p| p + 1);
            indent = src[ls..].chars().take_while(|c| *c == ' ' || *c == '\t').collect();
            indent.push_str("    ");
        }
        out.push_str(&src[..nl + 1]);
        out.push_str(&indent);
        out.push_str(&stmt);
        out.push('\n');
        out.push_str(&src[nl + 1..]);
        (out, line_of(nl + 1), false)
    } else {
        // One-line body: insert right after `{`.
        out.push_str(&src[..body.open + 1]);
        out.push(' ');
        out.push_str(&stmt);
        if body.close == body.open + 1 {
            out.push(' ');
        }
        out.push_str(&src[body.open + 1..]);
        (out, line_of(body.open), false)
    }
}

/// Picks one fn body of the package at `dir` (seeded) and edits it in place on disk.
pub fn touch(dir: &Path, seed: u64) -> Result<Touch, String> {
    let files = stave_files(dir)?;
    let mut all: Vec<(usize, FnBody)> = Vec::new();
    let mut texts = Vec::with_capacity(files.len());
    for (i, f) in files.iter().enumerate() {
        let t = std::fs::read_to_string(f).map_err(|e| format!("{}: {e}", f.display()))?;
        all.extend(fn_bodies(&t).into_iter().map(|b| (i, b)));
        texts.push(t);
    }
    if all.is_empty() {
        return Err(format!("{}: no fn bodies", dir.display()));
    }
    let (fi, body) = all[(mix64(seed) % all.len() as u64) as usize];
    let (text, line, in_place) = touch_source(&texts[fi], body, seed);
    std::fs::write(&files[fi], text).map_err(|e| format!("{}: {e}", files[fi].display()))?;
    Ok(Touch { file: files[fi].clone(), line, in_place })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SRC: &str = r#"import std.io;

// fn commented(x: i32) { }
pub struct S { pub a: i32 }
trait T { fn sig(&self) -> i32; }
extern "C" { fn puts(s: *const u8) -> i32; }
extern "C" fn add(a: i32, b: i32) -> i32 { return a + b; }
dispatch fn sum(xs: &[f32]) -> f32 { avx2 => sum_v3, _ => sum_base, }
impl S {
    pub fn get(&self) -> i32 {
        let f = fn(x: i32) -> i32 { return x + 1; };
        return f(self.a);
    }
    fn empty(&self) {}
}
impl T for S { fn sig(&self) -> i32 { return 1; } }
fn h() -> fn(i32) -> i32 { return g; }
fn k<const N: usize>() -> A<{N + 1}> {
    io.print(f"{N} }} {{\n");
    let c: char = '}';
}
test fn t1() { }
"#;

    #[test]
    fn finds_exactly_the_bodies() {
        let bs = fn_bodies(SRC);
        let heads: Vec<String> = bs
            .iter()
            .map(|b| {
                let ls = SRC[..b.open].rfind('\n').map_or(0, |p| p + 1);
                SRC[ls..b.open].trim().to_string()
            })
            .collect();
        assert_eq!(
            heads,
            [
                "extern \"C\" fn add(a: i32, b: i32) -> i32",
                "pub fn get(&self) -> i32",
                "fn empty(&self)",
                "impl T for S { fn sig(&self) -> i32",
                "fn h() -> fn(i32) -> i32",
                "fn k<const N: usize>() -> A<{N + 1}>",
                "test fn t1()",
            ]
        );
        for b in &bs {
            assert_eq!(&SRC[b.open..=b.open], "{");
            assert_eq!(&SRC[b.close..=b.close], "}");
        }
    }

    #[test]
    fn multi_line_insert_then_rewrite_in_place() {
        let bs = fn_bodies(SRC);
        let (a, line, inp) = touch_source(SRC, bs[1], 3);
        assert!(!inp);
        assert!(a.lines().nth(line - 1).is_some_and(|l| l.starts_with("        let touch_0: u32 = ")));
        assert_eq!(a.lines().count(), SRC.lines().count() + 1);
        let bs2 = fn_bodies(&a);
        assert_eq!(bs2.len(), bs.len());
        let (b, line2, inp2) = touch_source(&a, bs2[1], 3);
        assert!(inp2);
        assert_eq!(line2, line);
        assert_eq!(b.lines().count(), a.lines().count());
        assert_ne!(a, b);
    }

    #[test]
    fn one_line_and_empty_bodies() {
        let bs = fn_bodies(SRC);
        let (a, line, _) = touch_source(SRC, bs[0], 1);
        assert_eq!(a.lines().count(), SRC.lines().count());
        assert!(
            a.lines()
                .nth(line - 1)
                .is_some_and(|l| l.contains("{ let touch_0: u32 = ") && l.ends_with("return a + b; }"))
        );
        let (e, line, _) = touch_source(SRC, bs[2], 1);
        assert!(
            e.lines()
                .nth(line - 1)
                .is_some_and(|l| l.contains("fn empty(&self) { let touch_0: u32 = ") && l.ends_with("; }"))
        );
    }

    #[test]
    fn fresh_names_do_not_collide() {
        assert_eq!(fresh_index("let touch_4: u32 = 1; let touch_x = 2;"), 5);
        assert_eq!(fresh_index("nothing"), 0);
    }
}
