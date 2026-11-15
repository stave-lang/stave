//! The stage-gates dumps of the loader's tables (contracts/stage-gates.md): `strings` and `names`,
//! beside [`crate::dump_types`]. Line-oriented, `\n`-terminated, no trailing spaces, independent
//! of the thread count. The `*_into` forms append to a buffer (the driver writes files from them;
//! `names` per file, so files can be rendered in parallel and written in file order).

use crate::Program;
use crate::resolve::Res;

const HEX: &[u8; 16] = b"0123456789abcdef";

/// The `strings` dump: `S hash32 "escaped-bytes"` per interned string, by S. S is the canonical
/// string id (rank of (xxHash32 seed 0, bytes)); hash32 is that hash as 8 lowercase hex digits.
/// Escaping: bytes 0x20–0x7E as themselves except `\` → `\\` and `"` → `\"`; 0x0A → `\n`,
/// 0x09 → `\t`; every other byte → `\xHH`.
pub fn dump_strings(prog: &Program) -> String {
    let mut buf = Vec::new();
    strings_into(prog, &mut buf);
    bytes_to_string(buf)
}

/// [`dump_strings`] appended to `buf`.
pub fn strings_into(prog: &Program, buf: &mut Vec<u8>) {
    let strings = &prog.strings;
    for (s, text) in strings.iter() {
        push_u32(buf, s.0);
        buf.push(b' ');
        let h = strings.hash(s);
        for k in (0..8).rev() {
            buf.push(HEX[((h >> (k * 4)) & 0xf) as usize]);
        }
        buf.extend_from_slice(b" \"");
        escape(text.as_bytes(), buf);
        buf.extend_from_slice(b"\"\n");
    }
}

/// The `names` dump: `I N -> decl D` / `I N -> local I N2` per resolving identifier use, by
/// (I, N), then by the identifier's byte offset. A use resolved to a declaration (item, member,
/// module, error decl) prints `-> decl D`; a local binding or a generic parameter prints
/// `-> local I N2` (N2 = its binding or `GenericParam` node). `self`, `Self`, built-in types and
/// fns, unresolved uses, and uses inside dropped items have no line.
pub fn dump_names(prog: &Program) -> String {
    let mut buf = Vec::new();
    for f in 0..prog.files.len() {
        names_file_into(prog, f, &mut buf);
    }
    bytes_to_string(buf)
}

/// The `names` lines of file `f` (by [`stave_diag::FileId`] index) appended to `buf`; the files'
/// lines in file order are [`dump_names`].
pub fn names_file_into(prog: &Program, f: usize, buf: &mut Vec<u8>) {
    let Some(file) = prog.files.get(f) else { return };
    for k in 0..file.parsed.items.len() {
        let i = file.first_item.saturating_add(k as u32);
        if !prog.items.get(i as usize).is_some_and(|it| it.ok) {
            continue;
        }
        let Some(res) = prog.res.get(i as usize) else { continue };
        for u in &res.uses {
            let (local, target) = match u.res {
                Res::Decl(d) => (false, d.0),
                Res::Local(b) => (true, b.node),
                Res::Generic(g) => (true, g.node),
                Res::SelfValue(_) | Res::SelfType | Res::Prim(_) | Res::Builtin(_) | Res::Unresolved => {
                    continue;
                }
            };
            push_u32(buf, i);
            buf.push(b' ');
            push_u32(buf, u.node);
            if local {
                buf.extend_from_slice(b" -> local ");
                push_u32(buf, i);
                buf.push(b' ');
            } else {
                buf.extend_from_slice(b" -> decl ");
            }
            push_u32(buf, target);
            buf.push(b'\n');
        }
    }
}

/// Appends the stage-gates `escaped-bytes` form of `bytes`.
fn escape(bytes: &[u8], buf: &mut Vec<u8>) {
    for &b in bytes {
        match b {
            b'\\' => buf.extend_from_slice(b"\\\\"),
            b'"' => buf.extend_from_slice(b"\\\""),
            b'\n' => buf.extend_from_slice(b"\\n"),
            b'\t' => buf.extend_from_slice(b"\\t"),
            0x20..=0x7e => buf.push(b),
            _ => buf.extend_from_slice(&[b'\\', b'x', HEX[(b >> 4) as usize], HEX[(b & 0xf) as usize]]),
        }
    }
}

/// Appends the decimal form of `n`.
fn push_u32(buf: &mut Vec<u8>, n: u32) {
    let mut tmp = [0u8; 10];
    let mut i = tmp.len();
    let mut v = n;
    loop {
        i -= 1;
        tmp[i] = b'0' + (v % 10) as u8;
        v /= 10;
        if v == 0 {
            break;
        }
    }
    buf.extend_from_slice(&tmp[i..]);
}

/// The dumps are ASCII (every other byte is escaped).
fn bytes_to_string(buf: Vec<u8>) -> String {
    String::from_utf8(buf).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned())
}

#[cfg(test)]
mod tests {
    use super::{escape, push_u32};

    #[test]
    fn escaping() {
        let mut b = Vec::new();
        escape(b"a\\\"\n\t\x7f\x00~ \xc3\xa9", &mut b);
        assert_eq!(b, b"a\\\\\\\"\\n\\t\\x7f\\x00~ \\xc3\\xa9");
    }

    #[test]
    fn decimal() {
        let mut b = Vec::new();
        for n in [0, 7, 10, 4_294_967_295] {
            push_u32(&mut b, n);
            b.push(b' ');
        }
        assert_eq!(b, b"0 7 10 4294967295 ");
    }
}
