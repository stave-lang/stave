//! DWARF v5 debug sections: `.debug_line` with one sequence per function, plus a one-unit
//! `.debug_info`/`.debug_abbrev` whose `DW_AT_stmt_list` points at it (so `addr2line` and
//! debuggers find the table).

use crate::{LineRow, SourceFile};

const LINE_BASE: i64 = -5;
const LINE_RANGE: u64 = 14;
const OPCODE_BASE: u8 = 13;
const STD_OPCODE_LENGTHS: [u8; 12] = [0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1];

const DW_LNS_COPY: u8 = 1;
const DW_LNS_ADVANCE_PC: u8 = 2;
const DW_LNS_ADVANCE_LINE: u8 = 3;
const DW_LNS_SET_FILE: u8 = 4;
const DW_LNS_SET_COLUMN: u8 = 5;
const DW_LNE_END_SEQUENCE: u8 = 1;
const DW_LNE_SET_ADDRESS: u8 = 2;
const DW_LNCT_PATH: u64 = 1;
const DW_LNCT_DIRECTORY_INDEX: u64 = 2;
const DW_FORM_STRING: u64 = 0x08;
const DW_FORM_UDATA: u64 = 0x0f;

/// The machine code range of one function and its rows (offsets relative to `addr`).
#[derive(Clone, Copy, Debug)]
pub struct LineSeq<'a> {
    pub addr: u64,
    pub len: u64,
    pub rows: &'a [LineRow],
}

pub fn uleb(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

pub fn sleb(out: &mut Vec<u8>, mut v: i64) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        let done = (v == 0 && b & 0x40 == 0) || (v == -1 && b & 0x40 != 0);
        if done {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

fn cstr(out: &mut Vec<u8>, s: &str) {
    // An interior NUL would truncate the name; drop it rather than corrupt the table.
    out.extend(s.bytes().filter(|&b| b != 0));
    out.push(0);
}

/// Builds `.debug_line`. `FileId(n)` is DWARF file entry `n + 1`; entry 0 (the v5 primary file)
/// repeats `files[0]`, or is `cu_name` when `files` is empty. Entry 1 must exist because the
/// line-program state starts at file 1 (binutils rejects the table otherwise; GCC does the same).
/// Directory 0 is `comp_dir`, then each distinct file directory in first-use order.
pub fn debug_line(comp_dir: &str, cu_name: &str, files: &[SourceFile], seqs: &[LineSeq<'_>]) -> Vec<u8> {
    let mut dirs: Vec<&str> = vec![comp_dir];
    let mut file_dirs = Vec::with_capacity(files.len());
    for f in files {
        let d = match dirs.iter().position(|d| *d == f.dir) {
            Some(i) => i,
            None => {
                dirs.push(&f.dir);
                dirs.len() - 1
            }
        };
        file_dirs.push(d as u64);
    }
    // minimum_instruction_length, maximum_operations_per_instruction, default_is_stmt,
    // line_base, line_range, opcode_base
    let mut hdr = vec![1, 1, 1, LINE_BASE as i8 as u8, LINE_RANGE as u8, OPCODE_BASE];
    hdr.extend_from_slice(&STD_OPCODE_LENGTHS);
    hdr.push(1); // directory_entry_format_count
    uleb(&mut hdr, DW_LNCT_PATH);
    uleb(&mut hdr, DW_FORM_STRING);
    uleb(&mut hdr, dirs.len() as u64);
    for d in &dirs {
        cstr(&mut hdr, d);
    }
    hdr.push(2); // file_name_entry_format_count
    uleb(&mut hdr, DW_LNCT_PATH);
    uleb(&mut hdr, DW_FORM_STRING);
    uleb(&mut hdr, DW_LNCT_DIRECTORY_INDEX);
    uleb(&mut hdr, DW_FORM_UDATA);
    if files.is_empty() {
        uleb(&mut hdr, 2);
        for _ in 0..2 {
            cstr(&mut hdr, cu_name);
            uleb(&mut hdr, 0);
        }
    } else {
        uleb(&mut hdr, files.len() as u64 + 1);
        cstr(&mut hdr, &files[0].name);
        uleb(&mut hdr, file_dirs[0]);
        for (f, d) in files.iter().zip(&file_dirs) {
            cstr(&mut hdr, &f.name);
            uleb(&mut hdr, *d);
        }
    }

    let mut prog = Vec::new();
    for s in seqs {
        sequence(&mut prog, s);
    }

    let mut out = Vec::with_capacity(12 + hdr.len() + prog.len());
    let unit_len = 2 + 1 + 1 + 4 + hdr.len() + prog.len();
    out.extend_from_slice(&(unit_len as u32).to_le_bytes());
    out.extend_from_slice(&5u16.to_le_bytes());
    out.push(8); // address_size
    out.push(0); // segment_selector_size
    out.extend_from_slice(&(hdr.len() as u32).to_le_bytes());
    out.extend_from_slice(&hdr);
    out.extend_from_slice(&prog);
    out
}

fn sequence(out: &mut Vec<u8>, s: &LineSeq<'_>) {
    out.extend_from_slice(&[0, 9, DW_LNE_SET_ADDRESS]);
    out.extend_from_slice(&s.addr.to_le_bytes());
    let (mut file, mut line, mut col, mut off) = (1u64, 1i64, 0u64, 0u64);
    for r in s.rows {
        let (f, c) = (u64::from(r.file.0) + 1, u64::from(r.column));
        if f != file {
            out.push(DW_LNS_SET_FILE);
            uleb(out, f);
            file = f;
        }
        if c != col {
            out.push(DW_LNS_SET_COLUMN);
            uleb(out, c);
            col = c;
        }
        let da = u64::from(r.offset).saturating_sub(off);
        let mut dl = i64::from(r.line) - line;
        if !(LINE_BASE..LINE_BASE + LINE_RANGE as i64).contains(&dl) {
            out.push(DW_LNS_ADVANCE_LINE);
            sleb(out, dl);
            dl = 0;
        }
        let special = (dl - LINE_BASE) as u64 + LINE_RANGE * da + u64::from(OPCODE_BASE);
        if special <= 255 {
            out.push(special as u8);
        } else {
            if da != 0 {
                out.push(DW_LNS_ADVANCE_PC);
                uleb(out, da);
            }
            if dl != 0 {
                out.push(DW_LNS_ADVANCE_LINE);
                sleb(out, dl);
            }
            out.push(DW_LNS_COPY);
        }
        line = i64::from(r.line);
        off += da;
    }
    let rest = s.len.saturating_sub(off);
    if rest != 0 {
        out.push(DW_LNS_ADVANCE_PC);
        uleb(out, rest);
    }
    out.extend_from_slice(&[0, 1, DW_LNE_END_SEQUENCE]);
}

/// `.debug_abbrev` for [`debug_info`]: one `DW_TAG_compile_unit` without children.
pub fn debug_abbrev() -> Vec<u8> {
    vec![
        1, 0x11, 0, // code 1, DW_TAG_compile_unit, DW_CHILDREN_no
        0x25, 0x08, // DW_AT_producer, DW_FORM_string
        0x03, 0x08, // DW_AT_name, DW_FORM_string
        0x1b, 0x08, // DW_AT_comp_dir, DW_FORM_string
        0x10, 0x17, // DW_AT_stmt_list, DW_FORM_sec_offset
        0x11, 0x01, // DW_AT_low_pc, DW_FORM_addr
        0x12, 0x07, // DW_AT_high_pc, DW_FORM_data8 (length)
        0, 0, 0,
    ]
}

/// `.debug_info`: one v5 compile unit covering `[low, low + len)` with line table offset 0.
pub fn debug_info(cu_name: &str, comp_dir: &str, low: u64, len: u64) -> Vec<u8> {
    let mut die = Vec::new();
    uleb(&mut die, 1);
    cstr(&mut die, "stave stage0");
    cstr(&mut die, cu_name);
    cstr(&mut die, comp_dir);
    die.extend_from_slice(&0u32.to_le_bytes());
    die.extend_from_slice(&low.to_le_bytes());
    die.extend_from_slice(&len.to_le_bytes());
    let mut out = Vec::new();
    let unit_len = 2 + 1 + 1 + 4 + die.len();
    out.extend_from_slice(&(unit_len as u32).to_le_bytes());
    out.extend_from_slice(&5u16.to_le_bytes());
    out.push(0x01); // DW_UT_compile
    out.push(8);
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&die);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FileId;

    #[test]
    fn leb() {
        let mut v = Vec::new();
        uleb(&mut v, 624485);
        assert_eq!(v, [0xe5, 0x8e, 0x26]);
        v.clear();
        sleb(&mut v, -123456);
        assert_eq!(v, [0xc0, 0xbb, 0x78]);
        v.clear();
        sleb(&mut v, 63);
        sleb(&mut v, 64);
        sleb(&mut v, -64);
        sleb(&mut v, -65);
        assert_eq!(v, [0x3f, 0xc0, 0x00, 0x40, 0xbf, 0x7f]);
    }

    /// Decodes a line program and returns (address, file, line, column) rows plus end addresses.
    fn decode(sec: &[u8]) -> Vec<(u64, u64, i64, u64, bool)> {
        let hlen = u32::from_le_bytes(sec[8..12].try_into().unwrap()) as usize;
        let mut i = 12 + hlen;
        let mut rows = Vec::new();
        let (mut addr, mut file, mut line, mut col) = (0u64, 1u64, 1i64, 0u64);
        let rd_uleb = |i: &mut usize| {
            let (mut v, mut s) = (0u64, 0);
            loop {
                let b = sec[*i];
                *i += 1;
                v |= u64::from(b & 0x7f) << s;
                s += 7;
                if b & 0x80 == 0 {
                    return v;
                }
            }
        };
        let rd_sleb = |i: &mut usize| {
            let (mut v, mut s) = (0i64, 0);
            loop {
                let b = sec[*i];
                *i += 1;
                v |= i64::from(b & 0x7f) << s;
                s += 7;
                if b & 0x80 == 0 {
                    if s < 64 && b & 0x40 != 0 {
                        v |= -1 << s;
                    }
                    return v;
                }
            }
        };
        while i < sec.len() {
            let op = sec[i];
            i += 1;
            match op {
                0 => {
                    let _len = rd_uleb(&mut i);
                    let sub = sec[i];
                    i += 1;
                    if sub == 2 {
                        addr = u64::from_le_bytes(sec[i..i + 8].try_into().unwrap());
                        i += 8;
                    } else {
                        rows.push((addr, file, line, col, true));
                        (addr, file, line, col) = (0, 1, 1, 0);
                    }
                }
                1 => rows.push((addr, file, line, col, false)),
                2 => addr += rd_uleb(&mut i),
                3 => line += rd_sleb(&mut i),
                4 => file = rd_uleb(&mut i),
                5 => col = rd_uleb(&mut i),
                op if op >= OPCODE_BASE => {
                    let adj = u64::from(op - OPCODE_BASE);
                    addr += adj / LINE_RANGE;
                    line += LINE_BASE + (adj % LINE_RANGE) as i64;
                    rows.push((addr, file, line, col, false));
                }
                _ => panic!("unexpected opcode {op}"),
            }
        }
        rows
    }

    #[test]
    fn rows_round_trip() {
        let f = FileId(0);
        let g = FileId(1);
        let rows = [
            LineRow { offset: 0, file: f, line: 10, column: 1 },
            LineRow { offset: 4, file: f, line: 11, column: 5 },
            LineRow { offset: 400, file: f, line: 3, column: 5 },
            LineRow { offset: 404, file: g, line: 100_000, column: 0 },
            LineRow { offset: 404, file: g, line: 100_001, column: 0 },
        ];
        let files = [
            SourceFile { dir: "src".into(), name: "a.stv".into() },
            SourceFile { dir: "lib".into(), name: "b.stv".into() },
        ];
        let seqs = [
            LineSeq { addr: 0x40_1000, len: 500, rows: &rows },
            LineSeq { addr: 0x40_2000, len: 8, rows: &rows[..1] },
        ];
        let sec = debug_line(".", "a.stv", &files, &seqs);
        let got = decode(&sec);
        let mut want = Vec::new();
        for s in &seqs {
            for r in s.rows {
                want.push((
                    s.addr + u64::from(r.offset),
                    u64::from(r.file.0) + 1,
                    i64::from(r.line),
                    u64::from(r.column),
                    false,
                ));
            }
            want.push((s.addr + s.len, 1, 1, 0, true));
        }
        // end_sequence rows carry the state at the end; compare addresses for those
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(&want) {
            if w.4 {
                assert_eq!((g.0, g.4), (w.0, true));
            } else {
                assert_eq!(g, w);
            }
        }
    }
}
