//! The mutator of printed IR text: byte (flip, delete, insert, duplicate a range), token (replace
//! with a token from a list of limits and keywords, delete, insert, swap, a number changed by one
//! or set to a limit) and line (delete, duplicate, swap) changes from a seeded xorshift. The
//! reader's fuzz test and the stage1 verdict comparison (`ir_text_mutants`) use this one mutator.

/// xorshift64.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 { 0 } else { (self.next() % n as u64) as usize }
    }
}

/// Tokens other inputs hold: ids, numbers at their limits, keywords, punctuation.
const TOKENS: &[&str] = &[
    "%0",
    "%4294967295",
    "%4294967296",
    "b0",
    "b9",
    "t0",
    "t99999",
    "s0",
    "d0",
    "@0",
    "@0.u",
    "@1.0",
    "0",
    "-1",
    "1",
    "255",
    "256",
    "4294967295",
    "18446744073709551615",
    "18446744073709551616",
    "170141183460469231731687303715884105727",
    "-170141183460469231731687303715884105728",
    "170141183460469231731687303715884105728",
    "0x0000",
    "0xffff",
    "0x7fc00001",
    "0x80000000",
    "0x0000000000000001",
    "0x7ff8000000000001",
    "bf16",
    "x00",
    "xff",
    "()",
    "(",
    ")",
    ",",
    ":",
    "=",
    "!",
    "|",
    "{",
    "}",
    "->",
    "<=",
    "\"\"",
    "\"\\x00\"",
    "\"\\xff\"",
    "fn",
    "type",
    "shape",
    "claim",
    "pre",
    "post",
    "checked",
    "main",
    "cell",
    "decl",
    "static",
    "data",
    "ref",
    "site",
    "str",
    "node",
    "edge",
    "nodes",
    "runstack",
    "truncated",
    "br",
    "cbr",
    "switch",
    "ret",
    "unreachable",
    "_:",
    "call",
    "intrinsic.syscall3",
    "intrinsic.x",
    "add",
    "slot",
    "load",
    "store",
    "known",
    "unbuilt",
    "rec",
    "#",
    "\t",
    "\r",
];

/// One to three byte-, token- or line-level changes of `base`.
pub fn mutate(r: &mut Rng, base: &[u8]) -> Vec<u8> {
    let mut v = base.to_vec();
    for _ in 0..1 + r.below(3) {
        match r.below(10) {
            // byte level
            0 if !v.is_empty() => {
                let i = r.below(v.len());
                v[i] = r.next() as u8;
            }
            1 if !v.is_empty() => {
                let i = r.below(v.len());
                v.remove(i);
            }
            2 => {
                let i = r.below(v.len() + 1);
                v.insert(i, r.next() as u8);
            }
            3 if !v.is_empty() => {
                let i = r.below(v.len());
                let n = r.below(64).min(v.len() - i);
                let dup: Vec<u8> = v[i..i + n].to_vec();
                v.splice(i..i, dup);
            }
            // token level: the tokens of one line
            4..=7 => {
                let lines: Vec<&[u8]> = v.split(|&c| c == b'\n').collect();
                let k = r.below(lines.len());
                let mut toks: Vec<Vec<u8>> = lines[k].split(|&c| c == b' ').map(<[u8]>::to_vec).collect();
                let j = r.below(toks.len());
                match r.below(5) {
                    0 => toks[j] = TOKENS[r.below(TOKENS.len())].as_bytes().to_vec(),
                    1 => {
                        toks.remove(j);
                    }
                    2 => toks.insert(j, TOKENS[r.below(TOKENS.len())].as_bytes().to_vec()),
                    3 => {
                        let i = r.below(toks.len());
                        toks.swap(i, j);
                    }
                    _ => {
                        // a number: change it by one, or to a limit
                        if let Some(d) = toks[j].iter().position(u8::is_ascii_digit) {
                            let (head, digits) = toks[j].split_at(d);
                            let n: u128 =
                                std::str::from_utf8(digits).ok().and_then(|s| s.parse().ok()).unwrap_or(0);
                            let m = match r.below(4) {
                                0 => n.wrapping_add(1),
                                1 => n.wrapping_sub(1),
                                2 => u128::from(u32::MAX),
                                _ => u128::MAX,
                            };
                            toks[j] = [head, m.to_string().as_bytes()].concat();
                        }
                    }
                }
                let line = toks.join(&b' ');
                let mut out: Vec<Vec<u8>> = lines.iter().map(|l| l.to_vec()).collect();
                out[k] = line;
                v = out.join(&b'\n');
            }
            // line level
            _ => {
                let mut lines: Vec<Vec<u8>> = v.split(|&c| c == b'\n').map(<[u8]>::to_vec).collect();
                let (i, j) = (r.below(lines.len()), r.below(lines.len()));
                match r.below(3) {
                    0 => {
                        lines.remove(i);
                    }
                    1 => {
                        let l = lines[i].clone();
                        lines.insert(j, l);
                    }
                    _ => lines.swap(i, j),
                }
                v = lines.join(&b'\n');
            }
        }
    }
    v
}
