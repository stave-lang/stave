//! EBNF reader for the syntactic part of `contracts/grammar.ebnf`: everything from the
//! `(* ---------- items` marker on. Lexical rules before the marker are not read; their names
//! (`Ident`, `IntLit`, ...) are terminals (see [`crate::lower::LEX_CLASSES`]).
//!
//! Notation: `A = x y | z ;`, `"t"` or `'t'` terminals, `( )` groups, postfix `? * +`,
//! `(* ... *)` comments (not nested). Every error carries the 1-based line of the source file.

use crate::Error;

/// Start of the syntactic part of the grammar file.
pub const ITEMS_MARKER: &str = "(* ---------- items";

/// Maximum nesting of groups and postfix operators (malformed input never overflows the stack).
const MAX_DEPTH: usize = 200;

/// EBNF expression tree. `Alt` holds `Seq`s; `Group` holds an `Alt`.
#[derive(Debug, Clone)]
pub enum Expr {
    Alt(Vec<Expr>),
    Seq(Vec<Expr>),
    /// Terminal spelling and line.
    Term(String, usize),
    /// Nonterminal or lexical class name and line.
    Name(String, usize),
    Group(Box<Expr>),
    Opt(Box<Expr>),
    Star(Box<Expr>),
    Plus(Box<Expr>),
    /// `X1..Xk (S X1..Xk)* S?`, produced only by [`crate::lower`]'s separator rewrite:
    /// `item` is the `Seq` X1..Xk, `sep` the terminal S.
    Sep {
        item: Box<Expr>,
        sep: Box<Expr>,
    },
}

/// Structural equality, ignoring line numbers.
pub fn same(a: &Expr, b: &Expr) -> bool {
    use Expr::*;
    let all = |x: &[Expr], y: &[Expr]| x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same(p, q));
    match (a, b) {
        (Alt(x), Alt(y)) | (Seq(x), Seq(y)) => all(x, y),
        (Term(x, _), Term(y, _)) | (Name(x, _), Name(y, _)) => x == y,
        (Group(x), Group(y)) | (Opt(x), Opt(y)) | (Star(x), Star(y)) | (Plus(x), Plus(y)) => same(x, y),
        (Sep { item: i1, sep: s1 }, Sep { item: i2, sep: s2 }) => same(i1, i2) && same(s1, s2),
        _ => false,
    }
}

/// One grammar rule `name = body ;`.
#[derive(Debug, Clone)]
pub struct Rule {
    pub name: String,
    pub line: usize,
    /// Always an `Expr::Alt` of `Expr::Seq`s.
    pub body: Expr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Tok {
    Term(String),
    Name(String),
    Punct(u8),
}

#[derive(Debug, Clone)]
struct Token {
    tok: Tok,
    line: usize,
}

fn describe(t: &Tok) -> String {
    match t {
        Tok::Term(s) => format!("terminal \"{s}\""),
        Tok::Name(s) => format!("name `{s}`"),
        Tok::Punct(c) => format!("`{}`", *c as char),
    }
}

fn tokenize(src: &str, first_line: usize) -> Result<Vec<Token>, Error> {
    let b = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    let mut line = first_line;
    while i < b.len() {
        let c = b[i];
        match c {
            b'\n' => {
                line += 1;
                i += 1;
            }
            b' ' | b'\t' | b'\r' => i += 1,
            b'(' if b.get(i + 1) == Some(&b'*') => {
                let start = line;
                i += 2;
                loop {
                    match b.get(i) {
                        None => return Err(Error::at(start, "unterminated comment `(*`")),
                        Some(b'*') if b.get(i + 1) == Some(&b')') => {
                            i += 2;
                            break;
                        }
                        Some(b'\n') => {
                            line += 1;
                            i += 1;
                        }
                        Some(_) => i += 1,
                    }
                }
            }
            b'"' | b'\'' => {
                let start = i + 1;
                let mut j = start;
                while j < b.len() && b[j] != c && b[j] != b'\n' {
                    j += 1;
                }
                if j >= b.len() || b[j] != c {
                    return Err(Error::at(line, "unterminated terminal literal"));
                }
                if j == start {
                    return Err(Error::at(line, "empty terminal literal"));
                }
                out.push(Token { tok: Tok::Term(src[start..j].to_string()), line });
                i = j + 1;
            }
            b'A'..=b'Z' | b'a'..=b'z' | b'_' => {
                let start = i;
                while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                    i += 1;
                }
                out.push(Token { tok: Tok::Name(src[start..i].to_string()), line });
            }
            b'=' | b';' | b'|' | b'(' | b')' | b'?' | b'*' | b'+' => {
                out.push(Token { tok: Tok::Punct(c), line });
                i += 1;
            }
            _ => {
                let ch = src[i..].chars().next().unwrap_or('?');
                return Err(Error::at(line, format!("unexpected character {ch:?}")));
            }
        }
    }
    Ok(out)
}

struct Parser {
    toks: Vec<Token>,
    i: usize,
    last_line: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.i).map(|t| &t.tok)
    }

    fn line(&self) -> usize {
        self.toks.get(self.i).map_or(self.last_line, |t| t.line)
    }

    fn punct(&self, c: u8) -> bool {
        self.peek() == Some(&Tok::Punct(c))
    }

    fn expect(&mut self, c: u8, what: &str) -> Result<(), Error> {
        if self.punct(c) {
            self.i += 1;
            return Ok(());
        }
        let got = self.peek().map_or_else(|| "end of file".to_string(), describe);
        Err(Error::at(self.line(), format!("expected `{}` {what}, got {got}", c as char)))
    }

    fn rules(&mut self) -> Result<Vec<Rule>, Error> {
        let mut out: Vec<Rule> = Vec::new();
        while let Some(t) = self.peek() {
            let line = self.line();
            let Tok::Name(name) = t.clone() else {
                return Err(Error::at(line, format!("expected a rule name, got {}", describe(t))));
            };
            self.i += 1;
            self.expect(b'=', &format!("after rule name `{name}`"))?;
            let body = self.alt(0)?;
            self.expect(b';', &format!("at the end of rule `{name}`"))?;
            if let Some(prev) = out.iter().find(|r| r.name == name) {
                return Err(Error::at(line, format!("rule `{name}` already defined on line {}", prev.line)));
            }
            out.push(Rule { name, line, body });
        }
        Ok(out)
    }

    fn alt(&mut self, depth: usize) -> Result<Expr, Error> {
        if depth > MAX_DEPTH {
            return Err(Error::at(self.line(), "grammar nested too deeply"));
        }
        let mut alts = vec![self.seq(depth)?];
        while self.punct(b'|') {
            self.i += 1;
            alts.push(self.seq(depth)?);
        }
        Ok(Expr::Alt(alts))
    }

    fn seq(&mut self, depth: usize) -> Result<Expr, Error> {
        let mut items = Vec::new();
        loop {
            match self.peek() {
                None | Some(Tok::Punct(b'|' | b')' | b';')) => break,
                Some(Tok::Punct(b'=')) => {
                    return Err(Error::at(
                        self.line(),
                        "unexpected `=` (missing `;` after the previous rule?)",
                    ));
                }
                _ => items.push(self.postfix(depth)?),
            }
        }
        Ok(Expr::Seq(items))
    }

    fn postfix(&mut self, depth: usize) -> Result<Expr, Error> {
        let mut x = self.atom(depth)?;
        let mut n = 0;
        while let Some(Tok::Punct(c @ (b'?' | b'*' | b'+'))) = self.peek() {
            let c = *c;
            self.i += 1;
            n += 1;
            if n > MAX_DEPTH {
                return Err(Error::at(self.line(), "too many postfix operators"));
            }
            x = match c {
                b'?' => Expr::Opt(Box::new(x)),
                b'*' => Expr::Star(Box::new(x)),
                _ => Expr::Plus(Box::new(x)),
            };
        }
        Ok(x)
    }

    fn atom(&mut self, depth: usize) -> Result<Expr, Error> {
        let line = self.line();
        let Some(t) = self.peek().cloned() else {
            return Err(Error::at(line, "unexpected end of file"));
        };
        self.i += 1;
        match t {
            Tok::Punct(b'(') => {
                let a = self.alt(depth + 1)?;
                self.expect(b')', "to close the group")?;
                Ok(Expr::Group(Box::new(a)))
            }
            Tok::Term(s) => Ok(Expr::Term(s, line)),
            Tok::Name(s) => Ok(Expr::Name(s, line)),
            Tok::Punct(_) => Err(Error::at(line, format!("expected a symbol or `(`, got {}", describe(&t)))),
        }
    }
}

/// Parses the rules from [`ITEMS_MARKER`] on. Line numbers are lines of `src`.
pub fn parse(src: &str) -> Result<Vec<Rule>, Error> {
    let Some(at) = src.find(ITEMS_MARKER) else {
        return Err(Error::new(format!("missing the items marker `{ITEMS_MARKER}`")));
    };
    let first_line = src[..at].matches('\n').count() + 1;
    let toks = tokenize(&src[at..], first_line)?;
    let last_line = first_line + src[at..].matches('\n').count();
    let rules = Parser { toks, i: 0, last_line }.rules()?;
    if rules.is_empty() {
        return Err(Error::at(first_line, "no rules after the items marker"));
    }
    Ok(rules)
}
