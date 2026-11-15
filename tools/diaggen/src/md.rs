//! Reads the code rows of the human tables (`contracts/diagnostic-codes.md` and the GPU tables it
//! delegates E11xx and E12xx to).

/// One table row with a code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    pub code: String,
    /// The `Rule` cell starts with "retired".
    pub retired: bool,
    /// The `Severity` cell, when the table has that column.
    pub severity: Option<String>,
    /// File the row came from, for messages.
    pub source: String,
}

/// Splits a table line into trimmed cells (`| a | b |` → `["a", "b"]`). Cells after one that
/// holds a `|` inside backticks are split too; only the leading cells (code, severity) and the
/// start of the rule cell are used.
fn cells(line: &str) -> Vec<&str> {
    let inner = line.trim().trim_start_matches('|').trim_end_matches('|');
    inner.split('|').map(str::trim).collect()
}

pub fn is_code(s: &str) -> bool {
    s.len() == 5 && s.starts_with('E') && s[1..].bytes().all(|b| b.is_ascii_digit())
}

fn is_separator(line: &str) -> bool {
    let t = line.trim();
    t.starts_with('|') && t.chars().all(|c| matches!(c, '|' | '-' | ':' | ' '))
}

/// Code rows of one markdown file: lines `| EXXXX | ... |` under a table header.
pub fn rows(source: &str, text: &str) -> Vec<Row> {
    let lines: Vec<&str> = text.lines().collect();
    let mut header: Vec<String> = Vec::new();
    let mut out = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if !line.trim_start().starts_with('|') {
            continue;
        }
        if lines.get(i + 1).is_some_and(|next| is_separator(next)) {
            header = cells(line).into_iter().map(|c| c.to_ascii_lowercase()).collect();
            continue;
        }
        let cs = cells(line);
        let Some(code) = cs.first().copied().filter(|c| is_code(c)) else {
            continue;
        };
        let col = |name: &str| header.iter().position(|h| h == name).and_then(|i| cs.get(i).copied());
        out.push(Row {
            code: code.to_string(),
            retired: col("rule").is_some_and(|r| r.to_ascii_lowercase().starts_with("retired")),
            severity: col("severity").map(str::to_string),
            source: source.to_string(),
        });
    }
    out
}
