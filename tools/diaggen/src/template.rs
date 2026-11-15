//! Template syntax: `{hole}` inserts a hole, `{{` and `}}` are literal braces.

/// One piece of a parsed template.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Segment {
    Text(String),
    Hole(String),
}

/// Parses `template`. Errors: an unclosed `{`, an empty or malformed hole name, a lone `}`.
pub fn parse(template: &str) -> Result<Vec<Segment>, String> {
    let mut out = Vec::new();
    let mut text = String::new();
    let mut rest = template;
    while let Some(i) = rest.find(['{', '}']) {
        text.push_str(&rest[..i]);
        let tail = &rest[i..];
        if let Some(after) = tail.strip_prefix("{{") {
            text.push('{');
            rest = after;
        } else if let Some(after) = tail.strip_prefix("}}") {
            text.push('}');
            rest = after;
        } else if let Some(after) = tail.strip_prefix('{') {
            let end = after.find('}').ok_or_else(|| format!("unclosed `{{` in template {template:?}"))?;
            let name = &after[..end];
            if !is_hole_name(name) {
                return Err(format!("bad hole name {name:?} in template {template:?}"));
            }
            if !text.is_empty() {
                out.push(Segment::Text(std::mem::take(&mut text)));
            }
            out.push(Segment::Hole(name.to_string()));
            rest = &after[end + 1..];
        } else {
            return Err(format!("lone `}}` in template {template:?} (write `}}}}`)"));
        }
    }
    text.push_str(rest);
    if !text.is_empty() {
        out.push(Segment::Text(text));
    }
    Ok(out)
}

/// Hole names: lower-case ASCII letter, then letters, digits, `_`.
pub fn is_hole_name(s: &str) -> bool {
    let mut b = s.bytes();
    matches!(b.next(), Some(c) if c.is_ascii_lowercase())
        && b.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'_')
}

/// The holes `template` uses, in order of appearance.
pub fn holes(template: &str) -> Result<Vec<String>, String> {
    Ok(parse(template)?
        .into_iter()
        .filter_map(|s| match s {
            Segment::Hole(h) => Some(h),
            Segment::Text(_) => None,
        })
        .collect())
}
