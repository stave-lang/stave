//! Human and JSON forms of single diagnostics, and renderer errors.

mod common;

use stave_diag::{Diagnostic, LineIndex, RenderError, Span, emit_human, emit_json, render, spelled};

#[test]
fn human_form_with_related_and_fix() {
    let (files, file) = common::sample_files();
    let holes = [("expected", "i64"), ("found", "i32"), ("operand", "1")];
    let d = Diagnostic::new(
        "E0201",
        None,
        &holes,
        Some(Span::new(file, 37, 1)),
        &[("expected", Span::new(file, 13, 3))],
    )
    .expect("E0201");
    let want = "src/main.stv:2:19: error[E0201]: mismatched types: expected `i64`, found `i32`\n\
                \x20 src/main.stv:2:19: this is `i32`\n\
                \x20 src/main.stv:1:14: `i64` required by this\n\
                \x20 fix: convert explicitly: `1 as i64` (or `as%`, `as|`, `as?`, `as~` when lossy)\n\
                \x20 edit src/main.stv:2:19 (1 byte): \"1 as i64\"\n";
    assert_eq!(emit_human(&[d], &files).expect("human"), want);
}

#[test]
fn locationless_human_and_json() {
    let (files, _) = common::sample_files();
    let d =
        Diagnostic::new("E1302", Some("cc"), &[("major", "7"), ("minor", "5")], None, &[]).expect("E1302");
    assert_eq!(
        emit_human(std::slice::from_ref(&d), &files).expect("human"),
        "error[E1302]: `--compiler gpu`: no usable GPU (compute capability 7.5 < 8.6)\n\
         \x20 fix: use `--compiler auto` or `--compiler cpu`: stage1 needs compute capability 8.6 or higher\n"
    );
    assert_eq!(
        emit_json(&[d], &files).expect("json"),
        "{\"kind\":\"diagnostic\",\"code\":\"E1302\",\"severity\":\"error\",\
         \"message\":\"`--compiler gpu`: no usable GPU (compute capability 7.5 < 8.6)\",\"related\":[],\
         \"fix\":{\"description\":\"use `--compiler auto` or `--compiler cpu`: stage1 needs compute \
         capability 8.6 or higher\",\"applicability\":\"placeholder\",\"edits\":[]}}\n"
    );
}

#[test]
fn repeated_roles_print_in_given_order_and_missing_roles_are_left_out() {
    let (_, file) = common::sample_files();
    let roles = [("import", Span::new(file, 40, 1)), ("import", Span::new(file, 4, 1))];
    let d = Diagnostic::new("E0102", None, &[("module", "a.b")], Some(Span::new(file, 0, 2)), &roles)
        .expect("E0102");
    let labels: Vec<u32> = d.related().iter().map(|r| r.span.offset).collect();
    assert_eq!(labels, [0, 40, 4]);
    // E0501 without the `use` role: that label is not printed.
    let holes = [
        ("access", "borrow mutably"),
        ("place", "*v"),
        ("loan", "`&`"),
        ("loaned", "*v"),
        ("later", "later here"),
    ];
    let d = Diagnostic::new(
        "E0501",
        None,
        &holes,
        Some(Span::new(file, 0, 1)),
        &[("loan", Span::new(file, 4, 1))],
    )
    .expect("E0501");
    assert_eq!(d.related().len(), 2);
    assert!(
        Diagnostic::new(
            "E0501",
            None,
            &holes,
            Some(Span::new(file, 0, 1)),
            &[("bogus", Span::new(file, 4, 1))]
        )
        .is_err()
    );
}

#[test]
fn render_errors_instead_of_panics() {
    assert!(matches!(render("E9999", None, &[]), Err(RenderError::UnknownCode(_))));
    assert!(matches!(render("bogus", None, &[]), Err(RenderError::UnknownCode(_))));
    assert!(matches!(render("E0101", Some("nope"), &[("name", "x")]), Err(RenderError::UnknownCase { .. })));
    assert!(matches!(render("E0101", None, &[]), Err(RenderError::MissingHole { .. })));
    assert!(matches!(
        render("E0101", None, &[("name", "x"), ("zz", "1")]),
        Err(RenderError::UnknownHole { .. })
    ));
    assert!(matches!(
        render("E0101", None, &[("name", "x"), ("name", "y")]),
        Err(RenderError::DuplicateHole { .. })
    ));
    assert!(matches!(render("E0101", None, &[("name", "")]), Err(RenderError::BadValue { .. })));
    assert!(matches!(render("E0002", None, &[("size", "12a")]), Err(RenderError::BadValue { .. })));
    assert!(matches!(render("E1302", None, &[("reason", "made up")]), Err(RenderError::BadValue { .. })));
    // Escaped braces render as literal braces.
    let r = render("E0014", None, &[("path", "S")]).expect("E0014");
    assert_eq!(r.message, "struct literal or pattern after `S` needs `.{`");
    assert_eq!(r.fix.expect("fix").description, "write `S.{ ... }`");
}

#[test]
fn source_holes_collapse_whitespace_in_one_line_text() {
    // Every whitespace run (newlines included) becomes one space; the ends are trimmed.
    let r = render("E0006", None, &[("hole", " \t a +\n\r\n  b\x0B\x0C c \n")]).expect("E0006");
    assert_eq!(r.message, "f-string hole `{a + b c}` is not a name or field path");
    assert_eq!(
        r.fix.expect("fix").description,
        "bind the value to a local first (`let v = a + b c;`), then write `{v}` in the f-string"
    );
    // Edit replacements keep the source bytes unchanged.
    let holes = [("expected", "i64"), ("found", "i32"), ("operand", "f(\n  x)")];
    let fix = render("E0201", None, &holes).expect("E0201").fix.expect("fix");
    assert!(fix.description.starts_with("convert explicitly: `f( x) as i64`"), "{}", fix.description);
    assert_eq!(fix.edits[0].replacement, "f(\n  x) as i64");
    // Other hole kinds still refuse a newline.
    assert!(matches!(
        render("E0201", None, &[("expected", "i\n64"), ("found", "i32"), ("operand", "1")]),
        Err(RenderError::BadValue { .. })
    ));
}

#[test]
fn spelled_source_text_is_a_valid_value_of_every_one_line_hole_kind() {
    assert_eq!(spelled("'a'\n        .. 'd'"), "'a' .. 'd'");
    assert_eq!(spelled("  f(\r\n\tx)\x0B"), "f( x)");
    assert_eq!(spelled("x"), "x");
    let text = spelled("'a'\n  .. 'd'");
    let r = render("E0219", None, &[("place", &text)]).expect("E0219");
    assert_eq!(
        r.message,
        "`for` cannot iterate the array or slice `'a' .. 'd'` by value: iterate a reference to it"
    );
    // The renderer still refuses the raw value, and a value that is empty after the rewrite.
    assert!(matches!(render("E0219", None, &[("place", "'a'\n.. 'd'")]), Err(RenderError::BadValue { .. })));
    assert!(matches!(
        render("E0219", None, &[("place", &spelled(" \n "))]),
        Err(RenderError::BadValue { .. })
    ));
}

#[test]
fn line_index() {
    let li = LineIndex::new(b"ab\n\ncd");
    assert_eq!(li.line_count(), 3);
    assert_eq!(li.line_col(0), Some((1, 1)));
    assert_eq!(li.line_col(2), Some((1, 3)));
    assert_eq!(li.line_col(3), Some((2, 1)));
    assert_eq!(li.line_col(4), Some((3, 1)));
    assert_eq!(li.line_col(6), Some((3, 3)), "end of file");
    assert_eq!(li.line_col(7), None);
    assert_eq!(LineIndex::new(b"").line_col(0), Some((1, 1)));
}

#[test]
fn every_entry_is_sorted_and_findable() {
    let all = stave_diag::entries();
    assert!(all.windows(2).all(|w| w[0].code < w[1].code));
    for e in all {
        assert_eq!(stave_diag::entry(e.code).map(|x| x.code), Some(e.code));
    }
}
