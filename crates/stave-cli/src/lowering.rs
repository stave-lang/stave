//! `stave lowering --target T [--op OP]` (contracts/cli.md): prints the lowering table of the
//! target's tier (`stave_target::lowering`), three header lines starting with `#`, then one line
//! per (op, type class) in table order:
//!
//! `op  types  class  cost  recipe[  | +feature: class cost recipe]...[  | note: ...][  | unverified]`
//!
//! A shift's types end in `by const` / `by vector`. A U line has `-` for cost and recipe. The op
//! and types columns are padded to the widest entry of the tier's whole table, so a line is the
//! same with or without `--op`. `--op` takes the IR name (`vec.gather`) or the name after its
//! first dot (`gather`). Always stage0 (`--compiler` is accepted and ignored).

use std::fmt::Write as _;
use std::io::Write;

use stave_target::Triple;
use stave_target::lowering::{Cell, Class, Entry, Tier};

use crate::args::Args;
use crate::{EXIT_OK, EXIT_USAGE, not_implemented};

pub(crate) fn run(a: &Args, out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    let triple = a.target.triple;
    let Some(tier) = Tier::of(triple) else {
        return not_implemented(
            &format!("lowering --target {triple}"),
            "the GPU lowering table (contracts/gpu-ops.md) is not generated yet",
            err,
        );
    };
    match render(triple, tier, a.op.as_deref()) {
        Ok(text) => {
            let _ = out.write_all(text.as_bytes());
            EXIT_OK
        }
        Err(msg) => {
            let _ = writeln!(err, "stave: {msg}\nrun `stave --help` for usage");
            EXIT_USAGE
        }
    }
}

fn matches(e: &Entry, op: &str) -> bool {
    e.op == op || e.op.split_once('.').is_some_and(|(_, rest)| rest == op)
}

fn types(e: &Entry) -> String {
    if e.form.is_empty() { e.ty.to_string() } else { format!("{} by {}", e.ty, e.form) }
}

fn cost_recipe(c: &Cell) -> (String, &'static str) {
    if c.class == Class::U { ("-".into(), "-") } else { (c.cost.to_string(), c.recipe) }
}

/// The printed table of `tier` for `triple`, filtered to `op` if given; an op with no entry is an
/// error.
pub fn render(triple: Triple, tier: Tier, op: Option<&str>) -> Result<String, String> {
    let entries = tier.entries();
    let wop = entries.iter().map(|e| e.op.len()).max().unwrap_or(0);
    let wty = entries.iter().map(|e| types(e).len()).max().unwrap_or(0);
    let mut o = String::new();
    let _ = writeln!(o, "# stave lowering --target {triple}: tier {}", tier.name());
    o.push_str(
        "# op, types, class (N native, M multi-instruction, U unavailable), cost (instructions), recipe\n",
    );
    o.push_str("# recipe placeholders: * lane size letter, # lane bits, % float letter, ? s or u\n");
    let mut any = false;
    for e in entries.iter().filter(|e| op.is_none_or(|op| matches(e, op))) {
        let Some(base) = e.baseline(tier) else { continue };
        any = true;
        let (cost, recipe) = cost_recipe(base);
        let _ = write!(o, "{:<wop$}  {:<wty$}  {}  {cost:>3}  {recipe}", e.op, types(e), base.class.name());
        for v in e.variants(tier) {
            let (cost, recipe) = cost_recipe(v);
            let _ = write!(o, "  | +{}: {} {cost} {recipe}", v.feature.unwrap_or(""), v.class.name());
            if !v.note.is_empty() {
                let _ = write!(o, " (note: {})", v.note);
            }
            if v.unverified {
                o.push_str(" (unverified)");
            }
        }
        if !base.note.is_empty() {
            let _ = write!(o, "  | note: {}", base.note);
        }
        if base.unverified {
            o.push_str("  | unverified");
        }
        o.push('\n');
    }
    match op {
        Some(op) if !any => Err(format!("no lowering entry for op `{op}` on {triple}")),
        _ => Ok(o),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_and_columns() {
        let t = Triple::Aarch64(stave_target::A64Ext::Base);
        let all = render(t, Tier::Neon, None).unwrap();
        let g = render(t, Tier::Neon, Some("gather")).unwrap();
        assert_eq!(g, render(t, Tier::Neon, Some("vec.gather")).unwrap());
        let lines: Vec<&str> = g.lines().filter(|l| !l.starts_with('#')).collect();
        assert!(!lines.is_empty() && lines.iter().all(|l| l.starts_with("vec.gather ")), "{g}");
        assert!(lines.iter().all(|l| all.lines().any(|a| a == *l)));
        assert!(lines[0].contains("  U    -  -"), "{}", lines[0]);
        assert!(
            render(t, Tier::Neon, Some("add"))
                .unwrap()
                .lines()
                .all(|l| l.starts_with('#') || l.starts_with("vec.add "))
        );
        assert!(
            render(t, Tier::Neon, Some("nosuch")).unwrap_err().contains("no lowering entry for op `nosuch`")
        );
        let shl = render(t, Tier::Neon, Some("vec.shl")).unwrap();
        assert!(shl.contains(" by const ") && shl.contains(" by vector "), "{shl}");
    }
}
