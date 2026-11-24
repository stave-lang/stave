//! What the printer refuses: a module some line of whose text no reader accepts in steps 1 and 2
//! of checked-ir.md "Loading" (an id that names no line, a key with no spelling, a decl the
//! module does not record, a mark the structure pass rejects). A module free of these prints as
//! text that lexes, parses and passes the structure pass; whether it verifies is step 3's
//! question, which the printer does not ask (malformed but spellable modules print, so tests can
//! show any op).

use super::StackInput;
use crate::ir::{Body, FuncId, Module, Op, Operand, Site, Term};
use crate::types::{GenericArg, TypeId};
use crate::verify::{VerifyError, module_faults};
use crate::witness::Summary;

/// The faults of `m` (with its stack section) that make its text unloadable.
pub(crate) fn faults(m: &Module, stack: Option<&StackInput>) -> Vec<VerifyError> {
    let module_err = |msg: String| VerifyError { func: None, block: None, inst: None, msg };
    let mut errs: Vec<VerifyError> = module_faults(m).into_iter().map(module_err).collect();
    let nt = m.types.len();
    let ty_bad = |t: TypeId| t.index() >= nt;
    for (i, k) in m.types.keys().iter().enumerate() {
        let mut bad = None;
        k.for_each_component(|c| {
            if ty_bad(c) {
                bad = Some(c);
            }
        });
        if let Some(c) = bad {
            errs.push(module_err(format!("t{i}: component t{} out of range", c.0)));
        }
        let fields = m.types.shape(TypeId(i as u32)).into_iter().flat_map(|s| s.variants.iter().flatten());
        if let Some(f) = fields.copied().find(|f| ty_bad(*f)) {
            errs.push(module_err(format!("t{i}: shape field t{} out of range", f.0)));
        }
    }
    for (i, s) in m.statics.iter().enumerate() {
        if ty_bad(s.ty) {
            errs.push(module_err(format!("static {i}: type t{} out of range", s.ty.0)));
        }
        for r in &s.refs {
            let ok = match r.target {
                crate::DataTarget::Str(x) => x.index() < m.strings.len(),
                crate::DataTarget::Static(j) => (j as usize) < m.statics.len(),
            };
            if !ok {
                errs.push(module_err(format!("static {i}: address word at {} has no target", r.offset)));
            }
        }
    }
    for name in &m.intrinsics {
        if name.is_empty() || !name.bytes().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'_')
        {
            errs.push(module_err(format!("intrinsic name {name:?} is not of `[a-z0-9_]+`")));
        }
    }
    for (k, f) in m.funcs.iter().enumerate() {
        let func = Some(FuncId(k as u32));
        let mut err =
            |inst: Option<u32>, msg: String| errs.push(VerifyError { func, block: None, inst, msg });
        let targs = f.type_args.iter().filter_map(|a| match a {
            GenericArg::Type(t) => Some(*t),
            GenericArg::Const(_) => None,
        });
        if let Some(t) = f.params.iter().copied().chain(targs).chain(f.ret).find(|t| ty_bad(*t)) {
            err(None, format!("signature type t{} out of range", t.0));
        }
        if f.name_site != Site::NONE && f.name_site.index() >= m.sites.len() {
            err(None, format!("name site {} out of range", f.name_site.0));
        }
        if let Some(kn) = f.known
            && kn.path() != f.path
        {
            err(None, format!("`known` {kn:?} on a fn whose path is not its item's"));
        }
        let rows = m.stated.get(k).and_then(|s| s.as_deref()).is_some_and(has_rows);
        if rows && f.body.is_none() {
            err(None, "claims of a fn without a body".to_string());
        }
        if let Some(b) = &f.body {
            body_faults(m, b, &mut err);
        }
    }
    if let Some(s) = stack {
        errs.extend(stack_faults(m, s).into_iter().map(module_err));
    }
    errs
}

fn has_rows(s: &Summary) -> bool {
    !s.pre.is_empty() || !s.post.is_empty()
}

fn body_faults(m: &Module, b: &Body, err: &mut dyn FnMut(Option<u32>, String)) {
    if b.blocks.is_empty() {
        err(None, "a body without a block".to_string());
        return;
    }
    let (nv, nt) = (b.values.len(), m.types.len());
    let operand_bad = |o: &Operand| match *o {
        Operand::Value(v) => v.index() >= nv,
        Operand::Const(c) => c.index() >= b.consts.len(),
        Operand::Str(s) => s.index() >= m.strings.len(),
        Operand::Func(g) => g.index() >= m.funcs.len(),
        Operand::Type(t) => t.index() >= nt,
    };
    if b.values.iter().any(|v| v.ty.index() >= nt) {
        err(None, "a value type out of range".to_string());
    }
    if b.block_params.iter().any(|v| v.index() >= nv) {
        err(None, "a block param out of range".to_string());
    }
    for i in 0..b.insts.len() {
        let at = Some(i as u32);
        let named = match b.insts.op[i] {
            Op::Intrinsic(id) => id.index() < m.intrinsics.len(),
            op => op.name().is_some(),
        };
        if !named {
            err(at, "an op with no name".to_string());
        }
        if b.insts.ty.get(i).copied().flatten().is_some_and(|t| t.index() >= nt)
            || b.insts.result.get(i).copied().flatten().is_some_and(|v| v.index() >= nv)
        {
            err(at, "a result or result type out of range".to_string());
        }
        let span = b.insts.args.get(i).copied().unwrap_or_default();
        if b.operands
            .get(span.start as usize..(span.start + span.len) as usize)
            .is_none_or(|os| os.iter().any(operand_bad))
        {
            err(at, "an operand out of range".to_string());
        }
        if let Some(s) = b.insts.site.get(i).filter(|s| **s != Site::NONE && s.index() >= m.sites.len()) {
            err(at, format!("site {} out of range", s.0));
        }
    }
    for bd in &b.blocks {
        let targets: Vec<_> = match bd.term {
            Term::Br(t) => vec![t],
            Term::Cbr { cond, then, els } => {
                if operand_bad(&cond) {
                    err(None, "a terminator operand out of range".to_string());
                }
                vec![then, els]
            }
            Term::Switch { scrut, cases, default } => {
                if operand_bad(&scrut) {
                    err(None, "a terminator operand out of range".to_string());
                }
                let cs = b.cases.get(cases.start as usize..(cases.start + cases.len) as usize);
                let mut ts: Vec<_> = cs.unwrap_or(&[]).iter().map(|c| c.target).collect();
                ts.push(default);
                ts
            }
            Term::Ret(o) => {
                if o.is_some_and(|o| operand_bad(&o)) {
                    err(None, "a terminator operand out of range".to_string());
                }
                vec![]
            }
            Term::Unreachable => vec![],
        };
        for t in targets {
            let args = b.operands.get(t.args.start as usize..(t.args.start + t.args.len) as usize);
            if t.block.index() >= b.blocks.len() || args.is_none_or(|os| os.iter().any(operand_bad)) {
                err(None, format!("a branch to b{} out of range", t.block.0));
            }
        }
    }
}

/// The stack section's rules of "Sections" 9: nodes and fns named exist, edges ascending by
/// (from, to), each fn's nodes ascending.
fn stack_faults(m: &Module, s: &StackInput) -> Vec<String> {
    let n = s.nodes.len() as u32;
    let mut errs = Vec::new();
    if s.edges.iter().any(|e| e.from >= n || e.to >= n) {
        errs.push("stack section: an edge names no node".to_string());
    }
    if !s.edges.windows(2).all(|w| (w[0].from, w[0].to) < (w[1].from, w[1].to)) {
        errs.push("stack section: edges not ascending by (from, to)".to_string());
    }
    if s.nodes_of_func.len() > m.funcs.len() {
        errs.push("stack section: nodes of a fn the module does not have".to_string());
    }
    for ns in &s.nodes_of_func {
        if ns.iter().any(|&x| x >= n) || !ns.windows(2).all(|w| w[0] < w[1]) {
            errs.push("stack section: a fn's nodes out of range or not ascending".to_string());
        }
    }
    errs
}
