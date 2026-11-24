//! Compiler intrinsics, inline (contracts/intrinsics.md): `syscall0`–`syscall6`, `ccall0`–`ccall12`,
//! `cpuid`, `xgetbv`, `call_on_stack`, and the address intrinsics `thread_start_addr`, `static_addr`,
//! `main_addr`, `signal_handler_addr`, `site_table`, `pc_table`, `stack_table`, the CPU atomics
//! `atomic_{load,xchg,add,cas}_{u32,u64}` (sequentially consistent: `mov` / `xchg` / `lock xadd` /
//! `lock cmpxchg` on `[rcx]`), and `thread_clone` (a thread-creating system call whose child runs
//! `__stave_thread_start(arg)` on its new stack, then the `exit` system call), `cpu_init` (a
//! call of `runtime.cpu.init` when the module has it, else nothing), `thread_block` (`fs:[0]`),
//! and `copy_bytes` (a call of `std.slice.copy_raw` with the intrinsic's operands).

use stave_ir::intrinsics::{AtomicOp, Kind as IKind};
use stave_ir::layout::Shape;
use stave_ir::{Known, Operand, TypeId, Value};
use stave_obj::SymbolId;

use super::abi::{ARG_GPRS, CallAbi, Loc, Piece, RetLoc};
use super::func::*;
use super::{Res, bad, unsup};
use crate::enc::*;

const SYSCALL_ARGS: [Gpr; 6] = [Gpr::RDI, Gpr::RSI, Gpr::RDX, Gpr::R10, Gpr::R8, Gpr::R9];

impl Cg<'_, '_> {
    pub(crate) fn intrinsic(
        &mut self,
        kind: IKind,
        args: &[Operand],
        ty: Option<TypeId>,
        res: Option<Value>,
    ) -> Res<()> {
        let arity = |n: usize| {
            if args.len() == n {
                Ok(())
            } else {
                Err(bad(format!("intrinsic.{kind}: {} operands", args.len())))
            }
        };
        let addr = |cg: &mut Self, sym: Option<SymbolId>, what: &str| -> Res<()> {
            let sym = sym.ok_or_else(|| unsup(format!("intrinsic.{kind}: the program has no {what}")))?;
            let v = res.ok_or_else(|| bad("address intrinsic without a result"))?;
            let r = cg.alloc_g()?;
            cg.a.lea(Size::Q, r, Mem::rip_sym(sym, 0))?;
            cg.bind_g(v, r)
        };
        match kind {
            IKind::Syscall(k) => {
                arity(usize::from(k) + 1)?;
                for r in [RAX, RDI, RSI, RDX, Gpr::R8, Gpr::R9, RCX] {
                    self.reserve_g(r)?;
                }
                for (i, &o) in args.iter().enumerate().skip(1).rev() {
                    self.copy_to_g(o, SYSCALL_ARGS[i - 1])?;
                }
                self.copy_to_g(args[0], RAX)?;
                self.a.syscall();
                match res {
                    Some(v) => self.bind_g(v, RAX),
                    None => Ok(()),
                }
            }
            IKind::Ccall(k) => {
                let k = usize::from(k);
                arity(k + 1)?;
                let locs: Vec<Loc> = (0..k)
                    .map(|i| match ARG_GPRS.get(i) {
                        Some(&g) => Loc::Reg(Piece::G(g)),
                        None => Loc::Stack(8 * (i as u32 - 6)),
                    })
                    .collect();
                let stack = (8 * k.saturating_sub(6) as u32).div_ceil(16) * 16;
                let abi = CallAbi { args: locs, ret: RetLoc::Reg(Piece::G(RAX)), stack };
                self.call(Callee::Val(args[0]), &args[1..], &abi, res, true)
            }
            IKind::Atomic { op, wide } => {
                let w: &[u8] = if wide { &[0x48] } else { &[] };
                for r in [RAX, RCX, RDX] {
                    self.reserve_g(r)?;
                }
                let code: Vec<u8> = match op {
                    AtomicOp::Load => {
                        arity(1)?;
                        self.copy_to_g(args[0], RCX)?;
                        [w, &[0x8B, 0x01]].concat() // mov (e|r)ax, [rcx]
                    }
                    AtomicOp::Xchg | AtomicOp::Add => {
                        arity(2)?;
                        self.copy_to_g(args[1], RAX)?;
                        self.copy_to_g(args[0], RCX)?;
                        if op == AtomicOp::Xchg {
                            [w, &[0x87, 0x01]].concat() // xchg [rcx], (e|r)ax (implicitly locked)
                        } else {
                            [&[0xF0][..], w, &[0x0F, 0xC1, 0x01]].concat() // lock xadd [rcx], (e|r)ax
                        }
                    }
                    AtomicOp::Cas => {
                        arity(3)?;
                        self.copy_to_g(args[2], RDX)?;
                        self.copy_to_g(args[1], RAX)?;
                        self.copy_to_g(args[0], RCX)?;
                        [&[0xF0][..], w, &[0x0F, 0xB1, 0x11]].concat() // lock cmpxchg [rcx], (e|r)dx
                    }
                };
                self.a.raw(&code);
                match res {
                    Some(v) => self.bind_g(v, RAX),
                    None => Ok(()),
                }
            }
            IKind::ThreadClone => {
                // thread_clone(nr, a1..a5, arg): rax = nr, rdi rsi rdx r10 r8 = a1..a5, r9 = arg;
                // `syscall`; the child (rax == 0, on its new stack) calls __stave_thread_start(arg)
                // and ends with `exit` (60) of its status; the parent continues with the result.
                arity(7)?;
                let thread = self.cx.thread_sym.ok_or_else(|| {
                    unsup("intrinsic.thread_clone: the program has no `runtime.thread.entry`")
                })?;
                for r in [RAX, RDI, RSI, RDX, Gpr::R8, Gpr::R9, RCX] {
                    self.reserve_g(r)?;
                }
                self.copy_to_g(args[6], Gpr::R9)?;
                for (i, &o) in args.iter().enumerate().skip(1).take(5).rev() {
                    self.copy_to_g(o, SYSCALL_ARGS[i - 1])?;
                }
                self.copy_to_g(args[0], RAX)?;
                self.a.syscall();
                let parent = self.a.new_label();
                self.a.raw(&[0x48, 0x85, 0xC0]); // test rax, rax
                self.a.jcc(Cond::Ne, parent)?;
                self.a.raw(&[0x4C, 0x89, 0xCF]); // mov rdi, r9
                self.a.raw(&[0x31, 0xED]); // xor ebp, ebp
                self.a.raw(&[0x48, 0x83, 0xE4, 0xF0]); // and rsp, -16
                // only the child runs this: the parent keeps its upper halves
                if self.rel && self.ydirty {
                    self.a.vzeroupper();
                }
                self.a.call_sym(thread);
                self.a.raw(&[0x89, 0xC7]); // mov edi, eax
                self.a.raw(&[0xB8, 60, 0, 0, 0]); // mov eax, 60 (exit)
                self.a.syscall();
                self.a.ud2();
                self.a.bind(parent)?;
                match res {
                    Some(v) => self.bind_g(v, RAX),
                    None => Ok(()),
                }
            }
            IKind::CpuInit => {
                // a call of `runtime.cpu.init` if the program has it, else no code
                arity(0)?;
                let Some(g) = self.cx.m.known_body(Known::CpuInit) else { return Ok(()) };
                let gf = self.cx.m.func(g).ok_or_else(|| bad("callee out of range"))?;
                let abi = self.abi_of(&gf.params, gf.ret)?;
                let sym = self.func_sym(g)?;
                self.call(Callee::Sym(sym), &[], &abi, None, false)
            }
            IKind::CopyBytes => {
                // a call of `std.slice.copy_raw(dst, src, n)`, the intrinsic's operands as its
                // arguments
                arity(3)?;
                let g = self
                    .cx
                    .m
                    .known_body(Known::CopyRaw)
                    .ok_or_else(|| bad("intrinsic.copy_bytes without `std.slice.copy_raw`"))?;
                let gf = self.cx.m.func(g).ok_or_else(|| bad("callee out of range"))?;
                let abi = self.abi_of(&gf.params, gf.ret)?;
                let sym = self.func_sym(g)?;
                self.call(Callee::Sym(sym), args, &abi, None, false)
            }
            IKind::ThreadStartAddr => addr(self, self.cx.thread_sym, "`runtime.thread.entry`"),
            IKind::MainAddr => addr(self, self.cx.main_sym, "`main`"),
            IKind::SignalHandlerAddr => addr(self, self.cx.signal_sym, "`runtime.panic.on_signal`"),
            IKind::SiteTable => addr(self, Some(self.cx.site_table_sym), "site table"),
            IKind::PcTable => addr(self, Some(self.cx.pc_table_sym), "PC table"),
            IKind::StackTable => addr(self, Some(self.cx.stack_table_sym), "stack table"),
            IKind::StaticAddr => {
                arity(1)?;
                let k = self
                    .int_const(args[0])
                    .ok_or_else(|| unsup("intrinsic.static_addr of a non-constant index"))?;
                let sym = usize::try_from(k)
                    .ok()
                    .and_then(|k| self.cx.static_sym.get(k).copied())
                    .ok_or_else(|| bad("intrinsic.static_addr index out of range"))?;
                addr(self, Some(sym), "static")
            }
            IKind::Cpuid => {
                arity(2)?;
                let v = res.ok_or_else(|| bad("cpuid without a result"))?;
                let rt = ty.ok_or_else(|| bad("cpuid without a result type"))?;
                let offs: Vec<i64> = match &self.layout(rt)?.shape {
                    Shape::Fields(fs) if fs.len() == 4 => fs.iter().map(|f| f.0 as i64).collect(),
                    _ => return Err(bad("cpuid result must be a 4-tuple")),
                };
                for r in [RAX, RBX, RCX, RDX] {
                    self.reserve_g(r)?;
                }
                self.copy_to_g(args[1], RCX)?;
                self.copy_to_g(args[0], RAX)?;
                self.a.cpuid();
                let d = self.def_mem(v)?;
                for (r, off) in [RAX, RBX, RCX, RDX].into_iter().zip(offs) {
                    self.a.mov_mr(Size::D, disp(d, off)?, r)?;
                }
                Ok(())
            }
            IKind::Xgetbv => {
                arity(1)?;
                for r in [RAX, RCX, RDX] {
                    self.reserve_g(r)?;
                }
                self.copy_to_g(args[0], RCX)?;
                self.a.xgetbv();
                self.a.shift_ri(ShiftOp::Shl, Size::Q, RDX, 32)?;
                self.a.mov_rr(Size::D, RAX, RAX);
                self.a.alu_rr(AluOp::Or, Size::Q, RAX, RDX);
                match res {
                    Some(v) => self.bind_g(v, RAX),
                    None => Ok(()),
                }
            }
            IKind::ThreadBlock => {
                // the block's first word is its own address, so `fs:[0]` is the block
                arity(0)?;
                self.reserve_g(RAX)?;
                self.a.raw(&[0x64, 0x48, 0x8B, 0x04, 0x25, 0, 0, 0, 0]); // mov rax, fs:[0]
                match res {
                    Some(v) => self.bind_g(v, RAX),
                    None => Ok(()),
                }
            }
            IKind::CallOnStack => {
                arity(3)?;
                self.spill_caller_saved()?;
                self.reserve_g(RAX)?;
                self.reserve_g(RDI)?;
                self.copy_to_g(args[0], S0)?;
                self.copy_to_g(args[1], RDI)?;
                self.copy_to_g(args[2], RAX)?;
                self.below = self.below.max(8);
                self.a.push(RBX);
                self.a.mov_rr(Size::Q, RBX, RSP);
                self.a.mov_rr(Size::Q, RSP, RAX);
                if self.rel {
                    self.vclean();
                }
                self.a.call_rm(S0)?;
                self.ydirty = false;
                self.a.mov_rr(Size::Q, RSP, RBX);
                self.a.pop(RBX);
                match res {
                    Some(v) => self.bind_g(v, RAX),
                    None => Ok(()),
                }
            }
            IKind::SetThreadBlock | IKind::CpuBaseline | IKind::Ptx => {
                Err(unsup(format!("intrinsic.{kind} on x86-64")))
            }
        }
    }
}
