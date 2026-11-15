//! Hand-written encodings checked against objdump.

use super::check;
use crate::enc::*;

const RAX: Gpr = Gpr::RAX;
const RCX: Gpr = Gpr::RCX;
const RDX: Gpr = Gpr::RDX;
const RBX: Gpr = Gpr::RBX;
const RSP: Gpr = Gpr::RSP;
const RBP: Gpr = Gpr::RBP;
const RSI: Gpr = Gpr::RSI;
const RDI: Gpr = Gpr::RDI;
const R8: Gpr = Gpr::R8;
const R9: Gpr = Gpr::R9;
const R10: Gpr = Gpr::R10;
const R11: Gpr = Gpr::R11;
const R12: Gpr = Gpr::R12;
const R13: Gpr = Gpr::R13;
const R15: Gpr = Gpr::R15;

fn x(n: u8) -> Xmm {
    Xmm::new(n).unwrap()
}

#[test]
fn moves() {
    check(
        |a| {
            a.mov_rr(Size::Q, RAX, RCX);
            a.mov_rr(Size::B, RSI, RDI);
            a.mov_rr(Size::W, R8, RAX);
            a.mov_rr(Size::D, R15, R9);
            a.mov_ri(Size::Q, RAX, 5)?;
            a.mov_ri(Size::Q, R9, -1)?;
            a.mov_ri(Size::Q, RAX, 0x1_2345_6789)?;
            a.mov_ri(Size::Q, R12, 0xffff_ffff)?;
            a.mov_ri(Size::B, RSP, 1)?;
            a.mov_ri(Size::W, RCX, 0xffff)?;
            a.mov_ri(Size::D, R10, -2)?;
            a.mov_rm(Size::Q, RAX, Mem::base(RSP, 8))?;
            a.mov_rm(Size::D, R12, Mem::base(R13, 0))?;
            a.mov_mr(Size::B, Mem::index(RAX, R12, Scale::S4, -4), RBX)?;
            a.mov_mr(Size::Q, Mem::base(RBP, -0x1000), R15)?;
            a.mov_mr(Size::W, Mem::base(R12, 0x7f), RSI)?;
            a.mov_mi(Size::Q, Mem::base(RAX, 0), -2)?;
            a.mov_mi(Size::W, Mem::base(RAX, 2), 0x1234)?;
            a.mov_mi(Size::B, Mem::base(RSP, -128), 0xff)?;
            a.mov_rm(Size::Q, RAX, Mem::Sib { base: None, index: None, disp: 0x1000 })?;
            a.mov_rm(Size::Q, RAX, Mem::Sib { base: None, index: Some((RCX, Scale::S8)), disp: 0x10 })?;
            a.mov_rm(Size::Q, R11, Mem::index(R13, R13, Scale::S1, 0))?;
            Ok(())
        },
        &[
            "mov rax,rcx",
            "mov sil,dil",
            "mov r8w,ax",
            "mov r15d,r9d",
            "mov eax,0x5",
            "mov r9,0xffffffffffffffff",
            "movabs rax,0x123456789",
            "mov r12d,0xffffffff",
            "mov spl,0x1",
            "mov cx,0xffff",
            "mov r10d,0xfffffffe",
            "mov rax,QWORD PTR [rsp+0x8]",
            "mov r12d,DWORD PTR [r13+0x0]",
            "mov BYTE PTR [rax+r12*4-0x4],bl",
            "mov QWORD PTR [rbp-0x1000],r15",
            "mov WORD PTR [r12+0x7f],si",
            "mov QWORD PTR [rax],0xfffffffffffffffe",
            "mov WORD PTR [rax+0x2],0x1234",
            "mov BYTE PTR [rsp-0x80],0xff",
            "mov rax,QWORD PTR ds:0x1000",
            "mov rax,QWORD PTR [rcx*8+0x10]",
            "mov r11,QWORD PTR [r13+r13*1+0x0]",
        ],
    );
}

#[test]
fn lea_and_rip() {
    check(
        |a| {
            let end = a.new_label();
            a.lea(Size::Q, RAX, Mem::index(RBX, RCX, Scale::S2, 0))?;
            a.lea(Size::D, R8, Mem::base(RSP, 0x100))?;
            // trailing immediates are accounted for in the rip displacement
            a.lea(Size::Q, RAX, Mem::rip_label(end, 0))?; // 7 bytes, at 0x0d
            a.mov_mi(Size::D, Mem::rip_label(end, 0), 7)?; // 10 bytes, at 0x14
            a.alu_mi(AluOp::Cmp, Size::B, Mem::rip_label(end, 2), 1)?; // 7 bytes, at 0x1e
            a.movs(Prec::F64, x(3), Mem::rip_label(end, -8))?; // 8 bytes, at 0x25
            a.bind(end)?; // 0x2d
            a.nop();
            Ok(())
        },
        &[
            "lea rax,[rbx+rcx*2]",
            "lea r8d,[rsp+0x100]",
            "lea rax,[rip+0x19]",
            "mov DWORD PTR [rip+0xf],0x7",
            "cmp BYTE PTR [rip+0xa],0x1",
            "movsd xmm3,QWORD PTR [rip+0xfffffffffffffff8]",
            "nop",
        ],
    );
}

#[test]
fn alu() {
    check(
        |a| {
            a.alu_rr(AluOp::Add, Size::Q, RAX, RCX);
            a.alu_rr(AluOp::Sub, Size::D, R8, R9);
            a.alu_rr(AluOp::Xor, Size::B, RAX, RSI);
            a.alu_rr(AluOp::Adc, Size::Q, RDX, RBX);
            a.alu_rr(AluOp::Sbb, Size::W, RCX, RDX);
            a.alu_rr(AluOp::And, Size::Q, R15, R12);
            a.alu_rr(AluOp::Or, Size::D, RAX, RAX);
            a.alu_rr(AluOp::Cmp, Size::Q, RSP, RBP);
            a.alu_ri(AluOp::Add, Size::Q, RSP, 16)?;
            a.alu_ri(AluOp::Sub, Size::Q, RSP, 0x1000)?;
            a.alu_ri(AluOp::And, Size::Q, RAX, -16)?;
            a.alu_ri(AluOp::Cmp, Size::D, RAX, 0xffff_ffff)?;
            a.alu_ri(AluOp::Or, Size::W, RDX, 0x8000)?;
            a.alu_ri(AluOp::Cmp, Size::B, RDI, 0x80)?;
            a.alu_ri(AluOp::Xor, Size::Q, R10, 0x7fff_ffff)?;
            a.alu_rm(AluOp::Add, Size::Q, RAX, Mem::base(RDI, 0x7f))?;
            a.alu_mr(AluOp::Sub, Size::D, Mem::base(RDI, 0x80), RCX)?;
            a.alu_mi(AluOp::Cmp, Size::Q, Mem::base(RSP, 0), 1)?;
            a.alu_mi(AluOp::Add, Size::D, Mem::index(RAX, RBX, Scale::S8, 0x12345), 0x10000)?;
            a.test_rr(Size::Q, RAX, RAX);
            a.test_rr(Size::B, RSI, RCX);
            a.test_ri(Size::B, RCX, 1)?;
            a.test_ri(Size::D, Mem::base(RAX, 0), 0x100)?;
            a.test_ri(Size::Q, R9, -1)?;
            a.test_mr(Size::Q, Mem::base(RBX, 8), RAX)?;
            Ok(())
        },
        &[
            "add rax,rcx",
            "sub r8d,r9d",
            "xor al,sil",
            "adc rdx,rbx",
            "sbb cx,dx",
            "and r15,r12",
            "or eax,eax",
            "cmp rsp,rbp",
            "add rsp,0x10",
            "sub rsp,0x1000",
            "and rax,0xfffffffffffffff0",
            "cmp eax,0xffffffff",
            "or dx,0x8000",
            "cmp dil,0x80",
            "xor r10,0x7fffffff",
            "add rax,QWORD PTR [rdi+0x7f]",
            "sub DWORD PTR [rdi+0x80],ecx",
            "cmp QWORD PTR [rsp],0x1",
            "add DWORD PTR [rax+rbx*8+0x12345],0x10000",
            "test rax,rax",
            "test sil,cl",
            "test cl,0x1",
            "test DWORD PTR [rax],0x100",
            "test r9,0xffffffffffffffff",
            "test QWORD PTR [rbx+0x8],rax",
        ],
    );
}

#[test]
fn mul_div_shift() {
    check(
        |a| {
            a.imul_rr(Size::Q, RAX, RCX)?;
            a.imul_rr(Size::D, R10, Mem::base(R11, 4))?;
            a.imul_rri(Size::Q, RAX, RBX, 10)?;
            a.imul_rri(Size::Q, RAX, RBX, 1000)?;
            a.imul_rri(Size::W, RDX, Mem::base(RSI, 0), -3)?;
            a.unary(UnaryOp::Mul, Size::Q, RCX)?;
            a.unary(UnaryOp::Div, Size::Q, R8)?;
            a.unary(UnaryOp::Idiv, Size::D, RSI)?;
            a.unary(UnaryOp::Neg, Size::Q, RAX)?;
            a.unary(UnaryOp::Not, Size::B, RDX)?;
            a.unary(UnaryOp::Imul, Size::Q, Mem::base(RAX, 0))?;
            a.unary(UnaryOp::Div, Size::B, RDI)?;
            a.cqo();
            a.cdq();
            a.shift_ri(ShiftOp::Shl, Size::Q, RAX, 3)?;
            a.shift_ri(ShiftOp::Sar, Size::D, R9, 31)?;
            a.shift_ri(ShiftOp::Rol, Size::B, RAX, 1)?;
            a.shift_ri(ShiftOp::Ror, Size::W, RCX, 15)?;
            a.shift_ri(ShiftOp::Shr, Size::Q, Mem::base(RSP, 8), 63)?;
            a.shift_cl(ShiftOp::Shr, Size::Q, RDX)?;
            a.shift_cl(ShiftOp::Sar, Size::B, RSI)?;
            a.shift_cl(ShiftOp::Shl, Size::D, Mem::base(RDI, 0))?;
            Ok(())
        },
        &[
            "imul rax,rcx",
            "imul r10d,DWORD PTR [r11+0x4]",
            "imul rax,rbx,0xa",
            "imul rax,rbx,0x3e8",
            "imul dx,WORD PTR [rsi],0xfffd",
            "mul rcx",
            "div r8",
            "idiv esi",
            "neg rax",
            "not dl",
            "imul QWORD PTR [rax]",
            "div dil",
            "cqo",
            "cdq",
            "shl rax,0x3",
            "sar r9d,0x1f",
            "rol al,0x1",
            "ror cx,0xf",
            "shr QWORD PTR [rsp+0x8],0x3f",
            "shr rdx,cl",
            "sar sil,cl",
            "shl DWORD PTR [rdi],cl",
        ],
    );
}

const CC: [(Cond, &str); 16] = [
    (Cond::O, "o"),
    (Cond::No, "no"),
    (Cond::B, "b"),
    (Cond::Ae, "ae"),
    (Cond::E, "e"),
    (Cond::Ne, "ne"),
    (Cond::Be, "be"),
    (Cond::A, "a"),
    (Cond::S, "s"),
    (Cond::Ns, "ns"),
    (Cond::P, "p"),
    (Cond::Np, "np"),
    (Cond::L, "l"),
    (Cond::Ge, "ge"),
    (Cond::Le, "le"),
    (Cond::G, "g"),
];

#[test]
fn conditions() {
    let regs = [RAX, RSI, R8, R15];
    let mut want: Vec<String> = CC.iter().map(|(_, n)| format!("j{n} 0x0")).collect();
    for (i, (_, n)) in CC.iter().enumerate() {
        let r = regs[i % 4];
        want.push(format!("set{n} {}", r.name(Size::B)));
        want.push(format!("cmov{n} {},rcx", r.name(Size::Q)));
    }
    let mut a = Asm::new();
    let top = a.new_label();
    a.bind(top).unwrap();
    for (c, _) in CC {
        a.jcc(c, top).unwrap();
    }
    for (i, (c, _)) in CC.iter().enumerate() {
        a.setcc(*c, regs[i % 4]);
        a.cmov(*c, Size::Q, regs[i % 4], RCX).unwrap();
    }
    super::check_bytes(&a.finish().unwrap().code, &want);
    for (c, _) in CC {
        assert_eq!(c.negate().negate(), c);
        assert_ne!(c.negate(), c);
    }
}

#[test]
fn control_flow() {
    check(
        |a| {
            let top = a.new_label();
            let end = a.new_label();
            a.bind(top)?;
            a.jcc(Cond::O, top)?; // 0
            a.jcc(Cond::C, top)?; // 6
            a.jmp(top)?; // 12
            a.call(top)?; // 17
            a.jcc(Cond::Ne, end)?; // 22
            a.jmp(end)?; // 28
            a.bind(end)?; // 33
            a.call_rm(RAX)?;
            a.call_rm(Mem::base(RAX, 8))?;
            a.call_rm(R11)?;
            a.jmp_rm(R11)?;
            a.jmp_rm(Mem::index(RAX, RCX, Scale::S8, 0))?;
            a.push(RBX);
            a.push(R12);
            a.pop(R15);
            a.pop(RBP);
            a.push_i(16);
            a.push_i(1000);
            a.ret();
            a.leave();
            a.syscall();
            a.ud2();
            a.int3();
            a.nop();
            a.cpuid();
            a.xgetbv();
            Ok(())
        },
        &[
            "jo 0x0",
            "jb 0x0",
            "jmp 0x0",
            "call 0x0",
            "jne 0x21",
            "jmp 0x21",
            "call rax",
            "call QWORD PTR [rax+0x8]",
            "call r11",
            "jmp r11",
            "jmp QWORD PTR [rax+rcx*8]",
            "push rbx",
            "push r12",
            "pop r15",
            "pop rbp",
            "push 0x10",
            "push 0x3e8",
            "ret",
            "leave",
            "syscall",
            "ud2",
            "int3",
            "nop",
            "cpuid",
            "xgetbv",
        ],
    );
}

#[test]
fn symbol_relocs() {
    use stave_obj::{RelocKind, SymbolId};
    let mut a = Asm::new();
    let s = SymbolId(7);
    a.call_sym(s); // reloc at 1
    a.jmp_sym(s); // reloc at 6
    a.lea(Size::Q, RDI, Mem::rip_sym(s, 0)).unwrap(); // 48 8d 3d: reloc at 13
    a.mov_mi(Size::D, Mem::rip_sym(s, 4), 1).unwrap(); // c7 05: reloc at 19, 4 trailing
    a.mov_r_sym(RAX, s, 16); // 48 b8: reloc at 29
    let f = a.finish().unwrap();
    let r: Vec<_> = f.relocs.iter().map(|r| (r.offset, r.kind, r.addend)).collect();
    assert_eq!(
        r,
        [
            (1, RelocKind::X86Pc32, -4),
            (6, RelocKind::X86Pc32, -4),
            (13, RelocKind::X86Pc32, -4),
            (19, RelocKind::X86Pc32, 4 - 4 - 4),
            (29, RelocKind::Abs64, 16),
        ]
    );
    super::check_bytes(
        &f.code,
        &["call 0x5", "jmp 0xa", "lea rdi,[rip+0x0]", "mov DWORD PTR [rip+0x0],0x1", "movabs rax,0x0"]
            .map(String::from),
    );
}

#[test]
fn extensions() {
    check(
        |a| {
            a.movzx(Size::D, RAX, Size::B, RCX)?;
            a.movzx(Size::Q, RAX, Size::W, Mem::base(RDI, 0))?;
            a.movzx(Size::Q, RAX, Size::D, RCX)?;
            a.movzx(Size::D, R9, Size::B, RSI)?;
            a.movzx(Size::W, RAX, Size::B, RDX)?;
            a.movsx(Size::Q, RAX, Size::B, RSI)?;
            a.movsx(Size::Q, RAX, Size::D, RCX)?;
            a.movsx(Size::D, RAX, Size::W, Mem::base(RAX, 0))?;
            a.movsx(Size::Q, R12, Size::D, Mem::base(R12, 4))?;
            Ok(())
        },
        &[
            "movzx eax,cl",
            "movzx rax,WORD PTR [rdi]",
            "mov eax,ecx",
            "movzx r9d,sil",
            "movzx ax,dl",
            "movsx rax,sil",
            "movsxd rax,ecx",
            "movsx eax,WORD PTR [rax]",
            "movsxd r12,DWORD PTR [r12+0x4]",
        ],
    );
}

#[test]
fn sse2_scalar() {
    check(
        |a| {
            a.sse(SseOp::Add, Prec::F64, x(0), x(1))?;
            a.sse(SseOp::Mul, Prec::F32, x(8), x(15))?;
            a.sse(SseOp::Div, Prec::F64, x(2), Mem::base(RAX, 8))?;
            a.sse(SseOp::Sqrt, Prec::F64, x(0), x(0))?;
            a.sse(SseOp::Sub, Prec::F32, x(1), x(2))?;
            a.sse(SseOp::Min, Prec::F64, x(3), x(4))?;
            a.sse(SseOp::Max, Prec::F32, x(5), Mem::base(R12, 0))?;
            a.movs(Prec::F64, x(0), Mem::base(RSP, 0))?;
            a.movs_store(Prec::F32, Mem::base(RAX, 4), x(9))?;
            a.movs(Prec::F64, x(1), x(2))?;
            a.ucomis(Prec::F64, x(0), x(1))?;
            a.ucomis(Prec::F32, x(10), Mem::base(RBX, 0))?;
            a.comis(Prec::F64, x(0), x(1))?;
            a.cvtsi2s(Prec::F64, x(0), Size::Q, RAX)?;
            a.cvtsi2s(Prec::F32, x(1), Size::D, Mem::base(RAX, 0))?;
            a.cvtts2si(Prec::F64, Size::Q, RAX, x(0))?;
            a.cvtts2si(Prec::F32, Size::D, R8, x(12))?;
            a.cvts2si(Prec::F32, Size::D, RCX, x(1))?;
            a.cvts2s(Prec::F64, x(0), x(1))?;
            a.cvts2s(Prec::F32, x(2), Mem::base(RAX, 0))?;
            a.mov_xg(Size::Q, x(0), RAX)?;
            a.mov_gx(Size::Q, RAX, x(0))?;
            a.mov_xg(Size::D, x(3), R8)?;
            a.mov_gx(Size::D, R11, x(14))?;
            a.logic_p(PLogic::Xor, Prec::F64, x(0), x(0))?;
            a.logic_p(PLogic::And, Prec::F32, x(1), Mem::base(RAX, 0))?;
            a.logic_p(PLogic::Andn, Prec::F64, x(1), x(2))?;
            a.logic_p(PLogic::Or, Prec::F32, x(1), x(2))?;
            a.movaps(x(0), x(1));
            Ok(())
        },
        &[
            "addsd xmm0,xmm1",
            "mulss xmm8,xmm15",
            "divsd xmm2,QWORD PTR [rax+0x8]",
            "sqrtsd xmm0,xmm0",
            "subss xmm1,xmm2",
            "minsd xmm3,xmm4",
            "maxss xmm5,DWORD PTR [r12]",
            "movsd xmm0,QWORD PTR [rsp]",
            "movss DWORD PTR [rax+0x4],xmm9",
            "movsd xmm1,xmm2",
            "ucomisd xmm0,xmm1",
            "ucomiss xmm10,DWORD PTR [rbx]",
            "comisd xmm0,xmm1",
            "cvtsi2sd xmm0,rax",
            "cvtsi2ss xmm1,DWORD PTR [rax]",
            "cvttsd2si rax,xmm0",
            "cvttss2si r8d,xmm12",
            "cvtss2si ecx,xmm1",
            "cvtsd2ss xmm0,xmm1",
            "cvtss2sd xmm2,DWORD PTR [rax]",
            "movq xmm0,rax",
            "movq rax,xmm0",
            "movd xmm3,r8d",
            "movd r11d,xmm14",
            "xorpd xmm0,xmm0",
            "andps xmm1,XMMWORD PTR [rax]",
            "andnpd xmm1,xmm2",
            "orps xmm1,xmm2",
            "movaps xmm0,xmm1",
        ],
    );
}

#[test]
fn bit_scans_and_x87() {
    check(
        |a| {
            a.bsr(Size::Q, RAX, RCX)?;
            a.bsr(Size::D, R11, Mem::base(RBP, -8))?;
            a.bsf(Size::Q, R10, R15)?;
            a.bsf(Size::W, RDX, RSI)?;
            a.fld(Prec::F64, Mem::base(RBP, -16))?;
            a.fld(Prec::F32, Mem::base(R12, 4))?;
            a.fprem();
            a.fnstsw_ax();
            a.fstp(Prec::F64, Mem::base(RSP, 0))?;
            a.fstp(Prec::F32, Mem::base(R13, -4))?;
            a.fstp_st0();
            Ok(())
        },
        &[
            "bsr rax,rcx",
            "bsr r11d,DWORD PTR [rbp-0x8]",
            "bsf r10,r15",
            "bsf dx,si",
            "fld QWORD PTR [rbp-0x10]",
            "fld DWORD PTR [r12+0x4]",
            "fprem",
            "fnstsw ax",
            "fstp QWORD PTR [rsp]",
            "fstp DWORD PTR [r13-0x4]",
            "fstp st(0)",
        ],
    );
    let mut a = Asm::new();
    assert!(a.bsr(Size::B, RAX, RCX).is_err());
    assert!(a.bsf(Size::B, RAX, RCX).is_err());
}

#[test]
fn overflow_trap_sequence() {
    // add with signed and unsigned overflow traps, as the debug backend emits them
    check(
        |a| {
            let trap = a.new_label();
            a.alu_rr(AluOp::Add, Size::Q, RAX, RCX); // 0, 3 bytes
            a.jcc(Cond::O, trap)?; // 3
            a.alu_rr(AluOp::Add, Size::D, RAX, RCX); // 9, 2 bytes
            a.jcc(Cond::C, trap)?; // 11
            a.unary(UnaryOp::Mul, Size::Q, RCX)?; // 17, 3 bytes
            a.jcc(Cond::O, trap)?; // 20
            a.ret(); // 26
            a.bind(trap)?; // 27
            a.ud2();
            Ok(())
        },
        &["add rax,rcx", "jo 0x1b", "add eax,ecx", "jb 0x1b", "mul rcx", "jo 0x1b", "ret", "ud2"],
    );
}

#[test]
fn invalid_operands() {
    let mut a = Asm::new();
    let bad_index = Mem::Sib { base: Some(RAX), index: Some((RSP, Scale::S1)), disp: 0 };
    assert!(a.mov_rm(Size::Q, RAX, bad_index).is_err());
    assert!(a.lea(Size::W, RAX, Mem::base(RAX, 0)).is_err());
    assert!(a.lea(Size::B, RAX, Mem::base(RAX, 0)).is_err());
    assert!(a.mov_ri(Size::B, RAX, 256).is_err());
    assert!(a.mov_ri(Size::W, RAX, -0x8001).is_err());
    assert!(a.mov_ri(Size::D, RAX, 0x1_0000_0000).is_err());
    assert!(a.alu_ri(AluOp::Add, Size::Q, RAX, 0x8000_0000).is_err());
    assert!(a.mov_mi(Size::Q, Mem::base(RAX, 0), 0x8000_0000).is_err());
    assert!(a.imul_rr(Size::B, RAX, RCX).is_err());
    assert!(a.imul_rri(Size::B, RAX, RCX, 1).is_err());
    assert!(a.cmov(Cond::E, Size::B, RAX, RCX).is_err());
    assert!(a.shift_ri(ShiftOp::Shl, Size::D, RAX, 32).is_err());
    assert!(a.shift_ri(ShiftOp::Shl, Size::B, RAX, 8).is_err());
    assert!(a.movzx(Size::D, RAX, Size::D, RCX).is_err());
    assert!(a.movzx(Size::B, RAX, Size::B, RCX).is_err());
    assert!(a.movsx(Size::W, RAX, Size::D, RCX).is_err());
    assert!(a.cvtsi2s(Prec::F64, Xmm::new(0).unwrap(), Size::W, RAX).is_err());
    assert!(a.cvtts2si(Prec::F64, Size::B, RAX, Xmm::new(0).unwrap()).is_err());
    assert!(a.mov_xg(Size::W, Xmm::new(0).unwrap(), RAX).is_err());
    assert!(a.align(3).is_err());
    assert_eq!(a.offset(), 0, "failed instructions emit nothing");
    // labels
    let mut b = Asm::new();
    let l = b.new_label();
    b.bind(l).unwrap();
    assert_eq!(b.bind(l), Err(EncError::LabelRebound(l)));
    let mut c = Asm::new();
    assert!(c.jmp(l).is_err(), "label from another buffer");
    assert!(c.lea(Size::Q, RAX, Mem::rip_label(l, 0)).is_err());
    let m = c.new_label();
    c.jmp(m).unwrap();
    assert_eq!(c.finish().unwrap_err(), EncError::LabelUnbound(m));
    assert!(Gpr::new(16).is_none() && Xmm::new(16).is_none());
}

/// The instructions of a branch-free `select`: `cmov` on 64- and 32-bit registers, float bits
/// through GPRs, narrow tail loads and stores, and the indexed word loop of a long aggregate.
#[test]
fn select_sequences() {
    check(
        |a| {
            a.test_rr(Size::D, RAX, RAX);
            a.cmov(Cond::Ne, Size::Q, R11, R10)?;
            a.cmov(Cond::Ne, Size::Q, R11, Mem::base(RBP, -0x28))?;
            a.cmov(Cond::Ne, Size::D, R11, RBX)?;
            a.mov_gx(Size::D, R11, x(3))?;
            a.mov_gx(Size::Q, R10, x(14))?;
            a.mov_xg(Size::Q, x(2), R11)?;
            a.mov_xg(Size::D, x(9), R11)?;
            a.movzx(Size::D, R11, Size::W, Mem::base(RBP, -0x13))?;
            a.movzx(Size::D, RBX, Size::B, Mem::base(RBP, -0x11))?;
            a.mov_mr(Size::B, Mem::base(RBP, -0x8), R11)?;
            a.mov_mr(Size::W, Mem::base(RBP, -0x6), R11)?;
            a.mov_rm(Size::Q, R11, Mem::index(RBP, R10, Scale::S8, -0x208))?;
            a.mov_mr(Size::Q, Mem::index(RBP, R10, Scale::S8, -0x8), R11)?;
            a.mov_ri(Size::Q, RBX, 0)?;
            Ok(())
        },
        &[
            "test eax,eax",
            "cmovne r11,r10",
            "cmovne r11,QWORD PTR [rbp-0x28]",
            "cmovne r11d,ebx",
            "movd r11d,xmm3",
            "movq r10,xmm14",
            "movq xmm2,r11",
            "movd xmm9,r11d",
            "movzx r11d,WORD PTR [rbp-0x13]",
            "movzx ebx,BYTE PTR [rbp-0x11]",
            "mov BYTE PTR [rbp-0x8],r11b",
            "mov WORD PTR [rbp-0x6],r11w",
            "mov r11,QWORD PTR [rbp+r10*8-0x208]",
            "mov QWORD PTR [rbp+r10*8-0x8],r11",
            "mov ebx,0x0",
        ],
    );
}
