## Task

Write a stack machine over `i64` with the instructions (an enum) `Push(i64) Add Dup Swap Over Dec
Jz(usize) Jmp(usize) Print Halt`: Add pops two and pushes the sum; Dup copies the top; Swap swaps
the top two; Over pushes a copy of the second; Dec subtracts 1 from the top; Jz pops and jumps when
the value is 0; Jmp jumps; Print pops and prints the value on its own line; Halt stops. `run`
returns the number of instructions executed (Halt included) or an error: stack underflow at a pc,
or a jump target outside the program. Run these three programs:
1. `Push(0) Push(5) Dup Jz(12) Dup Print Swap Over Add Swap Dec Jmp(2) Swap Print Halt`
2. `Push(1) Add`
3. `Jmp(7)`

## Output

```text
5
4
3
2
1
15
ok after 57 steps
error: stack underflow at 1
error: bad jump to 7 at 0
```
