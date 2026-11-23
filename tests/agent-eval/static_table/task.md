Write a complete Stave program in one source file. It reads no input and exits with status 0.
Its standard output must be exactly the text under "Output" (every line ends with a newline).
Compute the values in the program; do not print them as literal text.

## Task

Declare a global immutable table of the first eight primes (`u32`), a constant `SCALE = 10`, a
global `MAX_WEIGHT` initialized to 19 * SCALE, and a global atomic call counter (`std.atomic`).
Write `weight(i: usize) -> u32` that counts the call and returns prime i times `SCALE`. Sum
`weight(i)` for all eight entries and print the sum, `MAX_WEIGHT`, and the number of calls.

## Output

```text
primes total 770
max weight 190
calls 8
```
