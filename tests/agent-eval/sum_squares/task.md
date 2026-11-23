Write a complete Stave program in one source file. It reads no input and exits with status 0.
Its standard output must be exactly the text under "Output" (every line ends with a newline).
Compute the values in the program; do not print them as literal text.

## Task

Print three sums, each computed with a loop:
- `squares`: the sum of i * i for i from 1 to 100, as a `u64`;
- `cubes`: the sum of i * i * i for i from 1 to 20, as a `u32`;
- `odds`: the sum of the odd numbers below 50, as an `i32`, computed with a `while` loop.

## Output

```text
squares 338350
cubes 44100
odds 625
```
