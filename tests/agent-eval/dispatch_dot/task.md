Write a complete Stave program in one source file. It reads no input and exits with status 0.
Its standard output must be exactly the text under "Output" (every line ends with a newline).
Compute the values in the program; do not print them as literal text.

## Task

Write a dot product of two `i32` slices of equal length with wrapping arithmetic in two versions:
a portable one with `i32x4` and one with `i32x8` that needs AVX2. Combine them in a `dispatch`
function that picks the AVX2 version at run time when the CPU has it. With a[i] = i - 40 and
b[i] = i % 7 + 1 for i in 0..100, print the dispatched result and whether it equals the portable
version's result.

## Output

```text
dot 3900
same true
```
