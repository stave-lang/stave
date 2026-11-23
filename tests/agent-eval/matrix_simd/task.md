Write a complete Stave program in one source file. It reads no input and exits with status 0.
Its standard output must be exactly the text under "Output" (every line ends with a newline).
Compute the values in the program; do not print them as literal text.

## Task

Multiply two 4x4 `i32` matrices stored row-major in `[i32; 16]` arrays with `i32x4` vectors: row i
of C is the sum over k of `A[i][k]` (splat to all lanes) times row k of B, with wrapping vector
arithmetic. A has rows (1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16); B has rows
(1, 0, 0, 1), (0, 2, 0, 0), (0, 0, 3, 0), (1, 0, 0, 4). Print the rows of C (space-separated) and
its trace.

## Output

```text
5 4 9 17
13 12 21 37
21 20 33 57
29 28 45 77
trace 127
```
