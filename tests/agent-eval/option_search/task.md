Write a complete Stave program in one source file. It reads no input and exits with status 0.
Its standard output must be exactly the text under "Output" (every line ends with a newline).
Compute the values in the program; do not print them as literal text.

## Task

Write a function `first_multiple(xs: &[u32], k: u32)` that returns an `Option` holding the index of
the first element of `xs` that is a multiple of `k`, or no value. With xs = `[7, 11, 14, 22, 35]`,
query k = 11, 5, 4, 7 in this order and print `<k>: index <i>` or `<k>: none`.

## Output

```text
11: index 1
5: index 4
4: none
7: index 0
```
