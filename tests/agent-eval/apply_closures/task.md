Write a complete Stave program in one source file. It reads no input and exits with status 0.
Its standard output must be exactly the text under "Output" (every line ends with a newline).
Compute the values in the program; do not print them as literal text.

## Task

Write `apply(xs: &mut [i32], f: fn(i32) -> i32)` that replaces each element by `f` of it, and
`compose(f: fn(i32) -> i32, g: fn(i32) -> i32, x: i32) -> i32` returning g(f(x)). Start with the
array `[1, 2, 3, 4, 5]` and a local `offset = 10`. First apply a closure computing `x * 2 + offset`
and print the array. Then apply a closure that returns `x - 1` and counts its calls in a local
counter; print the array and the count. Last, print compose of two named functions `double`
(x * 2) and `inc` (x + 1) at x = 7. Arrays print as space-separated numbers.

## Output

```text
after first: 12 14 16 18 20
after second: 11 13 15 17 19
calls 5
compose 15
```
