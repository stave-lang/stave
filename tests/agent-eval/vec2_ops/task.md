Write a complete Stave program in one source file. It reads no input and exits with status 0.
Its standard output must be exactly the text under "Output" (every line ends with a newline).
Compute the values in the program; do not print them as literal text.

## Task

Define a 2D vector struct `V2` with `i32` fields `x` and `y`, make it copyable, and implement the
operators `+`, `-` (binary), `*` (component-wise), and unary `-` for it, plus a method `dot`. With
a = (3, -2) and b = (1, 3), print the results below. Values print as `(x, y)`.

## Output

```text
a + b = (4, 1)
a - b = (2, -5)
a * b = (3, -6)
-a = (-3, 2)
a + b + b = (5, 4)
a . b = -3
```
