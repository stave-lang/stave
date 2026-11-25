## Task

Write six named functions on `i64`: `add sub mul div rem pow` (pow: a raised to the b-th power by
repeated multiplication). For the operations `('+', 7, 5)`, `('-', 7, 5)`, `('*', 7, 5)`,
`('/', 7, 5)`, `('%', 7, 5)`, `('^', 2, 10)` (a `char` and two `i64`), first choose the function
from the `char` with a `match` that yields a function value, then call it. Print
`<a> <op> <b> = <result>`.

## Output

```text
7 + 5 = 12
7 - 5 = 2
7 * 5 = 35
7 / 5 = 1
7 % 5 = 2
2 ^ 10 = 1024
```
