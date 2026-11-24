## Task

Define an error enum `MathErr` with variants `DivZero` and `Negative(i64)`. Write
`div(a: i64, b: i64) -> Result<i64, MathErr>` (error on b = 0), `isqrt(n: i64)` returning the
integer square root (floor) or `Negative(n)` for n < 0, and `calc(a, b)` that computes
`isqrt(div(a, b)) + 1`, passing errors on with `?`. Run calc on (100, 4), (7, 0), (-50, 2),
(99, 1) and print the value or the error as shown.

## Output

```text
calc(100, 4) = 6
calc(7, 0): division by zero
calc(-50, 2): negative -25
calc(99, 1) = 10
```
