## Task

Write `isqrt(n: u64) -> u64` (floor of the square root, by binary search) with a precondition
`n <= 1000000000000` and a postcondition stating that the result r satisfies r * r <= n and
(r + 1) * (r + 1) > n. Print isqrt of 0, 1, 15, 16, 1000000, 999999999999. Then write
`push_small(v: &mut Vec<u32>, x: u32)` (a `std.vec` vector) with the precondition x < 100 and a
postcondition that the length grew by one; push 5, 50, 99 and print the length.

## Output

```text
isqrt(0) = 0
isqrt(1) = 1
isqrt(15) = 3
isqrt(16) = 4
isqrt(1000000) = 1000
isqrt(999999999999) = 999999
len 3
```
