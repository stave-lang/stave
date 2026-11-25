## Task

Write `is_sorted(xs: &[u32]) -> bool` and `lower_bound(xs: &[u32], key: u32) -> usize` (the first
index whose element is >= key, by binary search) with the precondition that `xs` is sorted, the
postcondition that the result is at most `xs.len()`, and a deny clause stating that it does not
allocate, do I/O, or lock. Write `range_of(xs: &[u32], lo: u32, hi: u32) -> &[u32]` returning the
sub-slice of elements in [lo, hi), with the precondition lo <= hi and xs sorted. With
xs = `[1, 3, 3, 5, 8, 13, 21, 34]` print lower_bound of 5, 4, 40, 0, then the length and sum of
range_of(3, 14).

## Output

```text
lower_bound(5) = 3
lower_bound(4) = 3
lower_bound(40) = 8
lower_bound(0) = 0
in [3, 14): 5 values, sum 32
```
