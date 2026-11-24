## Task

Write `split_at_max(xs: &[i32])` returning two slices of `xs` as a tuple: the elements before the
first maximum, and the rest (starting at that maximum). Also write
`bigger(a: &i32, b: &i32) -> &i32` returning a reference to the larger value (`a` on ties). With
xs = `[4, -2, 9, 1, 9, -5, 3]` print the sum and length of both parts, then the bigger of the first
and last element.

## Output

```text
left sum 2 len 2
right sum 17 len 5
bigger 4
```
