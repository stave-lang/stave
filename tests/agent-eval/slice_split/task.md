Write a complete Stave program in one source file. It reads no input and exits with status 0.
Its standard output must be exactly the text under "Output" (every line ends with a newline).
Compute the values in the program; do not print them as literal text.

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
