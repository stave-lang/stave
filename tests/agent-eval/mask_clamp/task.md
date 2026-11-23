Write a complete Stave program in one source file. It reads no input and exits with status 0.
Its standard output must be exactly the text under "Output" (every line ends with a newline).
Compute the values in the program; do not print them as literal text.

## Task

Fill an `i32` array of 21 elements with `(i * 7) % 23 - 11` for i = 0..20. In place, with `i32x4`
vectors (a chunk loop with a masked tail), clamp every element to the range [-5, 5] and count the
elements that were changed, counting with vector compares and masks. Print the array
(space-separated) and the count.

## Output

```text
-5 -4 3 5 -5 1 5 -5 -1 5 -5 -3 4 5 -5 2 5 -5 0 5 -5
changed 12
```
