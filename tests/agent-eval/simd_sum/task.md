Write a complete Stave program in one source file. It reads no input and exits with status 0.
Its standard output must be exactly the text under "Output" (every line ends with a newline).
Compute the values in the program; do not print them as literal text.

## Task

Fill a `u32` array of 103 elements with 1, 2, ..., 103. With explicit `u32x4` vectors (a chunk loop
over the array with a scalar tail) compute the sum, the number of elements greater than 50
(count with a vector compare), and the maximum. Print them.

## Output

```text
sum 5356
above 50: 53
max 103
```
