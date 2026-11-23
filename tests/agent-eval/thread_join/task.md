Write a complete Stave program in one source file. It reads no input and exits with status 0.
Its standard output must be exactly the text under "Output" (every line ends with a newline).
Compute the values in the program; do not print them as literal text.

## Task

Fill a `u64` array of 1000 elements with 1, 2, ..., 1000. Sum the first and second halves
concurrently with `std.thread` (`join` of two closures), and print both sums and the total. Then
double every element in parallel with `std.thread` and print the new total.

## Output

```text
left 125250
right 375250
total 500500
doubled 1001000
```
