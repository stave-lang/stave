Write a complete Stave program in one source file. It reads no input and exits with status 0.
Its standard output must be exactly the text under "Output" (every line ends with a newline).
Compute the values in the program; do not print them as literal text.

## Task

Fill a `std.vec` vector of `i64` with the 20 values `(i * 37) % 101 - 50` for i = 0, 1, ..., 19.
Then print its length, smallest value, largest value, sum, and the number of negative values.

## Output

```text
n 20
min -50
max 47
sum -30
negatives 10
```
