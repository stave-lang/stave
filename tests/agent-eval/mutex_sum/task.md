Write a complete Stave program in one source file. It reads no input and exits with status 0.
Its standard output must be exactly the text under "Output" (every line ends with a newline).
Compute the values in the program; do not print them as literal text.

## Task

Declare a global `std.sync` mutex holding a `u64` total. Add every i in 0..1000 to it from a
parallel `std.thread` loop. Print the total. Then make a local mutex holding a `u64`, and in a
structured thread scope spawn two threads: one adds 1..=10 to it, the other adds 11..=20, one
locked addition per number. Print the local total.

## Output

```text
par_for total 499500
scope total 210
```
