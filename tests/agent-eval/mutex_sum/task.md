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
