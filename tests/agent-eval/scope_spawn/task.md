## Task

In one structured thread scope, spawn three threads: one sums the even numbers in 0..100 into a
local `evens`, one sums the odd numbers in 0..100 into a local `odds`, and one finds the maximum
of the array `[13, 97, 42, 8]` into a local `best` (all `u64`). After the scope print the three
results.

## Output

```text
evens 2450
odds 2500
max 97
```
