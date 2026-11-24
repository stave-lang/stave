## Task

Compute n! for n = 1, 2, 3, ... in `u64`, detecting overflow before it traps: use the full
128-bit product of the multiplication. Print the largest factorial that fits in `u64` and the
first n that overflows. Then print the largest n whose factorial fits in `u32`, found with a
checked conversion from the `u64` value.

## Output

```text
20! = 2432902008176640000
21! overflows u64
largest u32 factorial: 12! = 479001600
```
