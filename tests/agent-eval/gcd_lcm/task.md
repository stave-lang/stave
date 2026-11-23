Write a complete Stave program in one source file. It reads no input and exits with status 0.
Its standard output must be exactly the text under "Output" (every line ends with a newline).
Compute the values in the program; do not print them as literal text.

## Task

Write `gcd(a: u64, b: u64) -> u64` (Euclid's algorithm with a loop) and `lcm` using it. For the
pairs (12, 18), (17, 5), (100, 75) print `gcd(<a>, <b>) = <g>, lcm = <l>`.

## Output

```text
gcd(12, 18) = 6, lcm = 36
gcd(17, 5) = 1, lcm = 85
gcd(100, 75) = 25, lcm = 300
```
