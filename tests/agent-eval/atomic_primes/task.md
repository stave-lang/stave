Write a complete Stave program in one source file. It reads no input and exits with status 0.
Its standard output must be exactly the text under "Output" (every line ends with a newline).
Compute the values in the program; do not print them as literal text.

## Task

Declare two global `std.atomic` `u64` counters. In a parallel `std.thread` loop over 2..10000, for
each prime add 1 to the first counter and the prime to the second. Print both.

## Output

```text
primes below 10000: 1229
sum 5736396
```
