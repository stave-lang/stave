Write a complete Stave program in one source file. It reads no input and exits with status 0.
Its standard output must be exactly the text under "Output" (every line ends with a newline).
Compute the values in the program; do not print them as literal text.

## Task

Start from n = 27 (`u64`). Repeat until n is 1: if n is even, n = n / 2, else n = 3 * n + 1.
Print the number of steps taken and the largest value n reached.

## Output

```text
steps 111
peak 9232
```
