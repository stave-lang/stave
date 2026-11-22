Write a complete Stave program in one source file. It reads no input and exits with status 0.
Its standard output must be exactly the text under "Output" (every line ends with a newline).
Compute the values in the program; do not print them as literal text.

## Task

Define a struct `P { x: i32, y: i32, vx: i32, vy: i32 }` and store 10 particles in a `std.soa`
struct of arrays: particle i has x = i, y = 2 * i, vx = 3, vy = -1. Run 4 steps; each step adds vx
to x and vy to y for all particles, using `i32x4` loads and stores on the columns for full groups
of four and scalar code for the rest. Print the sum of the x column, the sum of the y column, and
particle 3 read back as one record.

## Output

```text
sum x 165
sum y 50
row 3: (15, 2)
```
