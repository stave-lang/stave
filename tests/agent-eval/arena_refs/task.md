## Task

Define a copyable struct `P { x: i64, y: i64 }` and create a `std.arena` arena of 1024 bytes. In
an inner block that owns a `std.vec` vector of shared references `&P`: for i = 0, 1, 2, 3, 4 allocate the
point (i + 1, i * i - 1) in the arena and push the reference the arena returns into the vector;
then print the sum of the x and of the y coordinates, read through the vector. After the block,
reset the arena, allocate one point (42, 0) and print its x.

## Output

```text
sum x 15
sum y 25
after reset: 42
```
