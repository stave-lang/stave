Write a complete Stave program in one source file. It reads no input and exits with status 0.
Its standard output must be exactly the text under "Output" (every line ends with a newline).
Compute the values in the program; do not print them as literal text.

## Task

Define a struct `Buffers` with two fields `front` and `back`, both a `std.vec` vector of `u32`.
Start with `front` empty and push 1, 2, 3, 4, 5 into `back`. Print both lengths. Swap the two
fields with `std.mem`, print both lengths again. Then take the `front` vector out of the struct,
leaving an empty vector in its place, and print the length and sum of the taken vector and the
new length of `front`.

## Output

```text
front 0 back 5
front 5 back 0
taken 5 sum 15
front 0
```
