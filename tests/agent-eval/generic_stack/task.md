Write a complete Stave program in one source file. It reads no input and exits with status 0.
Its standard output must be exactly the text under "Output" (every line ends with a newline).
Compute the values in the program; do not print them as literal text.

## Task

Write a generic `Stack<T>` struct backed by a `std.vec` vector with methods `new`, `push`, `pop`
(returns an `Option` of the value), `peek` (returns an `Option` of a reference to the top value),
and `len`. Use a `Stack<u32>`: push 10, 20, 30, print the top via `peek`, then pop until empty,
printing each value and finally `empty`. Then use a `Stack` of a non-Copy struct `Job` with fields
`id: u32` and `cost: u32`: push jobs (1, 5), (2, 8), (3, 2), pop one and print its id and cost, and
print the remaining length.

## Output

```text
top 30
pop 30
pop 20
pop 10
empty
job 3 cost 2
left 2
```
