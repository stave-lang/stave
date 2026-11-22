Write a complete Stave program in one source file. It reads no input and exits with status 0.
Its standard output must be exactly the text under "Output" (every line ends with a newline).
Compute the values in the program; do not print them as literal text.

## Task

Define a trait `Better` with `better(&self, other: &Self) -> bool`. Implement it for `u32` (larger
is better) and for a struct `Player` with fields `id: u32` and `score: u32` (higher score is
better; with equal scores the lower id is better). Write one generic function
`best<T: Better>(xs: &[T]) -> usize` returning the index of the best element (the first one when
several are equally good). Apply it to the numbers `[4, 19, 7, 19, 3]` and to the players
(id, score) = (3, 70), (7, 90), (2, 90), (5, 40).

## Output

```text
best number 19 at 1
best player 2 with 90 at 2
```
