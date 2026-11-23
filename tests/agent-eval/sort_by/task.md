Write a complete Stave program in one source file. It reads no input and exits with status 0.
Its standard output must be exactly the text under "Output" (every line ends with a newline).
Compute the values in the program; do not print them as literal text.

## Task

Define a copyable struct `Rec { id: u32, age: u32 }` and an insertion sort
`sort(xs: &mut [Rec], less: fn(&Rec, &Rec) -> bool)`. Sort the records (id, age) = (1, 30), (2, 25),
(3, 30), (4, 20), (5, 25), (6, 35) by age ascending with ties by id ascending, and print the ids;
then sort the result by age descending with ties by id ascending and print the ids. Pass the
orderings as closures.

## Output

```text
by age: 4 2 5 1 3 6
by age desc: 6 1 3 2 5 4
```
