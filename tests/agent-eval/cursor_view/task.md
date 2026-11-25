## Task

Define a view struct `Cursor` that borrows a `&[u32]` and holds a position, with a constructor
function, a method `next` returning an `Option<u32>` (the next element, or none at the end), and
a method `rest` returning the not yet read elements as a slice. Also write
`longest_run(xs: &[u32]) -> &[u32]` returning the first longest run of equal adjacent elements.
With xs = `[1, 1, 2, 2, 2, 3, 3, 3, 3, 1, 4, 4]`: read three elements with `next` and print them,
print the sum of `rest`, then print the value and length of the longest run.

## Output

```text
next 1
next 1
next 2
rest sum 25
longest run: 3 x 4
```
