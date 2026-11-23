Write a complete Stave program in one source file. It reads no input and exits with status 0.
Its standard output must be exactly the text under "Output" (every line ends with a newline).
Compute the values in the program; do not print them as literal text.

## Task

Define a view struct `Best { name: &str, score: u32 }` and a function
`offer(b: &mut Best, name: &str, score: u32)` that stores the name and score in `b` when the score
is higher than the current one (declare what the function stores). With the names `alice bob
carol dave` and the scores 72, 55, 91, 40, start from `Best` ("nobody", 0), offer every entry and
print the best. Also collect the names with a score over 50 in a `std.vec` vector of `&str` and
print them. Then build a second `Best` from only dave and bob, print it, and print the name of
the better of the two `Best` values with a function returning a `&str` taken from them.

## Output

```text
best: carol 91
over 50: alice bob carol
second: bob 55
winner: carol
```
