Write a complete Stave program in one source file. It reads no input and exits with status 0.
Its standard output must be exactly the text under "Output" (every line ends with a newline).
Compute the values in the program; do not print them as literal text.

## Task

Given the four words `alpha`, `beta`, `gamma`, `delta` (string literals in an array), build a
`std.string` String that joins them with `", "` and print it. Then print the number of words and
the byte length of the joined string. Then build and print a second String joining the words in
reverse order with the same separator.

## Output

```text
alpha, beta, gamma, delta
4 words, 25 bytes
delta, gamma, beta, alpha
```
