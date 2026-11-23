Write a complete Stave program in one source file. It reads no input and exits with status 0.
Its standard output must be exactly the text under "Output" (every line ends with a newline).
Compute the values in the program; do not print them as literal text.

## Task

Define an enum `Op` with the variants `Add`, `Sub`, `Mul`, `Div`, `Rem`, a function that applies an
`Op` to two `i64` values, and a function that returns the lower-case name of an `Op`. Apply each
operation, in the order above, to a = 20 and b = 6 and print `<name> <result>`.

## Output

```text
add 26
sub 14
mul 120
div 3
rem 2
```
