Write a complete Stave program in one source file. It reads no input and exits with status 0.
Its standard output must be exactly the text under "Output" (every line ends with a newline).
Compute the values in the program; do not print them as literal text.

## Task

Define a struct `Noisy { id: u32 }` whose destructor prints `drop <id>`. Loop i from 1 to 5: create
a Noisy with id i at the start of each iteration; when i is 2 go to the next iteration; when i is
4 leave the loop; otherwise print `body <i>`. After the loop print `after`. Then call a function
`make(id: u32) -> Noisy` that prints `made <id>` and returns a new Noisy, with id 9; print `end`
and return 0.

## Output

```text
body 1
drop 1
drop 2
body 3
drop 3
drop 4
after
made 9
end
drop 9
```
