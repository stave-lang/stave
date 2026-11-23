Write a complete Stave program in one source file. It reads no input and exits with status 0.
Its standard output must be exactly the text under "Output" (every line ends with a newline).
Compute the values in the program; do not print them as literal text.

## Task

Define a struct `Rect` with `u32` fields `w` and `h`, and methods `area`, `perimeter`, and
`is_square` (returns `bool`). For the rectangles 3x4, 5x5 and 2x7 (in this order) print one line
each: `<w>x<h> area <area> perimeter <perimeter> square <true|false>`.

## Output

```text
3x4 area 12 perimeter 14 square false
5x5 area 25 perimeter 20 square true
2x7 area 14 perimeter 18 square false
```
