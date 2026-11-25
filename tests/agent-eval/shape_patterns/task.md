## Task

Define `enum Shape { Circle { r: u32 }, Rect { w: u32, h: u32 }, Tri { a: u32, b: u32, c: u32 }, Dot }`.
Describe each shape with a `match` using struct patterns and guards:
a circle with r = 0 is `degenerate circle`, else `circle r=<r>`; a rect with w = h is
`square <w>`, else `rect <w>x<h>`; a triangle with three equal sides is `equilateral <a>`, with
two equal sides `isosceles`, else `scalene`; a dot is `dot`. Describe, in order: Circle r=5,
Circle r=0, Rect 4x4, Rect 3x8, Tri 6,6,6, Tri 5,5,8, Tri 3,4,5, Dot.

## Output

```text
circle r=5
degenerate circle
square 4
rect 3x8
equilateral 6
isosceles
scalene
dot
```
