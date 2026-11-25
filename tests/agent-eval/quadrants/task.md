## Task

Classify the `(i32, i32)` points (0, 0), (0, 5), (-3, 0), (2, 3), (-1, 4), (-2, -2), (5, -1) with a
`match`: `origin`; `y axis` (x is 0); `x axis` (y is 0); else the quadrant `I` (x > 0, y > 0),
`II` (x < 0, y > 0), `III` (x < 0, y < 0), or `IV`. Print `(<x>, <y>) <class>` per point.

## Output

```text
(0, 0) origin
(0, 5) y axis
(-3, 0) x axis
(2, 3) I
(-1, 4) II
(-2, -2) III
(5, -1) IV
```
