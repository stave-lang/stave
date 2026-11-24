## Task

Define a trait `Shape` with a method `area(&self) -> u64`, and two structs: `Square` (field `side`)
and `Tri` (fields `base`, `height`; area is base * height / 2, integer division). Implement the
trait for both. Write one generic function that returns the total area of a slice of any `Shape`
type. With the squares of side 3 and 5 and the triangles 4x6 and 10x3, print each area and the
two totals.

## Output

```text
square 3: 9
square 5: 25
triangle 4x6: 12
triangle 10x3: 15
squares total 34
triangles total 27
```
