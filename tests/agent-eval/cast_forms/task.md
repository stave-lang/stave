## Task

For each `i64` value 300, -5, 42 (in this order) convert it to `u8` in three ways and print
`<value>: sat <s> wrap <w> checked <c>`: `s` saturates to the `u8` range, `w` keeps the low 8 bits
(two's complement), and `c` is the value if it fits in `u8`, else the word `none`.

## Output

```text
300: sat 255 wrap 44 checked none
-5: sat 0 wrap 251 checked none
42: sat 42 wrap 42 checked 42
```
