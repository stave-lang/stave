## Task

Write two generic functions over any copyable type with operator traits:
`sum_all(xs: &[T], zero: T) -> T` (needs `+`) and `dot(a: &[T], b: &[T], zero: T) -> T` (needs `+`
and `*`). Define `Money { cents: u64 }` with `+`, and `Mod7 { v: u32 }` with `+` and `*` modulo 7.
Print: sum_all of `[1, 2, 3, 4, 5]` (`u32`); sum_all of `[4, -9, 2]` (`i64`); sum_all of the Money
values 1050, 299, 885 cents printed as dollars.cents; dot of `[2, 3]` and `[5, -4]` (`i64`); dot of
the Mod7 values `[3, 5, 6]` and `[4, 2, 6]`.

## Output

```text
u32 sum 15
i64 sum -3
money 22.34
i64 dot -2
mod7 dot 2
```
