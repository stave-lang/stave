## Task

Define a struct `Noisy { id: u32 }` whose destructor prints `drop <id>`, and a function
`consume(n: Noisy)` that prints `consume <id>`. In `main`, in this order: create Noisy 1 as `a`;
in an inner block create Noisy 2 and Noisy 3 and print `inner end`; create a mutable `d` = Noisy 4,
then assign Noisy 5 to `d`; pass `d` to `consume`; create Noisy 6; drop `a` explicitly; print
`main end`; return 0. Do not print anything else.

## Output

```text
inner end
drop 3
drop 2
drop 4
consume 5
drop 5
drop 1
main end
drop 6
```
