## Task

Represent arithmetic expressions as an enum `Expr` with variants `Num(i64)`, `Add(usize, usize)`,
`Mul(usize, usize)`, `Neg(usize)`, where the `usize` payloads are indexes of other nodes in one
`std.vec` vector of nodes. Evaluate with a recursive function. Build the nodes in this order:
0 = Num(2), 1 = Num(3), 2 = Add(0, 1), 3 = Num(4), 4 = Neg(3), 5 = Mul(2, 4), 6 = Num(7),
7 = Add(5, 6), 8 = Mul(7, 7). Print the value of nodes 2, 5, 7, 8 as `node <i> = <value>`.

## Output

```text
node 2 = 5
node 5 = -20
node 7 = -13
node 8 = 169
```
