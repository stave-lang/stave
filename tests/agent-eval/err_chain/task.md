## Task

An inventory holds items (id, stock, price): (1, 10, 25), (2, 3, 100), (3, 50, 7). Define
`LookupErr { NoSuchItem(u32) }` and `OrderErr { Lookup(u32), OutOfStock { id: u32, want: u32, have: u32 }, TooLarge(u64) }`.
`find(items, id) -> Result<usize, LookupErr>`; `place(items, id, qty) -> Result<u64, OrderErr>`
converts the lookup error with `map_err`, then checks stock, then rejects a cost
(qty * price, `u64`) above 250, and only then takes the stock. `batch(items, orders)` places
each order with `?` and returns the total cost. Place the single orders (id, qty) (1, 4), (2, 5),
(9, 1), (2, 3), (3, 20), (1, 6), (1, 1), then the batches [(3, 10), (3, 5), (2, 1)] and
[(3, 1), (9, 2)], and print the final stock of items 1, 2, 3.

## Output

```text
order 1 x4: ok 100
order 2 x5: out of stock (want 5, have 3)
order 9 x1: no item 9
order 2 x3: too large 300
order 3 x20: ok 140
order 1 x6: ok 150
order 1 x1: out of stock (want 1, have 0)
batch ok 205
batch failed: no item 9
stock: 0 2 14
```
