## Task

Define a struct `Step { id: u32 }` whose destructor prints `undo <id>`, and
`stage(id: u32, fail: bool) -> Result<Step, u32>` that prints `do <id>` and then fails with the id
or returns a new Step. Write `run(fail_at: u32) -> Result<u32, u32>` that runs stages 1, 2, 3 in
order with `?` (stage k fails when k = fail_at), prints `all done` and returns the sum of the three
ids. Call run with 2, 0, 3 and print `ok <sum>` or `failed at <id>` after each.

## Output

```text
do 1
do 2
undo 1
failed at 2
do 1
do 2
do 3
all done
undo 3
undo 2
undo 1
ok 6
do 1
do 2
do 3
undo 2
undo 1
failed at 3
```
