Write a complete Stave program in one source file. It reads no input and exits with status 0.
Its standard output must be exactly the text under "Output" (every line ends with a newline).
Compute the values in the program; do not print them as literal text.

## Task

Make two local `std.sync` mutexes: `balance` (a `u64`, starting at 100) and `log` (a `u64` count,
starting at 0), with the lock ranks the language requires for taking `log` while holding
`balance`. In a structured thread scope spawn two threads; each performs 5 deposits of 10, where
a deposit adds to `balance` and, while still holding `balance`, increments `log`. Print both
final values.

## Output

```text
balance 200
log entries 10
```
