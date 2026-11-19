# Contract: soundness arguments of the static rules

One short argument per rule whose failure would let an accepted program misbehave: what the rule
assumes, what it guarantees, and what would break it. These are arguments, not machine-checked
proofs. A change to a rule changes its argument here in the same commit; a rule with no argument
that still holds is not ready.

## Integer arithmetic: a result is exact or the program traps

**Claim.** Let `e` be an expression built from integer values with the trapping operators
`+ - * / %`, unary `-`, lossless `as`, and comparisons. Evaluating `e` either traps or gives the
value of the same expression over the unbounded integers.

**Argument.** By induction on `e`. A leaf is its own value. For an operator, the operands are
exact by the induction hypothesis; the operator computes the exact result `r`; it returns `r` when
`r` is a value of the result type and traps otherwise (`/` and `%` also trap on a zero divisor). A
lossless `as` changes the type, never the value. A comparison of exact values is exact.

**Use.** On a path with no trap, a contract clause over such expressions means what it says in
ordinary arithmetic. The prover may treat an integer node as the mathematical value under its
type, and a reader of `requires` / `ensures` needs no knowledge of machine widths beyond the type
ranges.

**Not covered.** `+% -% *% <<%` (wrap), `+| -| *|` (saturate), `as% as| as? as~`, floats, and
bit operations: each is a function of the operand bits, exact in its own sense, but not the
unbounded-integer operation of the same name. A rule that reads one of them as arithmetic needs
its own argument.

## Removed checks: the first failure

**Claim.** Removing every check that has an accepted witness does not change the behaviour of any
execution.

**Assumptions.** A witness is accepted only by the witness checker (`checked-ir.md`), which
re-derives each step from the function's IR and from certified summaries of its callees. Every
fact a witness cites is established earlier in the same execution: a branch taken, a check that
stayed, a check that was removed with its own witness, or a call that returned (its callee's
`ensures` rows, and its `requires` rows, which held at the call or the callee would have
trapped).

**Argument.** Suppose some execution of the program with checks removed differs from the
execution with all checks kept. Take the first removed check, in execution order, that would have
failed. Every fact its witness cites comes from an earlier point; all earlier removed checks
would have passed (this one is the first to fail), and kept checks and returned calls behave as
in the checked program. So every cited fact holds, the chain holds, and the check passes: a
contradiction.

**Recursion.** The argument orders checks by time, not by the call graph, so a function may use
its own `ensures` after a recursive call returns: that call's checks ran earlier. Nothing is
claimed about a call that does not return (partial correctness).

**What would break it.** A witness accepted by code other than the checker; a summary row not
certified against the callee's own IR; a fact taken from a point that does not dominate the
check; an unchecked entry reached by a call whose proof covered only some conjuncts of the
callee's `requires` (acceptance is all or nothing for that reason).

## Overwritten values: loans are read after the statement

**Claim.** When a statement ends the value of a place and reports no live loan of it, no
reference into the ended value is used after the statement. The statements are an assignment to
a place that holds a value (`p = e`, `p.f = e`, `*q = e`) and the initialization of a temporary
by another evaluation of its expression (a temporary has one slot).

**Assumptions.** The origin set of a local at a point holds the loan of every borrow its value
may hold there. A loan is live after a statement when a local that is used later holds it in the
state after the statement. A statement writes its destination last, after its operands are read
(the result of a call is written by a statement of its own).

**Argument.** The check must read the state after the statement. In
`x = H.{ s: x.name.as_str() }` nothing that is used later holds the borrow of `x.name` before
the statement: only the operand holds it, and the statement consumes the operand. After the
statement the new `x` holds it, and the assignment has dropped the old `x.name`. The state
before the statement cannot show that holder.

Reading the state after it is enough. The old value ends at the statement. A reference into it
does harm only when it is used later, and a reference used later is held by a local that is live
after the statement. The set of that local in the state after the statement holds the
reference's loan: the statement leaves the set of every other local as it is or adds to it, and
the set of the destination is that of the new value. So the loan is among the loans the write is
checked against, and the write conflicts with it.

The other statements that end a value give no local a new value that the state before them does
not show. A drop and the end of a local's storage give no value at all. A move out of a place is
a read of an operand: a loan in the value it makes came through an operand of the same
statement, and an operand's local is live at the statement. These are checked against the loans
live at the statement.

**Not claimed.** A destination that is never read again may take a borrow of its own old value:
no reference is used. A destination with drop glue is read by its drop.

**What would break it.** A statement that ends a value and gives a local a new one without this
check. A transfer function that leaves out of the destination's set a loan the new value holds.
Reading the loans live after a statement from the state before it.
