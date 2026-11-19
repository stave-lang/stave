# Contract: soundness arguments of the static rules

One short argument per rule whose failure would let an accepted program misbehave: what the rule
assumes, what it guarantees, and what would break it. These are arguments, not machine-checked
proofs. A change to a rule changes its argument here in the same commit; a rule with no argument
that still holds is not ready.

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
