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

## Stores of a call: the result does not depend on order

**Claim.** Applying the stores of one call until nothing changes ends, and the sets it ends at
do not depend on the order of the params, of the entries of a clause, or of the steps.

**Assumptions.** A call has one store per target param with a clause. With `P` the set of the
target argument and `S` the union over the clause's entries of the set of the source argument
(`from q`) or of what the views it borrows hold (`from copy q`), the store is the rule: every
local view in `reach(P) ∪ reach(S)` holds `S`. One view may leave out a loan of a place of its
own local that overlaps the place it is reached by, unless `S` holds a loan made through a
pointer with that loan. Every read is monotone in the state: larger sets give a larger `P`, `S`
and `reach`, and leave out fewer loans (which loans may be left out is fixed by their places). A
step only adds atoms. A borrowed local is read by its set at the call once a store has grown a
local in the body, and by what it held when it was borrowed before that; a loan in `P` or `S` is
live at the call, so its place has only gained atoms since, and the change of reading adds atoms
like a step.

**Argument.** The state is a finite family of sets over a finite set of origin atoms. A step
replaces the state `x` by `x ∪ f(x)` for one rule `f`. Each step that changes the state adds an
atom, so there are at most (sets × atoms) of them: the loop ends, in a state closed under every
rule. That state is the least closed state above the start. Let `y` be closed and above the
start; if `x` is below `y` then `x ∪ f(x)` is below `y ∪ f(y) = y`, so every state the loop
passes through is below `y`. Two orders end at closed states above the same start, each below
the other: the same state.

**Use.** A verdict does not change when a signature lists its params, or a clause its sources, in
another order. A `from copy` source and a `from copy` result are read from the end state, and
each store is checked against it: they see the stores of the same call, whichever param they
belong to.

**What would break it.** A rule that removes an atom. A read that gets smaller when a set grows.
A `from copy` source or a result read before the loop ends. A bound that stops the loop early.

## What a caller may assume about a storing call comes from the signature only

**Claim.** A local view that a store of a call reaches keeps the loan it is reached by. After
`set_link(&mut l, &mut b)` with `fn set_link(l: &mut Link from q, q: &mut Pick)`, `b` holds the
loan of `&mut b` and stays borrowed for as long as it is used. No rule that looks at the form of
the argument (a borrow expression, a temporary, a named pointer) may leave that loan out.

**Assumptions.** A call is checked against the callee's signature, never its body. A callee may
store through a param that the clause of a `&mut` param names without `copy`, when that clause
covers the value; the value may hold the origin of that same param.

**Argument.** Take two bodies for the signature above. The first is `l.to = q;`. The second
moves `q` into a by-value argument of a helper that returns two pointers into `*q`, and stores
one through the other: `*q` then holds a pointer into itself. Both fit the callee rules (the
clause of `l` names `q` and covers its origin). A rule that reads the signature and the form of
the arguments gives one verdict for both. If that verdict leaves the loan of `&mut b` out of
`b`, then after the second body no local holds the loan once `l` is dead, and a write of the
part of `b` that `b` points to is accepted. So `b` holds the loan. A local that holds a `&mut`
loan of itself conflicts with every later use of itself, which is the cost: the first body
would have allowed those uses.

**What would break it.** A caller rule that reads a callee's body, or leaves a loan out because
of how the argument is written. Accepting a use of `b` after the last use of `l` needs a promise
in the signature that the callee keeps the source pointer whole and stores nothing made from it.
`from copy q` is not that promise: it names what `*q` holds, so a callee under it may not keep
`q` at all. A promise about values that hold the origin of `q` is not enough either: once the
pointer is in a view, an exchange returns it by value with the origin of that view's content,
the one `from copy` names, and a rule that follows the origin of `q` no longer sees it.
