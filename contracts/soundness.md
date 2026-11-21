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

## Operands: a move is checked where it is evaluated

**Claim.** In an accepted program no operand of an expression reads, borrows, or moves a place
that an earlier by-value operand of the same expression moved.

**Assumptions.** The lowering evaluates the operands of a call (the receiver first), a literal,
an operator, `select`, and `p op= e` left to right, each to a value, and a nested call or borrow
of an operand before the operands after it. A by-value operand that is not Copy ends the bytes
of its place when it is evaluated (M9, M15).

**Argument.** The borrow CFG makes the move of every by-value operand that has operands after
it a statement of its own, at the operand's position, into a temporary the expression consumes
at its end. The statements of the later operands (nested calls, borrows, closure captures,
reads) and the expression's own statement come after it in the CFG, so each is checked against
a state in which the place is moved: a use of it or of a part of it is E0502, as for a use in a
later statement. Within one statement the operands are checked in order against the moves of
the ones before them, so two operands of one rvalue never share a moved place either.

**What would break it.** An expression kind whose operands the builder takes without this step;
a lowering that evaluates operands in another order than the CFG; a Copy operand treated as a
move (it reads its bytes where it stands and leaves them).

## Whole moves: no value is left with a moved-out part

**Claim.** In an accepted program a place is either wholly initialized or wholly moved-out:
no drop, destructor, read, or later move sees a value with a moved-out part.

**Assumptions.** Every by-value use of a non-Copy place is built by one CFG builder step
(`move_out`, through `op_of_place` for operands and receivers and `bind_name` for pattern
bindings of `let`, `match`, and `for`). A pattern that binds a non-Copy part by value reads its
parts into the bindings and then moves its source as one value; the lowering drops the parts it
does not bind at that point (contracts/lang-details.md "Whole moves").

**Argument.** Outside a pattern, a by-value use of a place with a field, tuple, or payload
projection is E0504 case `part`; a projection through a reference, an index, a capture, or a
static is E0504 already. Inside a pattern the bindings read parts and the source moves whole,
and the source itself must be a whole value by the same step; a part a pattern binds twice
(`P.{ a: x, a: y }`) is E0502, so each part has one owner. A value whose type has an
`impl Drop` is not taken apart at all (case `drop`): its destructor gets every part. So every
move operand names a whole local or temporary, the checker's move state is one bit per local,
and a write of a part of a moved local is E0502. What is moved-out is always a whole value, and
a whole value is dropped or read only when it is initialized.

**What would break it.** A by-value use built without `move_out`; a pattern whose ignored parts
the lowering keeps in the place; `p.read()` of a part (unsafe: its obligation).

## Places that are never moved from: statics

**Claim.** No accepted program moves the value of an immutable `static` or of a part of it.

**Assumptions.** A static has one value for the whole run; every use reads it in place, and
none drops it. The borrow CFG gives every use of an immutable static path a temporary standing
for its storage; field, tuple, and index steps are places under it.

**Argument.** A by-value use of a non-Copy place rooted at such a temporary is E0504 case
`static`, at the same builder step as every other move, so `let o = O;`, `O.into_inner()`, an
argument, a literal field, a match or `let` pattern binding by value, a `for` over a static
array, and a use inside a closure are all rejected. A Copy read copies bytes and leaves the
static's value whole; a borrow is a loan of the temporary that never conflicts with a move.
`const` items are not places: each use is a fresh value.

**What would break it.** A path to a static that the builder takes as a constant; a `static
mut` (unsafe, std and runtime only: not covered).

## Match guards only read

**Claim.** Running a guard leaves every place as it was, so an arm reached after a guard was
false sees the state from before the match.

**Assumptions.** A guard is evaluated after its arm's pattern is tested; when it is false the
next arm is tested on the same scrutinee and the same locals. The lowering and the borrow CFG
start every arm from the state before the match.

**Argument.** The CFG builder marks the evaluation of a guard expression (and of every
closure body written in it) and rejects there (E0505) each step that can change a place: a
by-value use of a non-Copy value (the only way to move), an assignment, a `&mut` borrow (the
only way a call can write a place: an argument by copy or `&` writes nothing outside
`UnsafeCell`), and `?` (which leaves the fn with the guard's moves undone). A write or `&mut`
of the arm's own bindings is E0503 and of the scrutinee E0501, as before. What is left reads
places and makes temporaries, which are dropped at the guard's end on both edges. So no move
or write made by a guard is undone or repeated by the next arm.

**What would break it.** Interior mutability (`Cell`, atomics, a `Mutex` locked in a guard):
a write through `&` is not a change this rule sees, and it is not unsound (no place is moved
or dropped twice); an unsafe write through a raw pointer (its obligation).

## Capture modes: one answer for every rule

**Claim.** A capture held by `&` is never written, mutably borrowed, or moved out by its
closure, so the thread rule may ask only `T: Sync` for it, and the borrow checker may give it a
shared loan.

**Assumptions.** The capture analysis (stave-sema `closures`) gives each capture one mode, and
the thread rules (E0802) and the borrow checker read that mode; no rule derives its own. A
capture is `&` only when every use of it is a read in one of the listed contexts (LANG.md §1,
contracts/lang-details.md §1), whatever dereferences lie between the local and the use; a
by-value use of a non-Copy place is a move out of the capture, E0504.

**Argument.** A write or `&mut` through a capture reaches the captured place by a chain of
steps rooted at the local; the analysis classifies the context the chain ends in, not the
chain, so a dereference on the way (`*t.0`, `*a[0]`, `w.v.push`) cannot hide it. An unlisted
context is a mutation (fail-closed). The borrow checker's CFG writes, mutably borrows, and
activates places explicitly; where it does one of these to a capture the analysis held by `&`,
or captures a local the analysis did not list, the body fails as not checked instead of being
accepted with two answers.

**What would break it.** A rule that computes a mode of its own (for example the borrow
checker strengthening a mode only for itself), a new expression form that mutates a place
without the CFG showing it, or a context added to the read list that can write.

## `for` over a place: the checker's loop is the lowered loop

**Claim.** What the borrow checker assumes a `for` loop touches is what the generated code
touches.

**Argument.** An array or `[T]` place is never a source (E0219, contracts/lang-details.md §4
`for` sources), so no loop reads a place the body may change without a borrow saying so: the
source is a reference (each element or chunk of `*r` is borrowed per iteration, `&mut` only
through a `&mut` source), an array value (the loop's own temporary, which nothing else names),
a range, or an iterator: a value is moved into a loop temporary at the head, advanced there
and dropped right after the loop on every path out, in both the CFG and the code; a `&mut` to
an iterator is advanced through the reference in both (a `&` to one is E0219). A reference chunk binding is a borrow of the block's chunk temporary, which is what the
code binds, so it cannot outlive the block, and the chunk of the source stays borrowed until the
store back.

**What would break it.** A `for` source form the CFG models as a copy and the code reads in
place (as an array place was before E0219), or a binding the code points into memory the CFG
does not model as borrowed.
