# Security policy

Stave is early, pre-release software. There are no supported releases yet; fixes go to `main`.

## Reporting a vulnerability

Report privately through the repository host's private vulnerability reporting: open the
repository's Security tab and choose "Report a vulnerability". Do not open a public issue or pull
request for a vulnerability.

## Scope

In scope:

- The compiler reading or writing files, or executing programs, outside what the command asked for.
- Memory-safety bugs in `runtime/` and `std/` that safe Stave code can trigger.
- A crafted input that makes the toolchain (compiler, `stave run`, `stave test`) execute code or
  write files the user did not ask for.

Not in scope: soundness holes in the type or borrow checker, and miscompilations, are ordinary
bugs; report them as public issues. Report one privately only if it can be exploited through the
toolchain itself.
