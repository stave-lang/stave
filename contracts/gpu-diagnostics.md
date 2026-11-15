# Contract: GPU diagnostic codes (v0.2)

New range **E11xx** in the shared registry
([diagnostic-codes.md](diagnostic-codes.md)). Codes are stable and
never reused. Existing codes are reused where the rule is the same:
- E0201: mismatched types;
- E0501: borrow conflict, with a launch as the borrow site;
- E0700/E0701: lowering classes;
- E0702: vector too wide;
- E0704: missing feature, including `cuda`.

Every error carries a suggested fix.

| Code | Rule | Suggested fix |
|---|---|---|
| E1101 | A CPU-only feature (heap, I/O, threads, `extern`, `dispatch`, f-strings) is reachable from a kernel. The call chain is listed. The listed chain is the shortest call chain from a kernel; ties are broken by the smallest caller decl id at each step | Move the work to the host, or pass the data in as a buffer |
| E1102 | Recursion or a call through a fn value in device code | Rewrite as a loop, or call the function directly |
| E1103 | Write to `gpu.Out`, `gpu.AtomicOut`, or `gpu.Shared` other than through `at(t)` or an atomic operation | Use `out.at(t)`, an atomic operation, or `unsafe("…")` with the obligation |
| E1104 | A `gpu.Launch` goes out of scope without `wait()` | Call `l.wait()?` before the end of the scope |
| E1105 | A collective in non-uniform control flow (the non-uniform source is shown) | Hoist the collective out of the branch; make the condition depend only on uniform values |
| E1106 | Memory-space mismatch (host vs device, `In` vs `Out`, global vs shared) | Shows the expected type, for example `&gpu.Buf<f32>` instead of `&[f32]`; use `dev.alloc_copy` |
| E1107 | Thread witness copied, stored, returned, or captured | Pass `&t` to device functions |
| E1108 | Invalid kernel signature (first parameter not a witness, return type, `needs`, generics, or a parameter type that is not a device-parameter type) | Shows the corrected signature |
| E1109 | A constant launch configuration exceeds the target's limits (threads per block, shared memory) | Shows the limit and a valid configuration |
| E1110 | A type that is not `gpu.Plain` used as a buffer element or kernel parameter | Use plain data: numbers, vectors, arrays, structs of those |
| E1111 | A `gpu.Tile` used after the next collective (including another `tile_load` or a call to a device fn that runs one). The collective that ended the tile's borrow is shown as a related span | Use the tile before that collective, or copy the needed elements into locals first; call `gpu.tile_load` again after the collective if the data is still needed |
| E1112 | Warning. A branch on a thread-varying value inside a loop of a kernel or device fn (threads of a warp diverge each iteration); the non-uniform source is a related span. Uniform branches and branches outside loops are not warned (all are listed by `stave branch-report --target gpu-sm86`). Planned | Use `c.select(a, b)` (lowers to `selp`, no divergence), or hoist the condition out of the loop |

Runtime errors are not diagnostics. They are `gpu.Error` values: probe errors, driver errors (with
the CUDA error name), out of memory, launch-precondition errors (grid shape, block size, limits),
and `KernelError { kernel, span, kind: Bounds | Overflow | DivZero | Shift, block, thread }`.
Variants and `Display` texts: [gpu-ops.md](gpu-ops.md) "Errors".
