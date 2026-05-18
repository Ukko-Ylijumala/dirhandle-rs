# DirFd — atomic fd wrapper

`DirFd` is a thin wrapper around a `RawFd` backed by `AtomicI32`. It exists for two reasons: cheap thread-safe sharing of a file-descriptor identity, and a sign-encoded state machine that distinguishes "open," "uninitialized," and "was open, now closed."

## State machine

The inner `AtomicI32` carries both the fd value and its lifecycle state:

| Inner value           | Meaning |
| --------------------- | ------- |
| `>= 0`                | Open file descriptor. **fd 0 is a valid open fd** — a process that closed stdin can legitimately receive it back from `open()`. |
| `i32::MIN`            | Uninitialized — the `Default` state. |
| `< 0` (not `i32::MIN`)| Stale: the fd was previously open and has been cleared. The original fd is recoverable as `!stored` (bitwise NOT). |

The stale encoding uses `!fd` rather than `-fd` so that the unique sentinel `0` and the unique sentinel for uninit don't collide. With `!fd`:

- open fd 0 → stale `-1`
- open fd 1 → stale `-2`
- open fd N → stale `-(N+1)`

`clear()` encodes the current open fd as `!current` if open, otherwise overwrites with `UNINIT_FD` (`i32::MIN`). A second `clear()` on an already-stale `DirFd` therefore "buries" it to uninitialized, losing the historical fd number — by design, since a thread holding a stale handle shouldn't pretend it still knows what was there.

`set()` is fail-safe: it refuses to overwrite a currently-open fd, returning `Err(existing)`. Callers must `clear()` first if they really mean to replace the fd.

Both `set()` and `clear()` use `AtomicI32::fetch_update`, so the check-and-store is a single CAS loop. Two concurrent `set()` calls cannot both succeed, and a `clear()` interleaved with a `set()` either fully precedes or fully follows it — never lands between the check and the store.

## Path resolution

`path()` resolves the fd back to a filesystem path by reading `/proc/self/fd/<fd>`. This is a hard dependency on Linux procfs being mounted; on systems where it isn't, `path()` returns `io::ErrorKind::Unsupported`. Other failure modes (`NotFound` for `0` or negative fds) are returned without touching procfs.

## Thread-safety and `AsFd`

Because the inner storage is atomic, `DirFd` is `Send + Sync` and can be cloned cheaply across threads — each `clone()` snapshots the current value into a fresh atomic.

The `AsFd` implementation uses `unsafe { BorrowedFd::borrow_raw(...) }`. The safety contract `BorrowedFd::borrow_raw` requires — that the fd is open and remains so for the borrow's lifetime — is **not** enforced inside `DirFd` itself. Note that `EntryExt` does **not** use `DirFd::as_fd`: it holds its own `BorrowedFd<'h>` borrowed directly from the parent `DirHandle`'s `nix::Dir` (see [entry-ext.md](entry-ext.md)). The remaining users of `DirFd::as_fd` must ensure `is_open()` is true before invoking it, or they violate the `BorrowedFd::borrow_raw` contract.
