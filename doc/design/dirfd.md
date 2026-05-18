# DirFd — atomic fd wrapper

`DirFd` is a thin wrapper around a `RawFd` backed by `AtomicI32`. It exists for two reasons: cheap thread-safe sharing of a file-descriptor identity, and a sign-encoded state machine that distinguishes "open," "uninitialized," and "was open, now closed."

## State machine

The inner `AtomicI32` carries both the fd value and its lifecycle state:

| Inner value | Meaning |
| ----------- | ------- |
| `> 0`       | Open file descriptor. |
| `== 0`      | Uninitialized — no fd has ever been stored. |
| `< 0`       | The fd was previously open as `abs(value)`, then explicitly cleared. |

`clear()` deliberately flips a positive fd to its negative twin, preserving the historical fd number so downstream debugging can still ask "what fd did this used to be?" Clearing an already-cleared `DirFd` resets it to `0`.

`set()` is fail-safe: it refuses to overwrite a currently-open fd, returning `Err(existing)`. Callers must `clear()` first if they really mean to replace the fd.

## Path resolution

`path()` resolves the fd back to a filesystem path by reading `/proc/self/fd/<fd>`. This is a hard dependency on Linux procfs being mounted; on systems where it isn't, `path()` returns `io::ErrorKind::Unsupported`. Other failure modes (`NotFound` for `0` or negative fds) are returned without touching procfs.

## Thread-safety and `AsFd`

Because the inner storage is atomic, `DirFd` is `Send + Sync` and can be cloned cheaply across threads — each `clone()` snapshots the current value into a fresh atomic.

The `AsFd` implementation uses `unsafe { BorrowedFd::borrow_raw(...) }`. The safety contract `BorrowedFd::borrow_raw` requires — that the fd is open and remains so for the borrow's lifetime — is **not** enforced inside `DirFd` itself. Every consumer of `DirFd::as_fd()` (today: `EntryExt::stat` and `EntryExt::open` — see [entry-ext.md](entry-ext.md)) must check `dirfd.is_open()` first and bail out otherwise. Preserve these guards when refactoring; removing them turns a logic error into UB.
