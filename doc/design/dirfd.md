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

Construction (`new(&fd)`, `From<RawFd>`, `DirHandle::fd()`) only snapshots the number; a `DirFd` never owns or closes an fd. `new()` takes a reference since 0.6.0 — by value it dropped, and thereby closed, an owned `File`/`OwnedFd` on the spot, leaving a `DirFd` that reported an open fd.

`set()` is fail-safe: it refuses to overwrite a currently-open fd, returning `Err(existing)`. Callers must `clear()` first if they really mean to replace the fd.

Both `set()` and `clear()` use `AtomicI32::fetch_update`, so the check-and-store is a single CAS loop. Two concurrent `set()` calls cannot both succeed, and a `clear()` interleaved with a `set()` either fully precedes or fully follows it — never lands between the check and the store.

## Path resolution

`path()` resolves the fd back to a filesystem path by reading `/proc/self/fd/<fd>`. This is a hard dependency on Linux procfs being mounted; on systems where it isn't, `path()` returns `io::ErrorKind::Unsupported`. Stale/uninitialized fds return `NotFound` without touching procfs.

A directory deleted while the fd is open still resolves in procfs, to `"<old path> (deleted)"`. `proc_fd_path()` refuses to hand that out: after a successful `readlink` it `statx`es the fd (`STATX_NLINK` only) and fails with `NotFound` when `stx_nlink == 0`, which is what `rmdir` leaves behind on the inode. The link count is used rather than the string suffix so that a directory merely *named* `"something (deleted)"` still resolves. `DirHandle::path()` and `EntryExt::path()` share the same helper, so all three entry points agree. The helper takes the bare `RawFd`: `DirFd::path()` has no `BorrowedFd` it could vouch for (see below), and a stale number either fails at the `readlink` or describes whatever reused the fd, which is a wrong answer but never undefined behaviour.

Availability is diagnosed *after* the fact, not probed up front: `/proc/self/fd` is itself a directory (its entries are the symlinks), so `readlink()` on it fails with `EINVAL` even when procfs is mounted — a pre-check built on that call would report procfs as missing unconditionally. Instead, `proc_fd_path()` attempts the per-fd `readlink` directly and only maps the error to `Unsupported` when the failure is `NotFound` *and* `/proc/self/fd` does not exist as a directory.

## Thread-safety and why `DirFd` is not `AsFd`

Because the inner storage is atomic, `DirFd` is `Send + Sync` and can be cloned cheaply across threads — each `clone()` snapshots the current value into a fresh atomic.

`DirFd` implemented `AsFd` from 0.3.8 to 0.7.0 and does not any more (0.7.1). A safe `as_fd()` carries Rust's I/O-safety contract: the returned `BorrowedFd` refers to an fd that stays open for the whole borrow. A `DirFd` cannot keep that promise — it is a snapshot of a number it does not own, the owner may close the fd at any time, and the kernel then hands the same number to the next `open()` anywhere in the process. Safe code holding the stale borrow would `fstat`, `openat` or `renameat` against a stranger's file, and generic code that trusts `AsFd` has no way to tell. The old impl only asserted `fd >= 0` at the instant of the call and left the rest to a doc comment, which is an unsafe contract attached to a safe trait method.

The replacement is `unsafe fn borrow_fd(&self) -> BorrowedFd<'_>`, with the real contract in its `# Safety` section: the fd is open now and stays open for the borrow, which in practice means holding the owner (the `DirHandle`, or the `OpenHandles` checkout) for that long. It keeps the `assert!(fd >= 0)`: `BorrowedFd::borrow_raw` forbids `-1`, and the uninit sentinel is no fd at all, so a stale or uninitialized `DirFd` panics deterministically rather than producing UB. The internal callers never needed the impl: they hold their own `BorrowedFd<'h>` (see [entry-ext.md](entry-ext.md)) or go through the `DirHandle`, which owns the `nix::Dir`; `DirFd::path()` passes the raw number to the procfs helper. `AsRawFd` stays, since a bare number makes no promise.
