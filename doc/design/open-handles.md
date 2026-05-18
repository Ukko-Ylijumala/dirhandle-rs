# OpenHandles — thread-safe handle pool

`OpenHandles` is a `DashMap<RawFd, DirHandle>` wrapper that lets multiple threads share a pool of open directory handles. Handles are checked out exclusively via `CheckedOutHandle<'a>`, which derefs to `DirHandle`.

## Why DashMap

DashMap provides per-shard `parking_lot::RwLock` concurrency. `OpenHandles` exposes:

- `open(path)` — opens a new `DirHandle`, inserts it, returns a checked-out reference.
- `get(fd)` — returns a checked-out reference if the fd is in the pool.
- `insert(handle)` — stores a handle; replaces any existing entry under the same fd.
- `close(fd)` / `close_all()` — drops the handle(s), closing the underlying file descriptor(s).
- `for_each` / `for_each_mut` — closure-based iteration in random order.
- `contains(fd)` / `contains_handle(&h)` — membership tests; the latter is `O(n)`.

All of these may take internal locks. Read the deadlock note below before composing them.

## Deadlock caveat

**Holding more than one `CheckedOutHandle` from the same thread can deadlock.** `CheckedOutHandle` wraps `RefMut<'_, RawFd, DirHandle>`, which is a write lock on a DashMap shard. If a second `get()` / `open()` / `for_each_mut()` call lands on the same shard while the first lock is still held, the thread blocks against itself.

The simplest discipline: drop or `close()` the current checkout before requesting another. If you find yourself wanting two handles at once, restructure to use scoped blocks or clone the data you need out first.

## Rc, not Arc

`CheckedOutHandle::close_callback` is `Rc<dyn Fn(RawFd)>`. `Rc` is deliberate — it prevents the type from being `Send`, so a checked-out handle cannot be moved across threads. Even though the underlying `DirHandle` is `Send`, moving an active lock guard across threads would violate DashMap's locking model. Do not "fix" this to `Arc`.

## Lifecycle

`CheckedOutHandle::close(self)` explicitly drops the `RefMut` before invoking the close callback. This ordering matters: the callback calls `OpenHandles::close(fd)`, which itself acquires a write lock, so the existing `RefMut` must be released first or you hit the same deadlock pattern described above.

Dropping a `CheckedOutHandle` without calling `close()` simply releases the lock — the handle stays in the pool, and the directory remains open.

## SizeOf accounting

Under the `size_of` feature, `OpenHandles` reports memory usage including unused DashMap capacity (`add_excess(...)`) and recurses into each entry. The per-handle cost uses `DHSIZE = 296`, hard-coded from the layout of `nix::dir::Dir` + `libc::DIR` + `libc::dirent`. Re-verify if either dependency changes its representation.
