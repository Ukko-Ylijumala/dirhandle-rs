# OpenHandles — thread-safe handle pool

`OpenHandles` is a `DashMap<RawFd, DirHandle>` wrapper that lets multiple threads share a pool of open directory handles. Handles are checked out exclusively via `CheckedOutHandle<'a>`, which derefs to `DirHandle`.

## Why DashMap

DashMap provides per-shard `parking_lot::RwLock` concurrency. `OpenHandles` exposes:

- `open(path)` — opens a new `DirHandle`, inserts it and returns a checked-out reference in one step (`DashMap::entry().insert()` hands back the write-locked `RefMut` directly, so a concurrent `close(fd)` cannot slip in between insert and checkout).
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

`CheckedOutHandle` carries a `PhantomData<Rc<()>>` marker. `Rc` is deliberate — it makes the type `!Send`, so a checked-out handle cannot be moved across threads. Even though the underlying `DirHandle` is `Send`, moving an active lock guard across threads would violate DashMap's locking model. Do not "fix" this to `Arc` or drop the marker; a `compile_fail` doc-test on the struct guards the property.

Earlier versions got the same `!Send` effect from an `Rc<dyn Fn(RawFd)>` close callback, which cost a heap allocation and a vtable call per checkout. The handle now holds a plain `&OpenHandles` and calls `close(fd)` on it directly.

## Lifecycle

`CheckedOutHandle::close(self)` explicitly drops the `RefMut` before calling `OpenHandles::close(fd)` on the pool. This ordering matters: `close(fd)` acquires a write lock on the shard, so the existing `RefMut` must be released first or you hit the same deadlock pattern described above. The unlock-then-remove sequence opens a tiny fd-reuse window (another thread closes the fd, the kernel recycles the number for a fresh handle, and the remove evicts the newcomer) — inherent to keying the pool by `RawFd`.

The map being keyed by `RawFd` also means the pool is sound only while each entry's fd number is owned by its `DirHandle`'s inner `Dir` — which `Dir` guarantees (it closes the fd only on drop, i.e. on removal from the map).

Dropping a `CheckedOutHandle` without calling `close()` simply releases the lock — the handle stays in the pool, and the directory remains open.

## SizeOf accounting

Under the `size_of` feature, `OpenHandles` reports the inline `(RawFd, DirHandle)` slot cost for used and unused DashMap capacity (`add_excess(...)`) and recurses into each entry for its heap children. A handle's heap child is glibc's directory stream: a ~48-byte header plus the `getdents` buffer, which `opendir`/`fdopendir` size as the directory's `st_blksize` clamped to 32 KiB .. 1 MiB. That varies well beyond the minimum — ZFS reports up to its 128 KiB recordsize (and different values for different directories), NFS commonly 1 MiB — so each `DirHandle` records its own stream size with one `fstat` right after the open (only under the feature; accounting itself then needs no syscalls). This dominates the footprint: a pool of 10k handles is on the order of 320 MiB at the 32 KiB minimum, ~1.3 GiB on ZFS and up to ~10 GiB on NFS, not the ~3 MB the old `DHSIZE = 296` estimate implied. Re-verify if glibc's `sysdeps/posix/opendir.c` changes.
