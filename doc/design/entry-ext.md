# EntryExt — enriched directory entry

`EntryExt<'h>` is built from a `nix::dir::Entry` and aims to feel like `std::fs::DirEntry` with a few additions. It is the unit yielded by every public iterator on `DirHandle`.

## Compact layout (v0.5.0)

The `Entry` is **not** retained. A `dirent` is 280 bytes, 256 of them the `d_name` array, and the old `EntryExt` (dirent + `BorrowedFd` + inline `OnceLock<Option<libc::stat>>`) weighed about 440 bytes — every lookahead push/pop, `Peekable` slot and sort comparison moved that much. `EntryExt::new` now copies out just what is needed:

| Field    | Type                                  | Notes |
| -------- | ------------------------------------- | ----- |
| `name`   | `EntryName`                           | Owned, NUL-terminated. Inline up to 38 bytes (`NAME_INLINE_CAP - 1`, so UUID-length names stay inline), heap `Box<[u8]>` beyond. 48 bytes. |
| `ino`    | `u64`                                 | |
| `d_type` | `Option<Type>`                        | As reported by `readdir`; `None` = `DT_UNKNOWN`. |
| `dirfd`  | `BorrowedFd<'h>`                      | |
| `stat`   | `OnceLock<Option<Box<libc::stat>>>`   | Boxed so the 144-byte stat only costs when actually taken. |

Total ≤ 80 bytes (unit-tested). Both `file_name()` (`&CStr`) and `name_as_bytes()` are free views of the stored bytes — no `strlen` per call, which the old `dirent`-backed accessors paid on every `Ord`/`Eq`/`Hash` use.

`Deref<Target = Entry>` is gone with it. The three accessors it used to expose are inherent now: `file_name()`, `ino()` and `d_type()` (the raw `readdir` type, as opposed to `file_type()` which adds the `fstatat` fallback). `EntryExt::new` / `new_statted` take `&Entry`.

## Lifetime binding

The `'h` parameter is the borrow on the parent `DirHandle`. Internally, `EntryExt` stores a `BorrowedFd<'h>` of the directory's file descriptor, not a snapshot. The borrow checker rejects code that holds an `EntryExt<'h>` past the lifetime of the underlying `DirHandle`:

```rust
let entries: Vec<EntryExt> = {
    let mut h = DirHandle::new(path)?;
    h.iter().collect()
};  // ❌ borrow of `h` is still held by `entries` — won't compile
```

Collecting into a `Vec` keeps the mutable borrow on the `DirHandle` alive for the lifetime of the vec. To outlive the parent handle, an entry would have to be reconstructed from owned data (`Entry`, parent path), not carried as-is.

This replaces an earlier design in which `EntryExt` held a cloned `DirFd` (independent `AtomicI32` snapshot), allowing zombie entries to survive past `DirHandle::drop` and silently issue `fstatat`/`openat2` against closed or reused fds.

## Cached stat

The entry holds an `OnceLock<Option<Box<libc::stat>>>`. `stat()` initialises it lazily (one heap allocation on first use); subsequent calls return the cached value. `stat_refresh()` clears the lock and re-stats. The `Option` inside the lock means "we already tried and failed" is cached too — repeated stat failures do not retrigger the syscall.

`stat()` is the only path that can promote an entry's known type from `None` to something useful: when `dirent.d_type` is `DT_UNKNOWN` (some filesystems never populate it), `file_type()` falls back to `EntryType(self.mode()).entry_t()`. The local `EntryType` struct exists because `std::sys::pal::unix::fs::FileType`, which performs the same `S_IFMT` masking, is private and cannot be imported.

Because the `BorrowedFd<'h>` guarantees the parent fd is alive for the entry's lifetime, `stat()` and `open()` do **not** need explicit liveness guards — the type system has already ruled out the closed-fd case for safe code paths.

## Safe `openat2`

`read()` and `write()` open the entry via `nix::fcntl::openat2` with `ResolveFlag::RESOLVE_BENEATH`. The kernel rejects any path that would resolve outside the parent dirfd — symlink loops, `..` traversals, or absolute paths. Do not "simplify" this to plain `openat` or `open` without a deliberate reason; it silently broadens the trust boundary. `O_CLOEXEC` is unconditionally OR'd into the flags so the opened file descriptors do not leak into exec'd children (directory fds opened by `get_dir_handle` get the same treatment, plus `O_DIRECTORY | O_NONBLOCK`).

`open_dir()` is the directory counterpart: it opens the entry as a new `DirHandle` through the same `openat2 + RESOLVE_BENEATH` path (with `O_DIRECTORY`), so recursive tree descent needs neither procfs nor path re-resolution and is immune to rename races by construction. Prefer it over `path()` + `DirHandle::new()` when walking trees.

Beyond `len()`/`mode()`, the cached stat also feeds `is_empty()`, `uid()`, `gid()`, `nlink()`, and `mtime()`/`atime()`/`ctime()`. The timestamps come back as `TimeSinceEpoch` (`f64` seconds — about microsecond precision); callers needing exact nanosecond timespecs should read `stat()` directly.

## Equality, ordering, hashing

`Eq`, `Ord` and `Hash` are all defined over explicit field tuples. Historical note, in case anyone is tempted to store the `Entry` again: nix fills the dirent from `readdir_r` into a `MaybeUninit` buffer and only `d_reclen` bytes are copied, while the libc derives compare/hash the entire struct including `d_off`, `d_reclen` and the uninitialized tail of the 256-byte `d_name` array. Delegating to those would make the same logical entry compare unequal (and hash differently) between two reads within one process. Copying the fields out in `new()` is what makes the current design immune.

- `PartialEq`/`Eq` — `(name_bytes, ino, parent dirfd)`.
- `Ord` — name first (unique within a directory, so sorting behaviour is name-order), with ino and dirfd as tie-breakers so that `cmp() == Equal ⇔ eq()`. Keep these two consistent; sorted-collection invariants depend on it.
- `std::hash::Hash` — `(name_bytes, ino)`, a subset of the `Eq` fields (equal entries hash equal). SipHash via the default hasher: not stable across processes; not safe for persistence. The `dirfd` is **deliberately omitted** — including it would invalidate hashes whenever the directory is reopened with a different fd.
- `custom_xxh3::Xxh3Hashable` — stable, hashes `(name_bytes, inode, typenum)`, where `typenum` is the kernel `DT_*` value (pinned `TYPENUM_*` constants, never nix's enum discriminant). This is the hash used for `DirectoryState` change detection, where stability across reopens is the whole point. See [state-tracking.md](state-tracking.md).
