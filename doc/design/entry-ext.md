# EntryExt — enriched directory entry

`EntryExt<'h>` is built from a `nix::dir::Entry` and aims to feel like `std::fs::DirEntry` with a few additions. It is the unit yielded by every public iterator on `DirHandle`.

## Compact layout (v0.5.0)

The `Entry` is **not** retained. Up to nix 0.30 an `Entry` was the raw 280-byte `dirent`, 256 bytes of it the `d_name` array, and the old `EntryExt` (dirent + `BorrowedFd` + inline `OnceLock<Option<libc::stat>>`) weighed about 440 bytes — every lookahead push/pop, `Peekable` slot and sort comparison moved that much. (nix 0.31's `Entry` is small but heap-allocates its name as a `CString`.) `EntryExt::new` copies out just what is needed:

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

`open_dir()` is the directory counterpart: it opens the entry as a new `DirHandle` through the same `openat2 + RESOLVE_BENEATH` path (with `O_DIRECTORY`), so recursive tree descent needs neither procfs nor path re-resolution. Prefer it over `path()` + `DirHandle::new()` when walking trees.

Unlike `read()`/`write()`, `open_dir()` also passes `O_NOFOLLOW`, so it never follows a symlink — not even one that stays beneath the parent (since 0.6.0; before, in-tree symlinks were followed). `RESOLVE_BENEATH` alone only stops escapes: `loop -> .` reopened the parent itself, an endless recursion for a walker, and a directory swapped for a symlink between `readdir` and `open_dir()` steered the walker into a sibling subtree (the race class of std's CVE-2022-21658). With `O_NOFOLLOW | O_DIRECTORY` a symlink fails with `ENOTDIR`, and whatever `open_dir()` opens is the directory entry itself. A caller that wants to follow in-tree symlinks can resolve them deliberately (`read_link` + its own cycle detection).

`read_nofollow()` is the file counterpart (since 0.6.3): `O_RDONLY | O_NOFOLLOW | O_NONBLOCK` with `RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS`. `read()` keeps following in-tree symlinks for compatibility; a consumer reading files that may be hostile (a malware scanner) must not let a planted `x -> ../other/file` redirect the read. `O_NONBLOCK` keeps a FIFO from blocking the open. `open_regular()` adds one `fstat` of the opened fd and refuses anything but a regular file (`InvalidInput`); that stat describes what was actually opened, so it also fills the cached stat when none was taken yet. It does not compare `st_ino` with the listing's `d_ino`: the two differ legitimately for a bind-mounted file and on some overlay filesystems.

All of these are thin wrappers around fd-and-name functions, for callers that kept a directory fd but not the entry: `DirHandle::open_at()` (which `open_dir()` calls), `read_nofollow_at()` and `open_regular_at()`. `DirHandle::path_fd()` (or `path_fd_at()`, of an entry of any directory fd, e.g. one borrowed while the handle's entries still borrow the handle) gives such a caller an `O_PATH` fd that outlives the handle, and `DirHandle::open_beneath()` opens a relative path with no symlink in any component, in chunks of at most `PATH_MAX - 1` bytes (each intermediate chunk opened `O_PATH`), so a directory deeper than `PATH_MAX` stays reachable. It refuses `..`: every chunk is resolved beneath its own fd, so a `..` at a chunk boundary could climb above the starting fd.

Beyond `len()`/`mode()`, the cached stat also feeds `is_empty()`, `uid()`, `gid()`, `nlink()`, and `mtime()`/`atime()`/`ctime()`. The timestamps come back as `TimeSinceEpoch` (`f64` seconds — about microsecond precision); callers needing exact nanosecond timespecs should read `stat()` directly.

## Equality, ordering, hashing

`Eq`, `Ord` and `Hash` are all defined over explicit field tuples. Historical note, in case anyone is tempted to store the `Entry` again: up to nix 0.30, nix filled the dirent from `readdir_r` into a `MaybeUninit` buffer and only `d_reclen` bytes were copied, while the libc derives compare/hash the entire struct including `d_off`, `d_reclen` and the uninitialized tail of the 256-byte `d_name` array. Delegating to those made the same logical entry compare unequal (and hash differently) between two reads within one process. nix 0.31 replaced that representation, but copying the fields out in `new()` keeps the design independent of nix's.

- `PartialEq`/`Eq` — `(name_bytes, ino, parent dirfd)`.
- `Ord` — name first (unique within a directory, so sorting behaviour is name-order), with ino and dirfd as tie-breakers so that `cmp() == Equal ⇔ eq()`. Keep these two consistent; sorted-collection invariants depend on it.
- `std::hash::Hash` — `(name_bytes, ino)`, a subset of the `Eq` fields (equal entries hash equal). SipHash via the default hasher: not stable across processes; not safe for persistence. The `dirfd` is **deliberately omitted** — including it would invalidate hashes whenever the directory is reopened with a different fd.
- `custom_xxh3::Xxh3Hashable` — stable, hashes `(name_bytes, inode, typenum)`, where `typenum` is the kernel `DT_*` value (pinned `TYPENUM_*` constants, never nix's enum discriminant). This is the hash used for `DirectoryState` change detection, where stability across reopens is the whole point. `xxh3_digest()` hashes the name one-shot, then that digest (xor `typenum`) and the inode as 16 bytes through a `QuickXxh3Hasher`, from registers - no copy of the name, no streaming hasher. It is therefore not the value `xxh3()` streams into a hasher (the trait does not require that); `DirectoryState` uses only `xxh3_digest()`. `entry_digest_covers_name_ino_type` checks that each field moves the digest on its own and pins a value. See [state-tracking.md](state-tracking.md).
