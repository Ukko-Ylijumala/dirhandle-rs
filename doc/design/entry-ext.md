# EntryExt — enriched directory entry

`EntryExt<'h>` wraps `nix::dir::Entry` and aims to feel like `std::fs::DirEntry` with a few additions. It is the unit yielded by every public iterator on `DirHandle`.

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

The entry holds an `OnceLock<Option<libc::stat>>`. `stat()` initialises it lazily; subsequent calls return the cached value. `stat_refresh()` clears the lock and re-stats. The `Option` inside the lock means "we already tried and failed" is cached too — repeated stat failures do not retrigger the syscall.

`stat()` is the only path that can promote an entry's known type from `None` to something useful: when `dirent.d_type` is `DT_UNKNOWN` (some filesystems never populate it), `file_type()` falls back to `EntryType(self.mode()).entry_t()`. The local `EntryType` struct exists because `std::sys::pal::unix::fs::FileType`, which performs the same `S_IFMT` masking, is private and cannot be imported.

Because the `BorrowedFd<'h>` guarantees the parent fd is alive for the entry's lifetime, `stat()` and `open()` do **not** need explicit liveness guards — the type system has already ruled out the closed-fd case for safe code paths.

## Safe `openat2`

`read()` and `write()` open the entry via `nix::fcntl::openat2` with `ResolveFlag::RESOLVE_BENEATH`. The kernel rejects any path that would resolve outside the parent dirfd — symlink loops, `..` traversals, or absolute paths. Do not "simplify" this to plain `openat` or `open` without a deliberate reason; it silently broadens the trust boundary.

## Hashing

`EntryExt` implements two distinct hash protocols:

- `std::hash::Hash` — delegates to `Entry::hash` (SipHash via the default hasher). Not stable across processes; not safe for persistence. The `dirfd` is **deliberately omitted** from the hash — including it would invalidate hashes whenever the directory is reopened with a different fd. The in-source TODO acknowledges this is worth re-examining.
- `custom_xxh3::Xxh3Hashable` — stable, hashes `(name_bytes, inode, typenum)`. This is the hash used for `DirectoryState` change detection, where stability across reopens is the whole point. See [state-tracking.md](state-tracking.md).

## `Deref<Target = Entry>`

`EntryExt` derefs to the underlying `nix::dir::Entry`, so all of nix's `Entry` accessors (`file_name`, `ino`, etc.) work directly. Don't shadow these on `EntryExt` unless you have a reason to diverge from nix's semantics.
