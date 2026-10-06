# DirHandle — A Rust Directory Handling Utility

> [!WARNING]
> **WORK IN PROGRESS — pre-1.0, no stability guarantees.**
> This crate is at version `0.6.x` and the public API is **actively churning**.
> Breaking changes are expected; they come with a minor version bump (`0.5.x` → `0.6.0`), while patch releases stay non-breaking.
> The test suite only runs on the developer's machine (no CI), and the crate is **not published to crates.io**.
> Do not pin this in production code that you cannot easily update.

> [!IMPORTANT]
> **Linux-only.** The implementation depends on Linux-specific syscalls
> (`openat2`, `fstatat`) via the [`nix`](https://crates.io/crates/nix) crate and on
> `/proc/self/fd` to resolve paths from file descriptors. It will not build or
> run on Windows, and macOS support is untested and unlikely to work.

## Overview

**DirHandle** is a Rust library for efficient directory traversal and change detection on Linux. It wraps `nix::dir` entries in a compact type with metadata caching, stable hashing, and prioritized iteration, and provides a thread-safe pool for managing many open directory handles at once.

## Features

- **Atomic file-descriptor wrapper (`DirFd`)** with sign-encoded open / uninitialized / stale state.
- **Extended directory entries (`EntryExt`)** with `OnceLock`-cached `stat()` results and `std::fs::DirEntry`-compatible accessors.
- **Hardened entry open** — `openat2` with `RESOLVE_BENEATH` rejects path traversals at the kernel boundary, and `open_dir()` never follows symlinks, so tree walks cannot loop or be redirected. `read_nofollow()` / `open_regular()` open files the same way (no symlink, no blocking on a FIFO, regular files only), and `DirHandle::open_at()` / `open_beneath()` / `read_nofollow_at()` / `open_regular_at()` do it from a directory fd and a name or relative path, past `PATH_MAX` too.
- **Change detection (`DirectoryState`)** tracking dir/file counts and stable `xxh3` hashes.
- **Lookahead iteration (`DirHandleIter`)** that preferentially yields directory entries before files.
- **Thread-safe handle pool (`OpenHandles`)** built on `DashMap` with explicit checkout semantics.
- **Optional memory accounting** via the `size_of` cargo feature.

## Project status

This library started life as a component inside a larger application and was extracted into its own crate. As of `0.6.x`:

- The public API is **unstable**. Method signatures, field names, and type shapes may change with any minor version (`0.5.0` removed `EntryExt`'s `Deref<Target = nix::dir::Entry>`, `0.6.0` moved to `nix` 0.31, whose types appear in the API).
- The stored `DirectoryState` hash values are stable within a minor series only; the digest scheme changed in `0.4.0` and `0.5.0`, and `0.6.0` kept it (see [`doc/design/state-tracking.md`](doc/design/state-tracking.md)).
- There is a test suite (`cargo test`: unit tests in `src/lib.rs`, integration tests in `tests/`), but it only runs on the developer's machine.
- The crate is **not published to crates.io** (`publish = false`). It is consumed as a git dependency only.
- Several dependencies (`custom_xxh3`, `timesince`, `miniutils`, `enhvec`, and a temporary fork of `size-of`) are also git-only. Expect occasional build breakage if those repos move.
- Tested on Linux with recent stable Rust. **No CI is configured.**

If any of the above is a dealbreaker for your use case, please wait for a `1.0` release before depending on this crate.

## Requirements

- Linux kernel ≥ 5.6 (for `openat2`).
- `/proc` mounted (procfs).
- Rust ≥ 1.89 (older versions may work; not tested).

## Installation

```toml
[dependencies]
dirhandle = { git = "https://github.com/Ukko-Ylijumala/dirhandle-rs" }
```

`nix` types (`Entry`, `Type`, `Errno`, ...) are part of the API. The crate re-exports the `nix` it was built with as `dirhandle::nix`, so name them through that rather than through a separate `nix` dependency that may be a different version.

Enable optional memory accounting:

```toml
[dependencies]
dirhandle = { git = "https://github.com/Ukko-Ylijumala/dirhandle-rs", features = ["size_of"] }
```

## Usage

### Open a directory and iterate

```rust
use std::path::Path;
use dirhandle::DirHandle;

let path = Path::new("/some/directory");
let mut handle = DirHandle::new(path).expect("Failed to open directory");

for entry in handle.iter() {
    println!("Found: {}", entry.name());
}
```

### Detect changes between scans

```rust
let _ = handle.iter().count(); // populate initial state
// ... time passes, directory may change ...
if handle.state_changed()? {
    println!("Directory contents changed!");
}
```

`state_changed()` re-scans the directory (`state_changed_fast()` first `fstat`s the directory and skips the re-scan when its mtime/ctime are exactly those recorded by the baseline pass, and were already settled back then); neither subscribes to inotify or similar. See [`doc/design/state-tracking.md`](doc/design/state-tracking.md) for the change-detection model and its lazy-population semantics.

### Manage many handles concurrently

```rust
use std::path::Path;
use dirhandle::OpenHandles;

let handles = OpenHandles::new();
if let Ok(mut handle) = handles.open(Path::new("/some/directory")) {
    for entry in handle.iter_sorted() {
        println!("Sorted entry: {}", entry.name());
    }
}
```

> [!CAUTION]
> Holding more than one checked-out handle on the same thread can deadlock — see
> [`doc/design/open-handles.md`](doc/design/open-handles.md) for the locking rules.

## Architecture

For non-trivial integration or contribution, read the design notes under [`doc/design/`](doc/design/):

- [`dirfd.md`](doc/design/dirfd.md) — atomic fd wrapper and its safety contract.
- [`entry-ext.md`](doc/design/entry-ext.md) — entry extensions, cached stat, safe open.
- [`iteration.md`](doc/design/iteration.md) — iterator variants and lookahead behavior.
- [`state-tracking.md`](doc/design/state-tracking.md) — change detection and hash stability.
- [`open-handles.md`](doc/design/open-handles.md) — handle pool and concurrency rules.

## License

Copyright (c) 2024–2026 Mikko Tanner. All rights reserved.

License: MIT OR Apache-2.0

## Contributing

Contributions are welcome, but please note that the API is still in flux. Opening an issue before submitting larger changes is recommended.

## Version history

- `0.3.5` — initial extracted-library release: split `DirHandle` code from a larger application.
- `0.3.8` — `EntryExt::name()` returns `String`, `AsFd` impl on `DirFd`, `nix` 0.30, temporary `size-of` fork to work around Rust ≥ 1.89 E0570.
- `0.3.9` — `DirFd` state machine re-encoded so fd 0 is a valid open fd, atomic `set`/`clear`, `as_fd()` panics instead of UB on a closed fd; `EntryExt` lifetime-bound to its `DirHandle`; `DT_UNKNOWN` entries are yielded instead of dropped; `StateChange` deltas are positive for "added".
- `0.3.10` — correctness fixes for path resolution (non-UTF-8 names, procfs detection), error handling and fd hygiene (`O_CLOEXEC`, `O_DIRECTORY`).
- `0.4.0` — entry identity over explicit `(name, ino, dirfd)` fields; sticky `readdir` errors end a pass instead of spinning; state finalised only on clean passes. **Digest scheme change** (sorted per-entry digests).
- `0.4.1` — `EntryExt::open_dir()` (`openat2` + `RESOLVE_BENEATH` descent), `DirHandle::from_fd()`, `DirHandle::stat()`/`mtime()`, `state_changed_fast()`, `uid`/`gid`/`nlink`/`mtime`/`atime`/`ctime` accessors.
- `0.4.2` — unit and integration test harnesses.
- `0.4.3` — fixes: `DirectoryState.when` stamped at pass start (mid-pass changes could otherwise evade `state_changed_fast()` permanently), atomic insert+checkout in `OpenHandles::open`, `size_of` accounting. Perf: plain iteration mode for `entries()`/`iter_sorted()`, O(1) lookahead pop, allocation-free `CheckedOutHandle`.
- `0.5.0` — **breaking**: compact `EntryExt` (≤ 80 bytes, no `Deref<Target = Entry>`; `file_name()`/`ino()`/`d_type()` are inherent, `new()` takes `&Entry`). **Digest scheme change**: commutative fold instead of sort-then-hash, `typenum()` yields kernel `DT_*` values.
- `0.5.1` — README refresh, `d_type` fallback test, clippy sweep, `EntryExt::open()` without `unsafe`.
- `0.5.2` — `DirHandle::path()` / `DirFd::path()` / `EntryExt::path()` fail with `NotFound` once the directory has been deleted, instead of returning procfs's `"<path> (deleted)"` string.
- `0.5.3` — fixes: `state_changed_fast()` compares the directory's timestamps with those recorded by the baseline pass instead of with the local clock, so server clock skew or a stale NFS attribute cache can no longer hide a change; `DirHandle::from_fd()` closes the fds it rejects instead of leaking them; `size_of` charges each handle its actual glibc stream buffer (`st_blksize`, 32 KiB .. 1 MiB) instead of a fixed 32 KiB.
- `0.6.0` — **Breaking**: `nix` 0.31 — with 0.30, `readdir` errors looked like a clean end of the listing, so partial listings were stored as the directory's state and never reported; nix types are part of the API, so the crate re-exports `dirhandle::nix`. `EntryExt::open_dir()` no longer follows symlinks, not even in-tree ones (`ENOTDIR`). `DirFd::new()` takes `&Fd` (by value it closed an owned fd on the spot). Perf: ~10% faster iteration from nix 0.31, one-shot per-entry digests (~2x faster, same values).
- `0.6.1` — `DirHandle::iter_untracked()`: a plain pass in `readdir` order without state tracking (no directory stamp, no digests).
- `0.6.2` — enhvec 0.6.5 / custom_xxh3 0.4.4. **Digest scheme change**: per-entry digest in two one-shot passes (~2.5x faster per entry). `iter_sorted()` yields from a plain `vec::IntoIter` (`ExactSizeIterator`, no longer keeps the handle borrowed while unused). Note: `entries()` / `entries_sorted()` pairs now keep the handle borrowed until dropped (`EnhVec` has its own `Drop`).
- `0.6.3` — current. Opens relative to a directory fd without following symlinks: `EntryExt::read_nofollow()` / `open_regular()` (the latter also fills the entry's cached stat), free functions `read_nofollow_at()` / `open_regular_at()`, `DirHandle::open_at()` (what `open_dir()` now delegates to), `DirHandle::open_beneath()` (a relative path with no symlink in any component, resolved in chunks past `PATH_MAX`), `DirHandle::path_fd()` / `path_fd_at()` (an `O_PATH` fd of a directory, or of a directory entry, that outlives the handle) and `AsFd` for `DirHandle`.
