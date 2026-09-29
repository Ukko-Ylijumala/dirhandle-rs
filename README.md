# DirHandle — A Rust Directory Handling Utility

> [!WARNING]
> **WORK IN PROGRESS — pre-1.0, no stability guarantees.**
> This crate is at version `0.3.x` and the public API is **actively churning**.
> Breaking changes between minor and patch versions are expected.
> There are currently **no automated tests** and the crate is **not published to crates.io**.
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
- **Hardened entry open** — `openat2` with `RESOLVE_BENEATH` rejects path traversals at the kernel boundary.
- **Change detection (`DirectoryState`)** tracking dir/file counts and stable `xxh3` hashes.
- **Lookahead iteration (`DirHandleIter`)** that preferentially yields directory entries before files.
- **Thread-safe handle pool (`OpenHandles`)** built on `DashMap` with explicit checkout semantics.
- **Optional memory accounting** via the `size_of` cargo feature.

## Project status

This library started life as a component inside a larger application and was extracted into its own crate. As of `0.5.x`:

- The public API is **unstable**. Method signatures, field names, and type shapes may change without notice (`0.5.0` removed `EntryExt`'s `Deref<Target = nix::dir::Entry>`, for example).
- The stored `DirectoryState` hash values are stable within a minor series only; the digest scheme changed in `0.4.0` and `0.5.0` (see [`doc/design/state-tracking.md`](doc/design/state-tracking.md)).
- There is a test suite (`cargo test`: unit tests in `src/lib.rs`, integration tests in `tests/`), but it only runs on the developer's machine.
- The crate is **not published to crates.io** (`publish = false`). It is consumed as a git dependency only.
- Several transitive dependencies (`custom_xxh3`, `timesince`, `miniutils`, `enhvec`, and a temporary fork of `size-of`) are also git-only. Expect occasional build breakage if those repos move.
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
- `0.5.3` — current. Fixes: `state_changed_fast()` compares the directory's timestamps with those recorded by the baseline pass instead of with the local clock, so server clock skew or a stale NFS attribute cache can no longer hide a change; `DirHandle::from_fd()` closes the fds it rejects instead of leaking them; `size_of` charges each handle its actual glibc stream buffer (`st_blksize`, 32 KiB .. 1 MiB) instead of a fixed 32 KiB.
