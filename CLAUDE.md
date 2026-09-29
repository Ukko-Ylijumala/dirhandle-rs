# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Build: `cargo build` (release: `cargo build --release`)
- Build with the optional accounting feature: `cargo build --features size_of`
- Lint: `cargo clippy --all-targets` (add `--features size_of` to lint gated code too)
- Format: `cargo fmt`
- Tests: `cargo test` runs unit tests (in `src/lib.rs` under `#[cfg(test)]`, covering private internals: `DirFd` state machine, `digest_of_digests`, `StateChange` ordering) and integration tests (`tests/integration.rs`, covering the public API against real temp directories). Run them before committing changes to iteration, state tracking, or fd handling.

## Crate shape

- Single-file library: all code lives in `src/lib.rs`. There is no `mod` tree and no examples/benches directories — additions of public API go here. Integration tests live in `tests/integration.rs`; the only dev-dependency is `libc` (integration tests are separate crates and cannot see the library's own deps).
- `publish = false` in `Cargo.toml`; downstream projects consume this crate via git dependency, not crates.io.
- Several dependencies (`custom_xxh3`, `timesince`, `miniutils`, `enhvec`, and a fork of `size-of`) are pulled from `github.com/Ukko-Ylijumala/*` git repos. The `size-of` fork specifically exists to work around a Rust ≥1.89 compiler error (E0570) in upstream — do not switch back to upstream `size-of` without verifying the fix is published.
- Linux-only: depends on `nix` (`fs` + `dir` features), `libc::stat`/`mode_t`, and `/proc/self/fd` for fd→path resolution. Anything that breaks procfs availability breaks `DirFd::path()` and `DirHandle::path()` by design (they return `io::Error` rather than panicking).
- There is no crate-wide `#![allow(dead_code)]`; the few intentionally unused helpers carry `#[expect(dead_code)]`, which fails the build if the helper gains a caller — remove the attribute when wiring one up.

## Design docs

The interesting design decisions are split out under `doc/design/`. Read the relevant file before making non-trivial changes to that subsystem:

- [`doc/design/dirfd.md`](doc/design/dirfd.md) — `DirFd`: atomic, sign-encoded fd state machine and its `AsFd` soundness contract.
- [`doc/design/entry-ext.md`](doc/design/entry-ext.md) — `EntryExt`: `OnceLock` stat caching, `openat2 + RESOLVE_BENEATH`, dual hash protocols.
- [`doc/design/iteration.md`](doc/design/iteration.md) — `DirHandleIter` lookahead heuristic, rewind semantics, and thread-safety rules.
- [`doc/design/state-tracking.md`](doc/design/state-tracking.md) — `DirectoryState`, the `StateChange` ordering, and the lazy-population model.
- [`doc/design/open-handles.md`](doc/design/open-handles.md) — `OpenHandles` / `CheckedOutHandle`, the same-thread deadlock caveat, and why the callback uses `Rc` not `Arc`.

## Editing pitfalls

- The `SizeOf` impl charges each handle its glibc directory stream: `DIR_STREAM_HEADER` plus the inline `getdents` buffer that `opendir`/`fdopendir` size as `st_blksize` clamped to `DIR_STREAM_BUF_MIN..=DIR_STREAM_BUF_MAX` (32 KiB .. 1 MiB). `st_blksize` is not a per-filesystem constant (ZFS: up to 128 KiB and varying per directory; NFS: often 1 MiB), so `DirHandle::from_dir()` records it with one `fstat` at open, gated on the feature. Every constructor must go through `from_dir()`. Re-verify if glibc's `sysdeps/posix/opendir.c` changes.
- `EntryExt::typenum()` uses the pinned `TYPENUM_*` constants (kernel `DT_*` values), never `Type as u8`: the values are part of the stable `DirectoryState` digests. The digest scheme itself lives in `DigestFold`; changing either breaks hash compatibility and needs a version note in `doc/design/state-tracking.md`.
- `tracing` is used with explicit `target = "..."` strings throughout. Preserve targets when adding or moving log statements so downstream filters keep working.
- nix's `Iter` rewinds the underlying `Dir` on drop (early or exhausted), so a partially-consumed `DirHandleIter` never leaves the `Dir` mid-stream. `DirectoryState` finalisation is stricter: it only happens on a clean, complete pass — early drops and `readdir` errors skip the state update.
- `EntryExt` does not retain the `nix::dir::Entry` (v0.5.0): `new()` copies out name, inode and `d_type` into a ≤80-byte struct (`EntryName` keeps names ≤38 bytes inline). Keep identity on our own fields: up to nix 0.30 the `Entry` wrapped a partially uninitialized dirent whose libc derives read garbage, and nix 0.31 changed its representation again (owned `CString` name).
- nix must stay ≥ 0.31. 0.30's `Dir` iterator used `readdir_r` and only checked for a `-1` return, so every `readdir` error looked like a clean end of stream: partial listings were finalised as `DirectoryState` and `error()` / `state_current()` never reported anything. `readdir_error_ends_pass_without_finalizing` guards this. nix types appear in the public API, hence `pub use nix;`.
- `unsafe impl Sync for OpenHandles` is sound only while every `&self` method on `DirHandle` stays away from the underlying `DIR*` stream (no readdir/telldir/seekdir). Anything touching stream position must take `&mut self`.
