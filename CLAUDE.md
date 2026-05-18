# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Build: `cargo build` (release: `cargo build --release`)
- Build with the optional accounting feature: `cargo build --features size_of`
- Lint: `cargo clippy --all-targets` (add `--features size_of` to lint gated code too)
- Format: `cargo fmt`
- Tests: none currently exist in-tree. `cargo test` will compile the crate but run zero tests.

## Crate shape

- Single-file library: all code lives in `src/lib.rs`. There is no `mod` tree and no examples/benches/tests directories — additions of public API go here.
- `publish = false` in `Cargo.toml`; downstream projects consume this crate via git dependency, not crates.io.
- Several dependencies (`custom_xxh3`, `timesince`, `miniutils`, `enhvec`, and a fork of `size-of`) are pulled from `github.com/Ukko-Ylijumala/*` git repos. The `size-of` fork specifically exists to work around a Rust ≥1.89 compiler error (E0570) in upstream — do not switch back to upstream `size-of` without verifying the fix is published.
- Linux-only: depends on `nix` (`fs` + `dir` features), `libc::stat`/`mode_t`, and `/proc/self/fd` for fd→path resolution. Anything that breaks procfs availability breaks `DirFd::path()` and `DirHandle::path()` by design (they return `io::Error` rather than panicking).
- `#![allow(dead_code)]` is set crate-wide — dead-code warnings will not flag unused helpers.

## Design docs

The interesting design decisions are split out under `doc/design/`. Read the relevant file before making non-trivial changes to that subsystem:

- [`doc/design/dirfd.md`](doc/design/dirfd.md) — `DirFd`: atomic, sign-encoded fd state machine and its `AsFd` soundness contract.
- [`doc/design/entry-ext.md`](doc/design/entry-ext.md) — `EntryExt`: `OnceLock` stat caching, `openat2 + RESOLVE_BENEATH`, dual hash protocols.
- [`doc/design/iteration.md`](doc/design/iteration.md) — `DirHandleIter` lookahead heuristic, rewind semantics, and thread-safety rules.
- [`doc/design/state-tracking.md`](doc/design/state-tracking.md) — `DirectoryState`, the `StateChange` ordering, and the lazy-population model.
- [`doc/design/open-handles.md`](doc/design/open-handles.md) — `OpenHandles` / `CheckedOutHandle`, the same-thread deadlock caveat, and why the callback uses `Rc` not `Arc`.

## Editing pitfalls

- The `SizeOf` impls hard-code `DHSIZE = 296` from the layout of `nix::dir::Dir` + `libc::DIR` + `libc::dirent`. Re-verify if `nix` or `libc` changes representation.
- `tracing` is used with explicit `target = "..."` strings throughout. Preserve targets when adding or moving log statements so downstream filters keep working.
- `DirHandleIter` rewinds the inner `Dir` only when run to exhaustion. If a consumer drops the iterator early, the underlying `Dir` is left mid-stream and state tracking does not update for that pass.
