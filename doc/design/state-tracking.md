# Directory state and change detection

Each `DirHandle` carries a `DirectoryState` snapshot:

| Field    | Meaning |
| -------- | ------- |
| `dirs`   | Count of directory entries (excluding `.`/`..`). |
| `files`  | Count of non-directory entries, **including** entries whose type could not be determined. |
| `hash_d` | Stable combined xxh3 digest of the directory entries. |
| `hash_f` | Stable combined xxh3 digest of the non-directory entries. |
| `when`   | Timestamp of the most recent snapshot — taken at the **start** of the pass that produced it, before the first `readdir` — or `None` if never populated. |
| `stamp`  | The directory's own `(mtime, ctime)` timespecs from an `fstat` just before that pass (`DirStamp`), or `None` if never populated or the `fstat` failed. Feeds the `state_changed_fast()` pre-check. |

`when` and `stamp` participate in neither `PartialEq` nor `Hash` — two snapshots with identical content compare (and hash) equal regardless of when they were taken.

## When state is computed

State population is **lazy and one-shot per handle** by default. `DirHandleIter::new` enables its `update` flag only when `state.when.is_none()` — i.e., the very first time the handle is iterated. Subsequent calls to `iter()` do **not** refresh the state, even if the directory has changed externally. Only a clean, complete pass finalises the state: early drops and `readdir` errors skip the update. `iter_untracked()` passes never do (they neither take the directory stamp nor digest the entries), so a handle iterated only that way keeps an empty `DirectoryState` until a tracked pass or `state_changed()` establishes one.

To force a refresh:

- `state_current()` — returns a fresh `DirectoryState` without touching the stored one. It iterates with `update` explicitly off, so it has no side effect even on a never-iterated handle.
- `state_changed()` — computes fresh, compares, updates stored, returns a bool. The **first call establishes the baseline and returns `false`**.

Both return `io::Result`: a `readdir` error ending the listing early surfaces as the error instead of a partial listing masquerading as the directory's state. On error the stored baseline is left untouched.

`state_changed_fast()` adds a timestamp pre-check in front of the full comparison. A directory's own mtime changes exactly when its entry list changes — which is precisely what `DirectoryState` tracks — so every snapshot pass records the directory's exact `(mtime, ctime)` (`stamp`). The pre-check reports "unchanged" after a single `fstat` only if the directory's timestamps are **identical** to that stamp **and** the stamp was settled, i.e. older than the pass start (`when`) by more than `MTIME_SLACK_SECS`. Anything else falls through to the full re-list + re-hash.

- **Stamp vs. stamp, not stamp vs. wall clock.** On NFS / SMB / FUSE the directory timestamps come from the server's clock. Up to 0.5.2 the pre-check compared them with the local `when`, so a server clock lagging ours by more than the slack — or a listing served from a stale attribute cache — made a real change look "clearly older than the baseline", and it stayed invisible until some later change. Two stamps come from the same clock, so skew cancels out.
- **Racy baselines.** A change landing in the same timestamp granule as the stamp's `fstat` does not move the timestamps (coarse kernel clocks tick every few ms; some filesystems store whole seconds). An unchanged stamp therefore only counts once it is settled; a racy baseline always gets the full comparison, which refreshes the stamp. The slack covers granularity, `f64` rounding and minor skew — skew now only affects this raciness guess, not change detection itself.
- **Backdating** the directory mtime (`touch -d` / `utimensat`) does not defeat the pre-check: any timestamp difference forces the full comparison, and those calls bump ctime too (covered by an integration test).

Both `stamp` and `when` are taken at the *start* of the pass. A change that lands mid-pass may or may not have been seen by `readdir`, but it moves the timestamps past the stamp and so forces the full check. Stamping the end of a pass would let such a change slip through — permanently, since the pre-check never touches the stored state (regression-tested in `state_when_is_stamped_at_pass_start`). `state_changed()` refreshes `when` and `stamp` even when the content is unchanged: a stale stamp would never match the directory again and silently disable the pre-check.

This is a deliberate trade-off: most callers iterate to consume entries, not to recompute hashes on every pass. If you add a new iteration entry point, decide explicitly whether it should participate in state tracking and follow the existing pattern.

## StateChange enum

`DirectoryState::change(&other)` returns the first detected difference, in order:

1. `DirNum(delta)` — directory count differs. `delta = other.dirs - self.dirs`; positive means "added since `self`," negative means "removed."
2. `FileNum(delta)` — file count differs. Same sign convention as `DirNum`.
3. `DirHash` — same counts, different directory-entry hash.
4. `FileHash` — same counts, different file-entry hash.
5. `Unchanged` — all four fields match.

Because the comparison short-circuits, a `DirNum`/`FileNum` result also implies the corresponding hash would have changed. This is intentional — callers usually want the "biggest" signal, and count changes are cheaper to act on than hash changes.

## Hash stability

Each entry contributes its per-entry `xxh3_digest()` (a `u64` over `(name_bytes, ino, typenum)` — since 0.6.0 hashed one-shot from a stack buffer instead of through a streaming hasher, same bytes and same values, pinned by a unit test; explicitly **not** the dirfd — see [entry-ext.md](entry-ext.md); `typenum` is the kernel `DT_*` value via the pinned `TYPENUM_*` constants, not nix's enum discriminant, so an upstream reorder cannot change the digests). The digests are folded into a `DigestFold` as they stream past: a wrapping **sum**, an **xor** and the **count**, mixed through one xxh3 in `finish()`. Both folds are commutative, so the result is independent of `readdir` order, and the accumulator is O(1) memory and O(n) time per pass — no per-entry `Vec` and no `n log n` sort. Either fold alone would let trivially constructed multisets collide (`{a, b}` vs `{c, d}` with `a + b == c + d`); the pair requires simultaneous sum *and* xor equality, and the count separates e.g. `{}` from `{0}` or `{7, 7}`. It is a change-detection fingerprint, not a cryptographic commitment. `DigestFold` is the single source of truth — both the lazy in-iterator finalisation and `directory_state()` go through it (and through the one `DirectoryState::from_pass()` constructor), so the two paths always produce comparable values.

A handle closed and reopened against the same directory produces identical hashes if the contents are unchanged. The digest *scheme* has changed twice, so hash values are stable within a series but not across these boundaries:

- **v0.4.0** — sorted per-entry digests instead of hashing name-sorted entries in sequence.
- **v0.5.0** — commutative fold instead of sort-then-hash, and `typenum` switched from nix's enum order (0..6, unknown 254) to the kernel `DT_*` values.

## `hash_all()`

`DirectoryState::hash_all()` returns `hash_d.rotate_left(32) ^ hash_f` — a cheap combined digest for callers that want a single u64 covering both halves of the state. Note this is **not** a cryptographic combination and collisions between e.g. `(hash_d=A, hash_f=B)` and `(hash_d=B, hash_f=A.rotate_right(32))` are trivially constructible. Treat it as a fingerprint, not an identity.
