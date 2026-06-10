# Directory state and change detection

Each `DirHandle` carries a `DirectoryState` snapshot:

| Field    | Meaning |
| -------- | ------- |
| `dirs`   | Count of directory entries (excluding `.`/`..`). |
| `files`  | Count of non-directory entries, **including** entries whose type could not be determined. |
| `hash_d` | Stable combined xxh3 digest of the directory entries. |
| `hash_f` | Stable combined xxh3 digest of the non-directory entries. |
| `when`   | Timestamp of the most recent snapshot, or `None` if never populated. |

`when` participates in neither `PartialEq` nor `Hash` — two snapshots with identical content compare (and hash) equal regardless of when they were taken.

## When state is computed

State population is **lazy and one-shot per handle** by default. `DirHandleIter::new` enables its `update` flag only when `state.when.is_none()` — i.e., the very first time the handle is iterated. Subsequent calls to `iter()` do **not** refresh the state, even if the directory has changed externally. Only a clean, complete pass finalises the state: early drops and `readdir` errors skip the update.

To force a refresh:

- `state_current()` — returns a fresh `DirectoryState` without touching the stored one. It iterates with `update` explicitly off, so it has no side effect even on a never-iterated handle.
- `state_changed()` — computes fresh, compares, updates stored, returns a bool. The **first call establishes the baseline and returns `false`**.

Both return `io::Result`: a `readdir` error ending the listing early surfaces as the error instead of a partial listing masquerading as the directory's state. On error the stored baseline is left untouched.

`state_changed_fast()` adds a timestamp pre-check in front of the full comparison: a directory's own mtime changes exactly when its entry list changes — which is precisely what `DirectoryState` tracks — so if the directory's mtime *and* ctime are both clearly older than the stored baseline (`MTIME_SLACK_SECS` of slack for filesystem granularity, `f64` rounding and clock skew), it reports "unchanged" after one `fstat` instead of a full re-list + re-hash. Backdating the directory mtime (`touch -d` / `utimensat`) does **not** defeat the pre-check: those calls bump ctime, which is also considered (covered by an integration test). Only direct clock manipulation or broken ctime semantics could produce a false "unchanged".

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

Each entry contributes its per-entry `xxh3_digest()` (a `u64` over `(name_bytes, ino, typenum)`, explicitly **not** the dirfd — see [entry-ext.md](entry-ext.md)); the digests are then **sorted** and fed into one final hasher by `digest_of_digests()`. Sorting makes the result independent of `readdir` order, and accumulating 8 bytes per entry (rather than cloning every `EntryExt` until the pass completes, as earlier versions did) keeps memory flat for huge directories. `digest_of_digests()` is the single source of truth — both the lazy in-iterator finalisation and `directory_state()` go through it, so the two paths always produce comparable values.

A handle closed and reopened against the same directory produces identical hashes if the contents are unchanged. Note that the digest *scheme* changed in v0.4.0 (sorted per-entry digests instead of hashing name-sorted entries in sequence): hash values are stable going forward but do not match those produced by earlier versions.

## `hash_all()`

`DirectoryState::hash_all()` returns `hash_d.rotate_left(32) ^ hash_f` — a cheap combined digest for callers that want a single u64 covering both halves of the state. Note this is **not** a cryptographic combination and collisions between e.g. `(hash_d=A, hash_f=B)` and `(hash_d=B, hash_f=A.rotate_right(32))` are trivially constructible. Treat it as a fingerprint, not an identity.
