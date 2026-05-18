# Directory state and change detection

Each `DirHandle` carries a `DirectoryState` snapshot:

| Field    | Meaning |
| -------- | ------- |
| `dirs`   | Count of directory entries (excluding `.`/`..`). |
| `files`  | Count of non-directory entries. |
| `hash_d` | Stable xxh3 hash of the sorted directory entries. |
| `hash_f` | Stable xxh3 hash of the sorted non-directory entries. |
| `when`   | Timestamp of the most recent snapshot, or `None` if never populated. |

## When state is computed

State population is **lazy and one-shot per handle** by default. `DirHandleIter::new` sets its internal `xxh` to `Some(...)` only when `state.when.is_none()` — i.e., the very first time the handle is iterated. Subsequent calls to `iter()` do **not** refresh the state, even if the directory has changed externally.

To force a refresh:

- `state_current()` — returns a fresh `DirectoryState` without touching the stored one.
- `state_changed()` — computes fresh, compares, updates stored if different, returns a bool.

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

Hashes are computed over entries **sorted ascending** before feeding the hasher (see the finalisation block at the end of `DirHandleIter::next`). Stability across reopens is provided by `EntryExt::xxh3` hashing `(name_bytes, ino, typenum)` and explicitly **not** the dirfd — see [entry-ext.md](entry-ext.md). This means a handle closed and reopened against the same directory produces identical hashes if the contents are unchanged.

## `hash_all()`

`DirectoryState::hash_all()` returns `hash_d.rotate_left(32) ^ hash_f` — a cheap combined digest for callers that want a single u64 covering both halves of the state. Note this is **not** a cryptographic combination and collisions between e.g. `(hash_d=A, hash_f=B)` and `(hash_d=B, hash_f=A.rotate_right(32))` are trivially constructible. Treat it as a fingerprint, not an identity.
