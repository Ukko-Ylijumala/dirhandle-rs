# Directory iteration

`DirHandle` exposes three iteration entry points plus a low-level escape hatch. All of them filter out `.` and `..` and yield `EntryExt`.

## Public surface

- `iter()` — lookahead-buffered iterator that **preferentially** yields directory entries before others. Rewinds the inner `nix::dir::Iter` when exhausted, so the handle can be iterated repeatedly.
- `iter_stat()` — same as `iter()`, but `stat()`s each entry eagerly before yielding.
- `iter_sorted()` — materialises both dir and file lists, sorts each alphabetically, and yields all directories first then all files. Returns a `DirHandleIterSorted` that owns the materialised vec.
- `unsafe raw_iter()` — peekable view of the raw `nix::dir::Iter`, including `.`/`..`. Marked `unsafe` to flag the thread-safety constraint described below.

There is also `entries(dirs, files)` / `entries_sorted()` for non-iterator access, returning `(EntryVec, EntryVec)` tuples.

## Dir-first lookahead

`DirHandleIter` keeps a `BufDeque<EntryExt>` with capacity `LOOKAHEAD_BUFFER_SIZE = 64`. `BufDeque::push` routes directory entries to the front and everything else to the back. On `next()`:

1. If the buffer has any entry, `try_pop_dir` pops the first directory found, else the oldest entry.
2. If the popped entry is a directory, return it.
3. Otherwise, try to peek the next raw entry. If `dirent.d_type` says it's a directory, push the file back into the buffer and return the directory. If `d_type` is `DT_UNKNOWN`, fetch one more entry to find out — push whichever loses back into the buffer.

The result is a **heuristic preference**, not a guarantee. Some filesystems never populate `d_type`, so the iterator may have to materialise the next entry to decide ordering. Calling code must not rely on strict dir-before-file ordering — for that, use `iter_sorted()`.

## Rewind semantics

When the inner iterator is exhausted, `DirHandleIter::next` returns `None` after (a) optionally finalising `DirectoryState` (see [state-tracking.md](state-tracking.md)) and (b) leaving the inner `Dir` rewound for the next iteration.

If the consumer drops the iterator before exhaustion, the underlying `Dir` is left mid-stream — there is no rewind-on-drop. State tracking, if it was going to update on this pass, will not update either.

## Thread-safety

`DirHandle` asserts `Send` but is **not** `Sync`. `readdir` is not safe to call concurrently against the same `DIR*`, and POSIX is expected to make `readdir_r` obsolete. Move a `DirHandle` across threads if you must; never share it. `raw_iter()` is marked `unsafe` specifically to force callers to acknowledge this when bypassing the safer wrappers.
