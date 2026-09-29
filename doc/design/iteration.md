# Directory iteration

`DirHandle` exposes four iteration entry points plus a low-level escape hatch. All of them filter out `.` and `..` and yield `EntryExt`.

Entries whose file type cannot be determined (`d_type` is `DT_UNKNOWN` and the `fstatat` fallback fails) are **yielded, not dropped** — their `file_type()` returns `None` and they classify as non-directories everywhere (lookahead heuristic, `entries()`, state counts/hashes). Dropping them would make a listable-but-unsearchable directory iterate as empty on filesystems that don't populate `d_type`.

On such `d_type`-less filesystems (some XFS configurations, NFS, FAT), note that plain `iter()` effectively degrades to `iter_stat()`: the dir-first heuristic and the type classification have to `fstatat` each entry once to decide. The result is cached per entry, but the cost is one syscall per entry either way.

## Public surface

- `iter()` — lookahead-buffered iterator that **preferentially** yields directory entries before others. Rewinds the inner `nix::dir::Iter` when exhausted, so the handle can be iterated repeatedly.
- `iter_stat()` — same as `iter()`, but `stat()`s each entry eagerly before yielding.
- `iter_untracked()` — a plain pass (straight `readdir` order, see below) that also skips state tracking: no `DirectoryState` is computed, so no directory `fstat` and no per-entry digest. For scanners that collect or partition the entries themselves and never use the handle's state; the handle's `DirectoryState` is left as it was.
- `iter_sorted()` — materialises both dir and file lists, sorts each alphabetically, and yields all directories first then all files. Returns a `DirHandleIterSorted` that owns the materialised vec.
- `unsafe raw_iter()` — peekable view of the raw `nix::dir::Iter`, including `.`/`..`. Marked `unsafe` to flag the thread-safety constraint described below.

There is also `entries(dirs, files)` / `entries_sorted()` for non-iterator access, returning `(EntryVec, EntryVec)` tuples.

`entries()`, and therefore `iter_sorted()` / `entries_sorted()`, plus the internal `directory_state()` pass run `DirHandleIter` in **plain mode**: a straight pass with no lookahead buffer, since the result gets partitioned and usually sorted afterwards anyway. Plain mode keeps the `.`/`..` filtering, the sticky-error semantics and the lazy `DirectoryState` finalisation; only the dir-first reordering is skipped. `iter_untracked()` is plain mode with the finalisation switched off as well.

## Dir-first lookahead

`DirHandleIter` keeps a `BufDeque<EntryExt>` with capacity `LOOKAHEAD_BUFFER_SIZE = 64`. `BufDeque::push` routes directory entries to the front and everything else to the back. On `next()`:

1. If the buffer has any entry, `try_pop_dir` pops the front entry. Buffered directories always form a prefix of the queue (`push` is the only way in), so when the dir counter is non-zero the front entry *is* the first directory — O(1), no scan. The pop also reports whether the entry is a directory, so `next()` never re-derives that.
2. If the popped entry is a directory, return it.
3. Otherwise, try to peek the next raw entry. If `dirent.d_type` says it's a directory, push the file back into the buffer and return the directory. If `d_type` is `DT_UNKNOWN`, fetch one more entry to find out — push whichever loses back into the buffer.

The result is a **heuristic preference**, not a guarantee. Some filesystems never populate `d_type`, so the iterator may have to materialise the next entry to decide ordering. Calling code must not rely on strict dir-before-file ordering — for that, use `iter_sorted()`.

## Error semantics

A `readdir` error terminates the pass. Errors must **not** be skipped-and-continued: a persistently failing stream (`ESTALE` on NFS, `EIO` on a dying disk) would otherwise spin the skip loop forever, one failing syscall per iteration. The free `next()` helper leaves the error **unconsumed** in the `Peekable` slot, making it sticky: `done()` treats a peeked `Err` as end-of-stream, and repeated calls return `None` from the cached peek without re-issuing syscalls. An error-terminated pass yields an incomplete listing, so `DirHandleIter` skips `DirectoryState` finalisation in that case (`when` stays `None`, a later clean pass computes it). `for_each` uses `map_while(Result::ok)` for the same stop-on-first-error behaviour.

All of this depends on nix actually reporting the errors, which requires nix ≥ 0.31 (`readdir` plus an errno check). nix 0.30 used `readdir_r`, which returns its error number instead of `-1`, and nix only checked for `-1` — every `readdir` error surfaced as a clean end of stream, so partial listings were finalised as the directory's state. glibc's `readdir_r` also silently skipped names longer than 255 bytes (possible on CIFS, ntfs3 and FUSE), which `readdir` returns. The integration test `readdir_error_ends_pass_without_finalizing` provokes a real `getdents64` failure by `dup2`-ing a regular file over the stream's fd.

## Rewind semantics

nix's `Iter` rewinds the underlying `Dir` (via `rewinddir`) in its `Drop` impl — unconditionally, whether the iterator was exhausted or dropped early. A partially-consumed `DirHandleIter` therefore does **not** leave the `Dir` mid-stream; the next `iter()` call always starts from the beginning.

`DirectoryState` finalisation is stricter than rewinding: it only happens when a pass runs to clean exhaustion. Dropping the iterator early, or a `readdir` error ending the pass, skips the state update (`when` stays `None`), so a later complete pass computes it instead.

## Thread-safety

`DirHandle` asserts `Send` but is **not** `Sync`. `readdir` is not safe to call concurrently against the same `DIR*`, and POSIX is expected to make `readdir_r` obsolete. Move a `DirHandle` across threads if you must; never share it. `raw_iter()` is marked `unsafe` specifically to force callers to acknowledge this when bypassing the safer wrappers.
