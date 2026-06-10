// Copyright (c) 2024-2026 Mikko Tanner. All rights reserved.

use custom_xxh3::{CustomXxh3Hasher, Xxh3Hashable};
use dashmap::{mapref::one::RefMut, DashMap};
use enhvec::{EnhVec, Sorting};
use libc;
use miniutils::{ToDebug, ToDisplay};
use nix::{
    dir::{Dir, Entry, Iter, Type},
    errno::Errno,
    fcntl::{openat2, AtFlags, OFlag, OpenHow, ResolveFlag},
    sys::stat::{fstatat, Mode},
};
use std::{
    cmp::{Eq, Ord, Ordering, PartialEq, PartialOrd},
    collections::VecDeque,
    ffi::OsStr,
    fmt::{self, Debug, Display, Formatter},
    fs::{read_link, File, OpenOptions},
    hash::{Hash, Hasher},
    io,
    iter::Peekable,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, RawFd},
    os::unix::ffi::OsStrExt,
    path::{Path, PathBuf},
    rc::Rc,
    sync::{
        atomic::{AtomicI32, Ordering::Relaxed},
        OnceLock,
    },
};
use timesince::TimeSinceEpoch;
use tracing::{debug, instrument, trace, warn};

#[cfg(feature = "size_of")]
use size_of::{Context, SizeOf};

const DOT1: &[u8] = b".";
const DOT2: &[u8] = b"..";
const LOOKAHEAD_BUFFER_SIZE: usize = 64;
const PROC_FD_PATH: &str = "/proc/self/fd";

/**
Since we cannot import [std::sys] directly (it's private), we need to
define our own `EntryType`, which is functionally a copy of the original
`std::sys::pal::unix::fs::FileType`.
*/
#[derive(Debug, Clone, Copy, Hash)]
pub struct EntryType(libc::mode_t);

#[rustfmt::skip]
impl EntryType {
    #[inline]
    pub fn is_dir(&self)     -> bool { self.is(libc::S_IFDIR) }
    #[inline]
    pub fn is_file(&self)    -> bool { self.is(libc::S_IFREG) }
    pub fn is_symlink(&self) -> bool { self.is(libc::S_IFLNK) }
    pub fn is_block(&self)   -> bool { self.is(libc::S_IFBLK) }
    pub fn is_char(&self)    -> bool { self.is(libc::S_IFCHR) }
    pub fn is_sock(&self)    -> bool { self.is(libc::S_IFSOCK) }
    pub fn is_fifo(&self)    -> bool { self.is(libc::S_IFIFO) }

    #[inline]
    fn is(&self, mode: libc::mode_t) -> bool { self.masked() == mode }
    #[inline]
    fn masked(&self) -> libc::mode_t { self.0 & libc::S_IFMT }

    /// Return the file type of the entry as a [nix::dir::Type] enum.
    pub fn entry_t(&self) -> Option<Type> {
        match self.masked() {
            libc::S_IFIFO  => Some(Type::Fifo),
            libc::S_IFCHR  => Some(Type::CharacterDevice),
            libc::S_IFDIR  => Some(Type::Directory),
            libc::S_IFBLK  => Some(Type::BlockDevice),
            libc::S_IFREG  => Some(Type::File),
            libc::S_IFLNK  => Some(Type::Symlink),
            libc::S_IFSOCK => Some(Type::Socket),
            /* libc::DT_UNKNOWN | */ _ => None,
        }
    }
}

/**
Sentinel for "uninitialized DirFd." Picked so it cannot collide with any
real fd: the kernel only hands out non-negative values, and `i32::MIN` is
distinct from every stale-fd encoding (see `clear()` below).
*/
const UNINIT_FD: RawFd = i32::MIN;

/**
A thin wrapper around a [[RawFd]] with some extra functionality.

Notably, a [DirFd] can resolve its path from the file descriptor, and a
negative value indicates that the file descriptor used to be open.

Due to internally using an [AtomicI32], it is thread-safe.

Mapping:
- `DirFd >= 0` : Open file descriptor. `fd == 0` is a valid (open) fd —
  it's stdin's slot, but a process that closed stdin can legitimately
  receive it from `open()`.
- `DirFd == i32::MIN` : Uninitialized (the default state).
- `DirFd < 0` (other than `i32::MIN`) : The fd was previously open and has
  been cleared. The original fd is recoverable as `!stored` (bitwise NOT),
  which keeps stale-from-fd-0 distinguishable from uninitialized.
*/
#[derive(Debug)]
pub struct DirFd(AtomicI32);

impl DirFd {
    pub fn new<Fd: AsRawFd>(fd: Fd) -> Self {
        DirFd(fd.as_raw_fd().into())
    }

    /// Returns the file descriptor.
    #[inline]
    pub fn fd(&self) -> RawFd {
        self.0.load(Relaxed)
    }

    /// Whether the file descriptor is open. fd 0 is a valid (open) fd.
    pub fn is_open(&self) -> bool {
        self.fd() >= 0
    }

    /**
    Set the inner file descriptor.

    - Returns the file descriptor if it was set successfully.
    - If the fd is already set, returns an error with the existing fd.

    In the latter case, the caller should clear the existing fd first.

    Atomic with respect to other `set` / `clear` calls: two concurrent
    `set`s cannot both succeed, and a concurrent `clear` either lands
    before or after — never in between the check and the store.
    */
    pub fn set(&self, fd: RawFd) -> Result<RawFd, RawFd> {
        self.0
            .fetch_update(Relaxed, Relaxed, |current| {
                if current >= 0 { None } else { Some(fd) }
            })
            .map(|_| fd)
    }

    /**
    Clear the file descriptor.

    If the fd is open, we encode it as `!fd` (bitwise NOT) so that even
    `fd == 0` produces a distinct non-zero stale marker (`-1`). If the
    stored value is already stale or uninit, we bury it at [UNINIT_FD].

    Atomic with respect to other `set` / `clear` calls.
    */
    pub fn clear(&self) {
        let _ = self.0.fetch_update(Relaxed, Relaxed, |current| {
            Some(if current >= 0 { !current } else { UNINIT_FD })
        });
    }

    /**
    This relies on proc filesystem being available due to the use of
    `/proc/self/fd` to resolve the path from the file descriptor.

    We return [[io::Error]] on:
    - no file descriptor
    - stale file descriptor
    - procfs not available
    - file descriptor not found in procfs
    */
    pub fn path(&self) -> io::Result<PathBuf> {
        let fd: RawFd = self.fd();
        if fd == UNINIT_FD {
            return Err(io::Error::new(io::ErrorKind::NotFound, "no file descriptor"));
        }
        if fd < 0 {
            return Err(io::Error::new(io::ErrorKind::NotFound, "stale file descriptor"));
        }
        proc_fd_path(fd)
    }
}

impl Clone for DirFd {
    fn clone(&self) -> Self {
        DirFd(self.fd().into())
    }
}

impl Default for DirFd {
    fn default() -> Self {
        DirFd(UNINIT_FD.into())
    }
}

impl PartialOrd for DirFd {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.fd().partial_cmp(&other.fd())
    }
}

impl Ord for DirFd {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.fd().cmp(&other.fd())
    }
}

impl PartialEq for DirFd {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.fd() == other.fd()
    }
}

impl Eq for DirFd {}

impl Hash for DirFd {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.fd().hash(state);
    }
}

impl AsFd for DirFd {
    /**
    Returns a [BorrowedFd] view of the stored fd.

    # Panics

    Panics if the `DirFd` is not in the "open" state (`is_open()` returns
    false). `BorrowedFd::borrow_raw` forbids `-1` and requires the fd to
    be open for the borrow's lifetime; passing a stale or uninitialized
    value here would be undefined behaviour, so we choose a deterministic
    panic over silent UB. Callers that may legitimately hold a closed
    `DirFd` should test `is_open()` before invoking this method.
    */
    fn as_fd(&'_ self) -> BorrowedFd<'_> {
        let fd: RawFd = self.fd();
        assert!(
            fd >= 0,
            "DirFd::as_fd called on closed or uninitialized DirFd (raw: {fd})",
        );
        /*
        SAFETY: We checked `fd >= 0`, satisfying the minimum precondition of
        `BorrowedFd::borrow_raw` (which forbids -1). Per-call OS-level
        openness of the fd remains the caller's responsibility, as documented
        on the method.
        */
        unsafe { BorrowedFd::borrow_raw(fd) }
    }
}

impl AsRawFd for DirFd {
    fn as_raw_fd(&self) -> RawFd {
        self.fd()
    }
}

impl From<RawFd> for DirFd {
    fn from(fd: RawFd) -> Self {
        DirFd(fd.into())
    }
}

impl From<&DirFd> for RawFd {
    fn from(dirfd: &DirFd) -> Self {
        dirfd.fd()
    }
}

/* ######################################################################### */

/**
This struct extends the [nix::dir::Entry] struct with additional methods.
The aim is to be closely compatible with the [std::fs::DirEntry] API.

Notable differences:
- `metadata()` is replaced with `stat()`, and we return a [libc::stat] struct
- `file_type()` is replaced with a custom implementation, which uses `fstatat()`
  if the file type is not available in the `dirent` struct
- the stat result is cached in a `OnceLock` to avoid calling `fstatat()`
  multiple times for the same entry.

The `'h` lifetime ties each entry to the [DirHandle] that produced it via
a [BorrowedFd]: the parent handle's directory fd must remain open for as
long as the entry exists, which the borrow checker enforces. Collecting
entries into a `Vec` keeps the handle borrowed for the lifetime of the
vec, so use-after-close zombies are not constructible from safe code.
*/
#[derive(Debug, Clone)]
pub struct EntryExt<'h> {
    entry: Entry,
    dirfd: BorrowedFd<'h>,
    stat: OnceLock<Option<libc::stat>>,
}

impl<'h> Eq for EntryExt<'h> {}

impl<'h> EntryExt<'h> {
    /// Create an `EntryExt` from a given [Entry] and its parent directory fd.
    #[instrument(level = "trace")]
    pub fn new(entry: Entry, dirfd: BorrowedFd<'h>) -> Self {
        Self {
            entry,
            dirfd,
            stat: OnceLock::new(),
        }
    }

    /// Create a new `EntryExt` and also `stat()` it before returning it.
    pub fn new_statted(entry: Entry, dirfd: BorrowedFd<'h>) -> Self {
        let new_e: EntryExt<'h> = Self::new(entry, dirfd);
        new_e.stat();
        new_e
    }

    /**
    Return the [libc::stat] struct for the entry (may be cached).

    NOTE: If for some reason we cannot stat the entry, we return `None`.
    */
    pub fn stat(&self) -> Option<libc::stat> {
        *self.stat.get_or_init(|| {
            match fstatat(&self.dirfd, self.file_name(), AtFlags::AT_SYMLINK_NOFOLLOW) {
                Ok(stat) => Some(stat),
                Err(_) => None,
            }
        })
    }

    /// Refresh and return the stat result for the entry.
    pub fn stat_refresh(&mut self) -> Option<libc::stat> {
        self.stat.take();
        self.stat()
    }

    /// Whether the entry has been statted.
    pub fn is_statted(&self) -> bool {
        self.stat.get().is_some()
    }

    /**
    Return the size of the file, in bytes.

    Note that the size is returned as a `u64` to match the [std::fs::Metadata]
    API, even though the underlying [libc::stat] struct uses `i64` for the size.
    Also, if for some reason we cannot stat() the file, we return `0`.
    */
    pub fn len(&self) -> u64 {
        match self.stat() {
            Some(stat) => stat.st_size as u64,
            None => 0,
        }
    }

    /// Return the mode of the file as a [libc::mode_t] value (`u32`).
    pub fn mode(&self) -> Option<libc::mode_t> {
        match self.stat() {
            Some(stat) => Some(stat.st_mode),
            None => None,
        }
    }

    /// Open this entry as a [std::fs::File] with the given flags.
    /// `O_CLOEXEC` is always added so the fd does not leak across `exec`.
    fn open(&self, flags: OFlag) -> io::Result<File> {
        let open_how: OpenHow = OpenHow::new()
            .flags(flags | OFlag::O_CLOEXEC)
            .resolve(ResolveFlag::RESOLVE_BENEATH);
        let fd = openat2(&self.dirfd, self.file_name(), open_how)?;
        Ok(unsafe { File::from_raw_fd(fd.into_raw_fd()) })
    }

    /// Open this entry for reading as a [std::fs::File] object.
    pub fn read(&self) -> io::Result<File> {
        self.open(OFlag::O_RDONLY)
    }

    /// Open this entry for read+write as a [std::fs::File] object.
    pub fn write(&self) -> io::Result<File> {
        self.open(OFlag::O_RDWR)
    }

    #[inline]
    pub fn name_as_bytes(&self) -> &[u8] {
        self.file_name().to_bytes()
    }
    /// Lossily converts the original `Cstr` entry name to a `String`.
    /// If you need the original name, use `file_name()` instead.
    #[inline]
    pub fn name(&self) -> String {
        self.file_name().to_string_lossy().into_owned()
    }

    /**
    This relies on proc filesystem being available due to the use of
    `/proc/self/fd` to resolve the parent directory by file descriptor.

    The entry name is joined as raw bytes ([OsStr]), so non-UTF-8 file
    names resolve to their real paths instead of a lossy approximation.
    */
    pub fn path(&self) -> io::Result<PathBuf> {
        let link: PathBuf = proc_fd_path(self.dirfd.as_raw_fd())?;
        Ok(link.join(OsStr::from_bytes(self.name_as_bytes())))
    }

    /// Return the file type of the entry as a [nix::dir::Type] enum.
    pub fn file_type(&self) -> Option<Type> {
        if let Some(entry_type) = self.entry.file_type() {
            Some(entry_type)
        } else {
            match self.mode() {
                Some(mode) => EntryType(mode).entry_t(),
                None => None, // couldn't stat the entry
            }
        }
    }

    /// Return the file type of the entry as a `u8` typenum. Unknown is 254.
    pub fn typenum(&self) -> u8 {
        match self.file_type() {
            Some(t) => t as u8,
            None => 254,
        }
    }

    #[inline]
    pub fn is_file(&self) -> bool {
        matches!(self.file_type(), Some(Type::File))
    }
    #[inline]
    pub fn is_dir(&self) -> bool {
        matches!(self.file_type(), Some(Type::Directory))
    }
    #[inline]
    pub fn is_symlink(&self) -> bool {
        matches!(self.file_type(), Some(Type::Symlink))
    }
    #[inline]
    pub fn is_block(&self) -> bool {
        matches!(self.file_type(), Some(Type::BlockDevice))
    }
    #[inline]
    pub fn is_char(&self) -> bool {
        matches!(self.file_type(), Some(Type::CharacterDevice))
    }
    #[inline]
    pub fn is_sock(&self) -> bool {
        matches!(self.file_type(), Some(Type::Socket))
    }
    #[inline]
    pub fn is_fifo(&self) -> bool {
        matches!(self.file_type(), Some(Type::Fifo))
    }
}

/* --------------------------------- */

/**
NOTE: equality is defined over `(name, inode, parent dirfd)`. We must NOT
delegate to `nix::dir::Entry`'s derived `PartialEq`: nix fills the dirent
from `readdir_r` into a `MaybeUninit` buffer and only `d_reclen` bytes get
copied, while libc's derived comparison reads the full 256-byte `d_name`
array (plus `d_off`/`d_reclen`) — i.e. uninitialized garbage. The same
logical entry read twice could compare unequal.
*/
impl<'h> PartialEq for EntryExt<'h> {
    fn eq(&self, other: &Self) -> bool {
        self.name_as_bytes() == other.name_as_bytes()
            && self.ino() == other.ino()
            && self.dirfd.as_raw_fd() == other.dirfd.as_raw_fd()
    }
}

impl<'h> Ord for EntryExt<'h> {
    /**
    Primarily by name (which is unique within a directory); inode and
    parent dirfd act as tie-breakers so the total order is consistent
    with [PartialEq] — `cmp() == Equal` if and only if `eq()`.
    */
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.name_as_bytes()
            .cmp(other.name_as_bytes())
            .then_with(|| self.ino().cmp(&other.ino()))
            .then_with(|| self.dirfd.as_raw_fd().cmp(&other.dirfd.as_raw_fd()))
    }
}

impl<'h> PartialOrd for EntryExt<'h> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'h> Deref for EntryExt<'h> {
    type Target = Entry;

    fn deref(&self) -> &Self::Target {
        &self.entry
    }
}

/* --------------------------------- */

impl<'h> Hash for EntryExt<'h> {
    /**
    Hashes `(name, inode)` — a subset of the [PartialEq] fields, so the
    `Hash`/`Eq` contract holds. The dirfd is deliberately omitted: hashing
    it would invalidate hashes whenever the directory is reopened under a
    different fd. We must not delegate to `Entry`'s derived `Hash` either,
    since that reads uninitialized dirent tail bytes (see [PartialEq]).

    NOTE: this method will **not** produce stable hashes across processes
    due to the standard `hash()` implementation's SipHash algorithm.
    Use `xxh3()` instead.
    */
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name_as_bytes().hash(state);
        self.ino().hash(state);
    }
}

impl<'h> Xxh3Hashable for EntryExt<'h> {
    fn xxh3<H: Hasher>(&self, state: &mut H) {
        state.write(self.name_as_bytes());
        state.write_u64(self.entry.ino());
        state.write_u8(self.typenum());
    }

    fn xxh3_digest(&self) -> u64 {
        let mut hasher: CustomXxh3Hasher = CustomXxh3Hasher::default();
        hasher.write(self.name_as_bytes());
        hasher.write_u64(self.entry.ino());
        hasher.write_u8(self.typenum());
        hasher.finish()
    }
}

/* ######################################################################### */

/**
The type of change detected in a directory, if any.

`DirNum` or `FileNum` change also implies a change of `DirHash` or `FileHash`,
respectively, but the reverse is not true. But since we first check for
`DirNum` and `FileNum` changes, that doesn't matter.
*/
#[derive(Debug)]
pub enum StateChange {
    Unchanged,
    /// Directory count change: positive = directories added since the previous
    /// snapshot, negative = removed.
    DirNum(i64),
    /// File count change: positive = files added since the previous snapshot,
    /// negative = removed.
    FileNum(i64),
    /// Same directory count, but hash of dir entries has changed
    DirHash,
    /// Same file count, but hash of file entries has changed
    FileHash,
}

impl StateChange {
    pub fn is_same(&self) -> bool {
        match self {
            StateChange::Unchanged => true,
            _ => false,
        }
    }
}

/* --------------------------------- */

/// This struct holds the state of a directory for change detection.
#[derive(Clone)]
pub struct DirectoryState {
    dirs: usize,
    files: usize,
    hash_d: u64,
    hash_f: u64,
    when: Option<TimeSinceEpoch>,
}

impl DirectoryState {
    /// Update the state object with new values if they differ.
    fn update(&mut self, state: DirectoryState) {
        match self == &state {
            true => self.when = state.when,
            false => *self = state,
        }
    }

    /**
    Compare two states and return the type of change as a [StateChange] enum.
    `self` is the previous snapshot; `other` is the newer one. The first
    change seen is returned, so `DirNum` or `FileNum` change implies
    a `DirHash` or `FileHash` change, respectively.

    Count deltas are signed as `other - self`, so a positive value means
    entries were added since `self` and a negative value means removed.
    */
    pub fn change(&self, other: &Self) -> StateChange {
        if self.dirs != other.dirs {
            return StateChange::DirNum(other.dirs as i64 - self.dirs as i64);
        }
        if self.files != other.files {
            return StateChange::FileNum(other.files as i64 - self.files as i64);
        }
        if self.hash_d != other.hash_d {
            return StateChange::DirHash;
        }
        if self.hash_f != other.hash_f {
            return StateChange::FileHash;
        }
        StateChange::Unchanged
    }

    /// Combined hash of directory and file entry hashes.
    pub fn hash_all(&self) -> u64 {
        self.hash_d.rotate_left(32) ^ self.hash_f
    }
}

impl Default for DirectoryState {
    fn default() -> Self {
        Self {
            dirs: 0,
            files: 0,
            hash_d: 0,
            hash_f: 0,
            when: None,
        }
    }
}

impl Eq for DirectoryState {}

impl PartialEq for DirectoryState {
    fn eq(&self, other: &Self) -> bool {
        self.dirs == other.dirs
            && self.files == other.files
            && self.hash_d == other.hash_d
            && self.hash_f == other.hash_f
    }
}

impl Hash for DirectoryState {
    /**
    `when` is excluded to uphold the `Hash`/`Eq` contract: [PartialEq]
    above compares only the content fields, so equal states must produce
    equal hashes regardless of when their snapshots were taken.
    */
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.dirs.hash(state);
        self.files.hash(state);
        self.hash_d.hash(state);
        self.hash_f.hash(state);
    }
}

impl Debug for DirectoryState {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "DirectoryState {{ dirs: {d}, files: {f}, hash_d: 0x{hd:x}, hash_f: 0x{hf:x}, when: {w} }}",
            d = self.dirs,
            f = self.files,
            hd = self.hash_d,
            hf = self.hash_f,
            w = match &self.when {
                Some(t) => t.to_debug(),
                None => "<uninit>".to_string(),
            },
        )
    }
}

impl Display for DirectoryState {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "DirectoryState {{ dirs: {d}, files: {f} ({w}) }}",
            d = self.dirs,
            f = self.files,
            w = match &self.when {
                Some(t) => t.to_display(),
                None => "<uninit>".to_string(),
            },
        )
    }
}

/* ######################################################################### */

// Convenience type alias.
type EntryVec<'h> = EnhVec<EntryExt<'h>>;

/**
An open handle to a directory, internally a [nix::dir::Dir] object. The aim
is to be somewhat compatible with the [std::fs::ReadDir] API, with the
following notable differences / enhancements:

- `iter_sorted()` sorts the returned entries alphabetically
- `iter()` rewinds after finishing, so it can be called multiple times
- `path()` returns the canonicalized path of the directory
  (relies on procfs being available and the inner Dir being open)
- change detection in the directory (or parts therein) is facilitated by:
  - numbers of directories and files are cached
  - (stable) hashes of directory and file entries are cached

NOTE: the counts and hashes are updated when the directory is iterated for
the first time, or when asked for explicitly, but they are **not** updated
automatically if the directory is changed externally.

NOTE: the [DirHandle] object is not thread-safe and it is not intended
to be shared between threads, see the following `readdir` manual:
https://www.gnu.org/software/libc/manual/html_node/Reading_002fClosing-Directory.html

Future versions of POSIX are likely to obsolete `readdir_r` and specify that it's
unsafe to call `readdir` simultaneously from multiple threads.
*/
#[derive(Debug, Eq)]
pub struct DirHandle {
    inner: Dir,
    state: DirectoryState,
}

impl DirHandle {
    /// Open a directory by path and return its handle object.
    #[instrument(level = "trace")]
    pub fn new(path: &Path) -> io::Result<Self> {
        Ok(Self {
            inner: get_dir_handle(path)?,
            state: DirectoryState::default(),
        })
    }

    /// Our file descriptor as a [DirFd] object.
    pub fn fd(&self) -> DirFd {
        self.inner.as_raw_fd().into()
    }

    /**
    This relies on proc filesystem being available due to the use of
    `/proc/self/fd` to resolve the path by file descriptor.
    */
    pub fn path(&self) -> io::Result<PathBuf> {
        Ok(self.fd().path()?)
    }

    /// Stored state of the directory as a [DirectoryState] object.
    pub fn state(&self) -> &DirectoryState {
        &self.state
    }

    /**
    Current state of the directory as a [DirectoryState] object.

    Does **not** touch the stored state — use `state_changed()` for that.
    Returns an error if a `readdir` failure cuts the listing short, since
    a partial listing must not masquerade as the directory's state.
    */
    pub fn state_current(&mut self) -> io::Result<DirectoryState> {
        directory_state(self)
    }

    /**
    Whether the state has changed. Also updates the stored state if so.

    The first call (or the first after an error) establishes the baseline
    and returns `false`. On a `readdir` error the stored state is left
    untouched, so a failed pass cannot corrupt the baseline.
    */
    pub fn state_changed(&mut self) -> io::Result<bool> {
        let first: bool = self.state.when.is_none();
        let current: DirectoryState = self.state_current()?;
        let changed: StateChange = self.state.change(&current);
        self.state.update(current);
        Ok(!first && !changed.is_same())
    }

    /**
    Return the inner [nix::dir::Iter] object and make it `Peekable`.

    NOTE: this iterator will return the special `.` and `..` entries.

    ## Safety
    The returned `Iter` is **not** thread-safe and must **not** be sent to
    another thread. It must be used only in the thread that created it.
    */
    pub unsafe fn raw_iter<'handle>(&'handle mut self) -> Peekable<Iter<'handle>> {
        self.inner.iter().peekable()
    }

    /**
    Run a closure on each [nix::dir::Entry]. This allows lower-level but
    safe access to the inner [nix::dir::Iter] iterator.

    NOTE: the special `.` and `..` entries will be processed as well.

    NOTE: iteration stops at the first `readdir` error — skipping errors
    would loop forever on a persistently failing stream (e.g. `ESTALE`).
    */
    pub fn for_each<F>(&mut self, mut f: F)
    where
        F: FnMut(&Entry),
    {
        self.inner
            .iter()
            .map_while(Result::ok)
            .for_each(|entry: Entry| {
                f(&entry);
            });
    }

    /**
    This iterator does the following:
    - skips the special `.` and `..` entries
    - rewinds after finishing
    - maintains a lookahead buffer to preferentially return directory
      entries before other entries (NOTE: not a guarantee)
    - yields entries even when their file type cannot be determined
      (`file_type()` returns `None`); the heuristic treats them as files
    - stops at the first `readdir` error (see `DirHandleIter::error()`)
    */
    pub fn iter(&'_ mut self) -> DirHandleIter<'_> {
        DirHandleIter::new(self, false)
    }

    /// Same as `iter()`, but also explicitly `stat()`s each entry.
    pub fn iter_stat(&'_ mut self) -> DirHandleIter<'_> {
        DirHandleIter::new(self, true)
    }

    /**
    Return the directory entries as a tuple of directories and files.
    The booleans specify whether to include directories and/or files.
    Entries whose type cannot be determined count as files (non-dirs).
    */
    pub fn entries<'handle>(
        &'handle mut self,
        dirs: bool,
        files: bool,
    ) -> (EntryVec<'handle>, EntryVec<'handle>) {
        let mut d_vec: EntryVec<'handle> = EnhVec::new();
        let mut f_vec: EntryVec<'handle> = EnhVec::new();
        self.iter().for_each(|entry: EntryExt<'handle>| {
            match entry.file_type() {
                Some(Type::Directory) => {
                    if dirs {
                        d_vec.push(entry)
                    }
                }
                _ => {
                    if files {
                        f_vec.push(entry)
                    }
                }
            }
        });
        (d_vec, f_vec)
    }

    /// Return the directory entries as sorted tuples of directories and files.
    pub fn entries_sorted<'handle>(&'handle mut self) -> (EntryVec<'handle>, EntryVec<'handle>) {
        let (mut dirs, mut files) = self.entries(true, true);
        dirs.sort(Sorting::Ascending);
        files.sort(Sorting::Ascending);
        (dirs, files)
    }

    /**
    This iterator does the following:
    - skips the special `.` and `..` entries
    - returns directory entries before all other entries
    - sorts both entry lists alphabetically (separately)
    */
    pub fn iter_sorted(&'_ mut self) -> DirHandleIterSorted<'_> {
        let (mut dirs, mut files) = self.entries(true, true);

        // NOTE: we sort in reverse order to pop the entries in alphabetical order
        dirs.sort(Sorting::Descending);
        files.sort(Sorting::Descending);
        files.extend(dirs);
        DirHandleIterSorted(files, PhantomData)
    }
}

/* --------------------------------- */

impl Hash for DirHandle {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

impl PartialEq for DirHandle {
    fn eq(&self, other: &Self) -> bool {
        // we only need to compare the inner [nix::dir::Dir] objects
        self.inner == other.inner
    }
}

impl AsRawFd for DirHandle {
    /**
    The file descriptor continues to be owned by the [DirHandle], so
    callers must not keep a [RawFd] after the `DirHandle` is dropped.
    */
    fn as_raw_fd(&self) -> RawFd {
        self.inner.as_raw_fd()
    }
}

/*
DirHandle is automatically Send (nix::dir::Dir is explicitly Send, the
rest of the fields are plain data), so no `unsafe impl` is needed — and
having one would silently mask a future non-Send field. This assertion
keeps the requirement checked at compile time: [OpenHandles] shares
handles across threads and needs `DirHandle: Send`.
*/
const _: () = {
    const fn assert_send<T: Send>() {}
    assert_send::<DirHandle>();
};

#[cfg(feature = "size_of")]
const DHSIZE: usize = 296;

#[cfg(feature = "size_of")]
impl SizeOf for DirHandle {
    fn size_of_children(&self, context: &mut Context) {
        // nix::dir::Dir:
        // - ptr::NonNull - 8 bytes
        // - libc::DIR - 8? bytes
        // - libc::dirent - 280 bytes
        // Total: 296 + 8 (padding?) = 304 bytes
        context.add(DHSIZE + 8).add_distinct_allocation();
    }
}

/* ######################################################################### */

/**
A wrapper around [std::collections::VecDeque] that provides additional
methods for handling different types, in this case [[EntryExt]] structs.
*/
#[derive(Debug)]
struct BufDeque<T> {
    q: VecDeque<T>,
    /**
    Number of directory entries currently buffered. Lets `try_pop_dir`
    skip the linear scan in the (common) all-files case. The counter
    stays in sync because `is_dir()` is stable per entry — `d_type` is
    fixed and the stat fallback result is cached — and all mutation
    goes through `push` / `try_pop_dir` (no `DerefMut` escape hatch).
    */
    n_dirs: usize,
}

impl<'h> BufDeque<EntryExt<'h>> {
    pub fn new(capacity: usize) -> Self {
        Self {
            q: VecDeque::with_capacity(capacity),
            n_dirs: 0,
        }
    }

    /// Push an entry to the buffer. Directories to the front, rest to back.
    pub fn push(&mut self, entry: EntryExt<'h>) {
        if entry.is_dir() {
            self.q.push_front(entry);
            self.n_dirs += 1;
        } else {
            self.q.push_back(entry);
        }
    }

    /// Any directory entries in the buffer?
    #[expect(dead_code)]
    pub fn has_dir(&self) -> bool {
        self.n_dirs > 0
    }

    /**
    Try to pop a directory entry. The first directory found is chosen,
    but if none are in the buffer, we pop the oldest (front) entry.
    */
    pub fn try_pop_dir(&mut self) -> Option<EntryExt<'h>> {
        if self.n_dirs == 0 {
            return self.q.pop_front();
        }
        match self.q.iter().position(|entry: &EntryExt<'h>| entry.is_dir()) {
            // directories are pushed to the front, so idx is almost always 0
            Some(idx) => {
                self.n_dirs -= 1;
                self.q.remove(idx)
            }
            None => {
                // fail-safe: counter desynced (should not be possible)
                self.n_dirs = 0;
                self.q.pop_front()
            }
        }
    }
}

/* --------------------------------- */

impl<'h> Default for BufDeque<EntryExt<'h>> {
    fn default() -> Self {
        Self::new(LOOKAHEAD_BUFFER_SIZE)
    }
}

impl<T> Deref for BufDeque<T> {
    type Target = VecDeque<T>;

    fn deref(&self) -> &Self::Target {
        &self.q
    }
}

/* ######################################################################### */

/// The return type of [DirHandle::iter]
#[derive(Debug)]
pub struct DirHandleIter<'handle> {
    inner: Peekable<Iter<'handle>>,
    dirfd: BorrowedFd<'handle>,
    buf: BufDeque<EntryExt<'handle>>,
    state: &'handle mut DirectoryState,
    stat: bool,
    /// shall we finalize the [DirectoryState] when the pass completes?
    update: bool,
    /// per-entry xxh3 digests of dir entries - used for state hashing
    dirs: Vec<u64>,
    /// per-entry xxh3 digests of file entries - used for state hashing
    files: Vec<u64>,
}

impl<'handle> DirHandleIter<'handle> {
    pub fn new(handle: &'handle mut DirHandle, stat: bool) -> Self {
        // lazy one-shot state population: only the first complete pass
        // over a handle computes the DirectoryState
        let update: bool = handle.state.when.is_none();
        Self::with_update(handle, stat, update)
    }

    /// Like `new()`, but with explicit control over whether this pass
    /// finalizes the [DirectoryState] of the parent handle.
    fn with_update(handle: &'handle mut DirHandle, stat: bool, update: bool) -> Self {
        let raw_fd: RawFd = handle.inner.as_raw_fd();
        /*
        SAFETY: the `&'handle mut DirHandle` borrow keeps `handle.inner`
        (a `nix::dir::Dir`) alive for `'handle`. The Dir owns the underlying
        fd and only closes it on drop, so the fd is valid for at least
        `'handle`. The `state: &'handle mut ...` field below extends the
        mut borrow for the whole iterator lifetime, which transitively
        keeps the Dir alive.
        */
        let dirfd: BorrowedFd<'handle> = unsafe { BorrowedFd::borrow_raw(raw_fd) };
        Self {
            dirfd,
            inner: handle.inner.iter().peekable(),
            buf: BufDeque::default(),
            state: &mut handle.state,
            stat,
            update,
            dirs: Vec::new(),
            files: Vec::new(),
        }
    }

    /// Is the inner iterator done? A sticky `readdir` error (see [next])
    /// also ends the iteration.
    #[inline]
    fn done(&mut self) -> bool {
        matches!(self.inner.peek(), None | Some(Err(_)))
    }

    /// The `readdir` error that ended this pass early, if any.
    pub fn error(&mut self) -> Option<Errno> {
        match self.inner.peek() {
            Some(Err(e)) => Some(*e),
            _ => None,
        }
    }

    /// Did the inner iterator stop early due to a `readdir` error?
    #[inline]
    fn errored(&mut self) -> bool {
        self.error().is_some()
    }

    /**
    Peek at the next entry in the inner iterator to try to check if it's
    a directory. This is not a guarantee, as the [nix::dir::Entry] API
    cannot guarantee that the file type is available. This is due to the
    type not always being known ([libc::dirent::d_type] may be `DT_UNKNOWN`).
    */
    fn is_next_dir(&mut self) -> Option<bool> {
        self.inner.peek().map_or(Some(false), |res| {
            res.map_or(Some(false), |e: Entry| {
                // `.` and `..` are both directories, but `get_one()` filters
                // them out — treating them as "next dir" here would wrongly
                // defer the current entry only to have the dot skipped.
                if matches!(e.file_name().to_bytes(), DOT1 | DOT2) {
                    return Some(false);
                }
                e.file_type()
                    .map_or(None, |t: Type| Some(t == Type::Directory))
            })
        })
    }

    /// Get one entry from the inner iterator.
    fn get_one(&mut self) -> Option<EntryExt<'handle>> {
        let entry: EntryExt<'handle> = next(&mut self.inner, self.dirfd, self.stat)?;
        if self.update {
            // store the entry's stable digest for state hashing — 8 bytes
            // per entry instead of cloning the whole EntryExt, which kept
            // the full listing in memory until the pass completed
            let digest: u64 = entry.xxh3_digest();

            #[cfg(debug_assertions)]
            debug!(target: "get_one", "{:?} : ino {} : xxh3_digest: 0x{digest:x}",
                entry.name(), entry.ino());

            match entry.file_type() {
                Some(Type::Directory) => self.dirs.push(digest),
                // files, special files and unknown types all count as files
                _ => self.files.push(digest),
            }
        }
        Some(entry)
    }

    /// Fill the lookahead buffer with entries.
    #[instrument(level = "trace", skip_all)]
    fn fill_buffer(&mut self) {
        while self.buf.len() < LOOKAHEAD_BUFFER_SIZE {
            if let Some(entry) = self.get_one() {
                self.buf.push(entry);
            } else {
                trace!(target: "iter_empty", "{:?}", self.inner);
                break;
            }
        }
    }
}

impl<'handle> Iterator for DirHandleIter<'handle> {
    type Item = EntryExt<'handle>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(entry) = self.buf.try_pop_dir() {
                if entry.is_dir() {
                    return Some(entry);
                } else {
                    match self.done() {
                        true => return Some(entry),
                        false => {
                            // give it another chance to return a directory entry,
                            // since all we have in the buffer are not directories
                            if let Some(is_dir) = self.is_next_dir() {
                                // entry type is known in `dirent.d_type`
                                if is_dir {
                                    self.buf.push(entry);
                                    return self.get_one();
                                } else {
                                    return Some(entry);
                                }
                            } else {
                                // we must get the next entry to determine its type
                                match self.get_one() {
                                    Some(extra) => {
                                        trace!(target: "try_get_extra_dir", "{:?} : {extra:?}", extra.name());
                                        if extra.is_dir() {
                                            // push the current entry back to the buffer...
                                            self.buf.push(entry);
                                            // ... and return the directory entry
                                            return Some(extra);
                                        } else {
                                            self.buf.push(extra);
                                            return Some(entry);
                                        }
                                    }
                                    None => {
                                        // inner iterator exhausted; the current
                                        // entry is the last one to yield.
                                        return Some(entry);
                                    }
                                }
                            }
                        }
                    }
                }
            } else if self.done() {
                debug!(target: "DirHandleIter::next", "iter_done: {:?}", self.inner);
                if self.errored() {
                    /*
                    the listing is incomplete, so we must not finalize the
                    DirectoryState from partial data; `when` stays as-is and
                    a later full pass will compute the state instead.
                    */
                    warn!(target: "DirHandleIter::next",
                        "readdir error ended iteration early (listing incomplete): {:?}",
                        self.inner.peek());
                    return None;
                }
                if self.update {
                    self.update = false;
                    self.state.dirs = self.dirs.len();
                    self.state.files = self.files.len();
                    self.state.hash_d = digest_of_digests(std::mem::take(&mut self.dirs));
                    self.state.hash_f = digest_of_digests(std::mem::take(&mut self.files));
                    self.state.when = TimeSinceEpoch::new().into();
                    trace!(target: "DirHandle.state", "{:?}", self.state);
                }
                return None;
            } else if self.buf.is_empty() {
                self.fill_buffer();
            }
        }
    }
}

/* ######################################################################### */

/**
A sorted iterator over the entries in a directory.

The lifetime parameter `'handle` is used to annotate that the entries
in the Vec are only valid while the parent [DirHandle] object exists.
*/
#[derive(Debug)]
pub struct DirHandleIterSorted<'handle>(EntryVec<'handle>, PhantomData<&'handle DirHandle>);

impl<'handle> Iterator for DirHandleIterSorted<'handle> {
    type Item = EntryExt<'handle>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.pop()
    }
}

/* ######################################################################### */

/**
A container for open directory handles ([DirHandle]s).

Thread-safe due to the inner [DashMap] being thread-safe.

**NOTE**: trying to check out more than 1 handle at a time from the same thread
(aka. holding more than one reference into `OpenHandles`) may lead to a deadlock
due to the internal locking of `DashMap`. You have been warned.
*/
#[derive(Default, Debug)]
pub struct OpenHandles(DashMap<RawFd, DirHandle>);

impl OpenHandles {
    pub fn new() -> Self {
        Self(DashMap::new())
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether we have such file descriptor.
    pub fn contains(&self, fd: RawFd) -> bool {
        self.0.contains_key(&fd)
    }
    /// Whether we have such handle. Slower than `contains(fd)`.
    pub fn contains_handle(&self, handle: &DirHandle) -> bool {
        self.0.iter().any(|entry| entry.value() == handle)
    }

    /**
    Check out a handle from the map. Only allows one borrow at a time due
    to internally using [DashMap::get_mut], which returns a [RefMut].
    This is also thread-safe as DashMap (RefMut) is thread-safe due to its
    internal locking using [parking_lot::RwLock].
    */
    fn checkout<'a>(&'a self, fd: RawFd) -> Option<CheckedOutHandle<'a>> {
        let ref_handle: RefMut<'a, RawFd, DirHandle> = self.0.get_mut(&fd)?;
        Some(CheckedOutHandle {
            inner: ref_handle,
            close_callback: Rc::new(|fd: i32| {
                self.close(fd);
            }),
        })
    }

    /**
    Open a directory, insert its handle into the map and return it.

    **NOTE**: may deadlock if called while holding any kind of reference
    into this [OpenHandles] in the same thread.
    */
    pub fn open(&'_ self, path: &Path) -> io::Result<CheckedOutHandle<'_>> {
        let handle: DirHandle = DirHandle::new(path)?;
        let fd: RawFd = handle.as_raw_fd();
        self.0.insert(fd, handle);
        /*
        Between the insert above and the checkout below, another thread on
        this same OpenHandles could call close(fd) and evict our just-opened
        handle. The fd value is freshly allocated by the kernel and the
        race is exceedingly rare, but turning it into io::Error is cheap and
        strictly better than panicking inside a library entry point.
        */
        self.checkout(fd).ok_or_else(|| {
            io::Error::other("handle closed concurrently between insert and checkout")
        })
    }

    /// Insert a handle into the map. Replaces an existing handle with the same
    /// file descriptor, if present.
    pub fn insert(&self, handle: DirHandle) {
        self.0.insert(handle.as_raw_fd(), handle);
    }

    /**
    Get a [DirHandle] if we have it.

    **NOTE**: may deadlock if called while holding any kind of reference
    into this [OpenHandles] in the same thread.
    */
    pub fn get(&'_ self, fd: RawFd) -> Option<CheckedOutHandle<'_>> {
        self.checkout(fd)
    }

    /// Remove a handle from the map if there are no other strong refs to it.
    pub fn remove(&self, fd: RawFd) -> Option<DirHandle> {
        self.0.remove(&fd).map(|(_, handle)| Some(handle))?
    }

    /// Close an open directory handle and release its file descriptor.
    pub fn close(&self, fd: RawFd) {
        self.0.remove(&fd);
    }

    /// Close all open directory handles (and release their file descriptors).
    pub fn close_all(&self) {
        self.0.clear();
    }

    /**
    Run a closure on each [DirHandle] in random order.

    **NOTE**: may deadlock if called while holding a mutable reference
    into this [OpenHandles] in the same thread.
    */
    pub fn for_each<F>(&self, mut f: F)
    where
        F: FnMut(&DirHandle),
    {
        self.0.iter().for_each(|item| {
            f(item.value());
        });
    }

    /**
    Run a mutating closure on each [DirHandle] in random order.

    **NOTE**: may deadlock if called while holding any kind of reference
    into this [OpenHandles] in the same thread.
    */
    pub fn for_each_mut<F>(&self, mut f: F)
    where
        F: FnMut(&mut DirHandle),
    {
        self.0.iter_mut().for_each(|mut item| {
            f(item.value_mut());
        });
    }
}

/*
SAFETY: DashMap<RawFd, DirHandle> is not auto-Sync because DirHandle is
!Sync (nix::dir::Dir is deliberately !Sync — readdir on a shared DIR*
races). Sharing &OpenHandles is still sound because DashMap's per-shard
RwLock makes &mut DirHandle access (checkout / for_each_mut) exclusive,
and concurrent shared access (for_each / iter) only reaches `&self`
methods of DirHandle — fd(), path(), state(), as_raw_fd(), Hash, Eq —
none of which touch the underlying DIR* stream state.

INVARIANT: keep it that way. Any future `&self` method on DirHandle
that reads or moves the DIR* position (readdir/telldir/seekdir/...)
silently breaks this impl and must take `&mut self` instead.
*/
unsafe impl Sync for OpenHandles {}

#[cfg(feature = "size_of")]
impl SizeOf for OpenHandles {
    fn size_of_children(&self, context: &mut Context) {
        if self.0.capacity() > 0 {
            let used: usize = (DHSIZE + 4) * self.0.len();
            let total: usize = (DHSIZE + 4) * self.0.capacity();
            context
                .add(used)
                .add_excess(total - used)
                .add_distinct_allocation();

            self.0.iter().for_each(|itm| {
                itm.key().size_of_children(context);
                itm.value().size_of_children(context);
            });
        }

        self.0.hasher().size_of_children(context);
    }
}

/**
An exclusively locked [DirHandle] from the [OpenHandles] container.

`CheckedOutHandle` implements [Deref] and [DerefMut] so you can use it
as a `DirHandle` directly. In addition, it has a `close()` method which
removes the `DirHandle` from parent `OpenHandles` (hence the directory
handle is closed and its file descriptor released when dropped).
*/
pub struct CheckedOutHandle<'a> {
    inner: RefMut<'a, RawFd, DirHandle>,
    /*
    Using `Rc` instead of `Arc` here is deliberate because we don't want to
    share the callback between threads and this also disallows moving a
    checked-out handle to another thread.
    */
    close_callback: Rc<dyn Fn(RawFd) + 'a>,
}

impl<'a> CheckedOutHandle<'a> {
    /**
    Close this [DirHandle] and release its file descriptor.

    NOTE: the internal lock must be released before removing the entry
    (see the deadlock caveat on [OpenHandles]), which opens a tiny
    window where another thread may close this fd and the kernel may
    recycle the number for a freshly opened handle — in that case the
    new entry gets evicted instead. Same fd-reuse caveat as documented
    on `OpenHandles::open`.
    */
    pub fn close(self) {
        let fd: i32 = *self.inner.key();
        drop(self.inner); // explicitly drop the RefMut
        (self.close_callback)(fd);
    }
}

impl<'a> Deref for CheckedOutHandle<'a> {
    type Target = DirHandle;

    fn deref(&self) -> &Self::Target {
        self.inner.value()
    }
}

impl<'a> DerefMut for CheckedOutHandle<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.value_mut()
    }
}

impl Debug for CheckedOutHandle<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "CheckedOutHandle({:?})", self.inner)
    }
}

/* ########################### UTILITY FUNCTIONS ########################### */

/**
Resolve an open file descriptor to its path via `/proc/self/fd/<fd>`.

NOTE: `/proc/self/fd` is itself a directory (its *entries* are symlinks),
so probing procfs availability must not `readlink()` the directory — that
fails with EINVAL even when procfs is mounted. We attempt the per-fd
resolution directly and only diagnose a missing procfs after a failure.
*/
fn proc_fd_path(fd: RawFd) -> io::Result<PathBuf> {
    read_link(format!("{}/{}", PROC_FD_PATH, fd)).map_err(|e| {
        if e.kind() == io::ErrorKind::NotFound && !Path::new(PROC_FD_PATH).is_dir() {
            io::Error::new(io::ErrorKind::Unsupported, "procfs not available")
        } else {
            e
        }
    })
}

/// Open a directory and return its handle.
fn get_dir_handle(path: &Path) -> io::Result<Dir> {
    /*
    - O_DIRECTORY: fail with ENOTDIR up front instead of e.g. blocking
      forever on a FIFO before `fdopendir` gets a chance to reject it.
    - O_CLOEXEC: don't leak directory fds to exec'd children.
    - O_NONBLOCK: belt-and-suspenders against blocking opens (matches
      what std::fs::ReadDir passes to open(2)).
    */
    let flags: OFlag =
        OFlag::O_RDONLY | OFlag::O_DIRECTORY | OFlag::O_CLOEXEC | OFlag::O_NONBLOCK;
    Ok(Dir::open(path, flags, Mode::empty())?)
}

/// Open a file and return its handle.
#[expect(dead_code)]
fn get_file_handle(path: &Path) -> io::Result<File> {
    Ok(OpenOptions::new().read(true).open(path)?)
}

/**
Combine per-entry xxh3 digests into a single stable digest. Sorting the
digests makes the result independent of `readdir` order; the per-entry
digests already cover `(name, inode, typenum)`, so the same set of
entries always produces the same combined digest.

This is the single source of truth for [DirectoryState] hashing — both
the lazy in-iterator computation and `directory_state()` go through it,
which keeps the two paths comparable.
*/
fn digest_of_digests(mut digests: Vec<u64>) -> u64 {
    digests.sort_unstable();
    let mut xxh: CustomXxh3Hasher = CustomXxh3Hasher::default();
    digests.iter().for_each(|d: &u64| xxh.write_u64(*d));
    xxh.finish()
}

/**
Return the state of a directory as a [DirectoryState] object.

This pass never finalizes the handle's stored state (that side effect
belongs to `DirHandle::state_changed`), and only accumulates 8 bytes per
entry instead of materializing the listing. A `readdir` error ends the
listing early, in which case we return the error instead of a partial
(and therefore wrong) state.
*/
#[instrument(level = "trace", skip_all, ret)]
fn directory_state(dir: &mut DirHandle) -> io::Result<DirectoryState> {
    let mut d_digests: Vec<u64> = Vec::new();
    let mut f_digests: Vec<u64> = Vec::new();
    let mut iter: DirHandleIter = DirHandleIter::with_update(dir, false, false);
    for entry in iter.by_ref() {
        match entry.file_type() {
            Some(Type::Directory) => d_digests.push(entry.xxh3_digest()),
            _ => f_digests.push(entry.xxh3_digest()),
        }
    }
    if let Some(errno) = iter.error() {
        return Err(io::Error::from_raw_os_error(errno as i32));
    }
    Ok(DirectoryState {
        dirs: d_digests.len(),
        files: f_digests.len(),
        hash_d: digest_of_digests(d_digests),
        hash_f: digest_of_digests(f_digests),
        when: TimeSinceEpoch::new().into(),
    })
}

/**
Return the next entry from the inner [nix::dir::Iter] as an [EntryExt],
skipping `.` and `..`.

Entries whose file type cannot be determined (`d_type` is `DT_UNKNOWN`
and the `fstatat` fallback fails, e.g. due to permission denied) are
yielded too — their `file_type()` returns `None` and the caller decides
what to do. Silently dropping them would make a listable-but-unsearchable
directory iterate as empty on filesystems that don't populate `d_type`.

A `readdir` error ends the iteration: a persistent error (e.g. `ESTALE`
on NFS, `EIO`) would otherwise be skipped forever and spin this loop. The
error is deliberately left **unconsumed** in the [Peekable] slot, so it is
sticky — callers can observe it via `peek()` and repeated calls return
`None` without re-issuing failing `readdir` syscalls.

If `stat == true`, we also `stat()` the entry before returning it.
*/
#[instrument(level = "trace", skip(it))]
fn next<'h>(
    it: &mut Peekable<Iter<'h>>,
    dirfd: BorrowedFd<'h>,
    stat: bool,
) -> Option<EntryExt<'h>> {
    loop {
        match it.peek()? {
            Ok(_) => {}
            Err(e) => {
                trace!(target: "readdir_err", "readdir failed, ending iteration: {e}");
                return None;
            }
        }
        let entry: Entry = it.next()?.expect("peeked entry was Ok");

        // Sadly it appears that we cannot rely on the special "." and ".."
        // entries being returned first by the libc `readdir` call, so to
        // filter them out we must match each name.
        if matches!(entry.file_name().to_bytes(), DOT1 | DOT2) {
            continue;
        }

        // convert the [nix::dir::Entry] to our `EntryExt`
        let entry: EntryExt<'h> = match stat {
            false => EntryExt::new(entry, dirfd),
            true => EntryExt::new_statted(entry, dirfd),
        };
        trace!(target: "name", "{:?} : {:?}", entry.name(), entry);
        return Some(entry);
    }
}
