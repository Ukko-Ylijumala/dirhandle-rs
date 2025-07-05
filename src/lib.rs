// Copyright (c) 2024-2025 Mikko Tanner. All rights reserved.

#![allow(dead_code)]

use custom_xxh3::{CustomXxh3Hasher, Xxh3Hashable};
use dashmap::{mapref::one::RefMut, DashMap};
use enhvec::{EnhVec, Sorting};
use libc;
use miniutils::{ToDebug, ToDisplay};
use nix::{
    dir::{Dir, Entry, Iter, Type},
    fcntl::{openat2, AtFlags, OFlag, OpenHow, ResolveFlag},
    sys::stat::{fstatat, Mode},
};
use std::{
    cmp::{Eq, Ord, Ordering, PartialEq, PartialOrd},
    collections::VecDeque,
    fmt::{self, Debug, Display, Formatter},
    fs::{read_link, File, OpenOptions},
    hash::{Hash, Hasher},
    io,
    iter::Peekable,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, RawFd},
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

impl EntryType {
    #[inline]
    pub fn is_dir(&self) -> bool {
        self.is(libc::S_IFDIR)
    }
    #[inline]
    pub fn is_file(&self) -> bool {
        self.is(libc::S_IFREG)
    }
    pub fn is_symlink(&self) -> bool {
        self.is(libc::S_IFLNK)
    }
    pub fn is_block(&self) -> bool {
        self.is(libc::S_IFBLK)
    }
    pub fn is_char(&self) -> bool {
        self.is(libc::S_IFCHR)
    }
    pub fn is_sock(&self) -> bool {
        self.is(libc::S_IFSOCK)
    }
    pub fn is_fifo(&self) -> bool {
        self.is(libc::S_IFIFO)
    }

    #[inline]
    fn is(&self, mode: libc::mode_t) -> bool {
        self.masked() == mode
    }
    #[inline]
    fn masked(&self) -> libc::mode_t {
        self.0 & libc::S_IFMT
    }

    /// Return the file type of the entry as a [nix::dir::Type] enum.
    pub fn entry_t(&self) -> Option<Type> {
        match self.masked() {
            libc::S_IFIFO => Some(Type::Fifo),
            libc::S_IFCHR => Some(Type::CharacterDevice),
            libc::S_IFDIR => Some(Type::Directory),
            libc::S_IFBLK => Some(Type::BlockDevice),
            libc::S_IFREG => Some(Type::File),
            libc::S_IFLNK => Some(Type::Symlink),
            libc::S_IFSOCK => Some(Type::Socket),
            /* libc::DT_UNKNOWN | */ _ => None,
        }
    }
}

/**
A thin wrapper around a [[RawFd]] with some extra functionality.

Notably, a [DirFd] can resolve its path from the file descriptor, and a
negative value indicates that the file descriptor used to be open.

Due to internally using an [AtomicI32], it is thread-safe.

Mapping:
- `DirFd > 0` : This is an open handle with this file descriptor.
- `DirFd == 0` : Uninitialized.
- `DirFd < 0` : We had an open handle as `abs(DirFd)`, but it was closed.
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

    /// Whether the file descriptor is open.
    pub fn is_open(&self) -> bool {
        self.fd() > 0
    }

    /**
    Set the inner file descriptor.

    - Returns the file descriptor if it was set successfully.
    - If the fd is already set, returns an error with the existing fd.

    In the latter case, the caller should clear the existing fd first.
    */
    pub fn set(&self, fd: RawFd) -> Result<RawFd, RawFd> {
        let current: i32 = self.fd();
        if current > 0 {
            return Err(current);
        }
        self.0.store(fd, Relaxed);
        Ok(fd)
    }

    /**
    Clear the file descriptor.

    If the fd is set, we "store" the negative value of the fd.
    If the fd is already cleared (negative), we set it to 0.
    */
    pub fn clear(&self) {
        let current: i32 = self.fd();
        if current > 0 {
            self.0.store(-current, Relaxed);
        } else {
            self.0.store(0, Relaxed);
        }
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
        if read_link(PROC_FD_PATH).is_err() {
            return Err(io::Error::new(io::ErrorKind::Unsupported, "procfs not available"));
        }
        if self.fd() == 0 {
            return Err(io::Error::new(io::ErrorKind::NotFound, "no file descriptor"));
        }
        if self.fd() < 0 {
            return Err(io::Error::new(io::ErrorKind::NotFound, "stale file descriptor"));
        }
        read_link(format!("{}/{}", PROC_FD_PATH, self.fd()))
    }
}

impl Clone for DirFd {
    fn clone(&self) -> Self {
        DirFd(self.fd().into())
    }
}

impl Default for DirFd {
    fn default() -> Self {
        DirFd(0.into())
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
    fn as_fd(&self) -> BorrowedFd {
        unsafe { BorrowedFd::borrow_raw(self.into()) }
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
*/
#[derive(Debug, Eq, Clone)]
pub struct EntryExt {
    entry: Entry,
    dirfd: DirFd,
    stat: OnceLock<Option<libc::stat>>,
}

impl EntryExt {
    /// Create an `EntryExt` from a given [Entry] and its parent directory fd.
    #[instrument(level = "trace")]
    pub fn new(entry: Entry, dirfd: DirFd) -> Self {
        Self {
            entry,
            dirfd,
            stat: OnceLock::new(),
        }
    }

    /// Create a new `EntryExt` and also `stat()` it before returning it.
    pub fn new_statted(entry: Entry, dirfd: DirFd) -> Self {
        let new_e: EntryExt = Self::new(entry, dirfd);
        new_e.stat();
        new_e
    }

    /**
    Return the [libc::stat] struct for the entry (may be cached).

    NOTE: If for some reason we cannot stat the entry, we return `None`.
    */
    pub fn stat(&self) -> Option<libc::stat> {
        if !self.dirfd.is_open() {
            warn!("Attempted to stat an entry with a closed directory fd: {:?}", self.dirfd);
            return None;
        }

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

    /// Open a [std::fs::File] object from a raw file descriptor.
    fn open(&self, flags: OFlag) -> io::Result<File> {
        if !self.dirfd.is_open() {
            warn!("Attempted to open a file from a closed directory fd: {:?}", self.dirfd);
            return Err(io::Error::new(io::ErrorKind::NotFound, "closed directory fd"));
        }

        let open_how: OpenHow = OpenHow::new()
            .flags(flags)
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
    #[inline]
    pub fn name(&self) -> &str {
        self.file_name().to_str().unwrap()
    }

    /**
    This relies on proc filesystem being available due to the use of
    `/proc/self/fd` to resolve the parent directory by file descriptor.
    */
    pub fn path(&self) -> io::Result<PathBuf> {
        match self.dirfd.path() {
            Ok(link) => Ok(link.join(self.name())),
            Err(e) => Err(e),
        }
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

impl PartialEq for EntryExt {
    fn eq(&self, other: &Self) -> bool {
        self.entry == other.entry && self.dirfd == other.dirfd
    }
}

impl Ord for EntryExt {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.file_name().cmp(&other.file_name())
    }
}

impl PartialOrd for EntryExt {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Deref for EntryExt {
    type Target = Entry;

    fn deref(&self) -> &Self::Target {
        &self.entry
    }
}

/* --------------------------------- */

impl Hash for EntryExt {
    #[inline]
    /// NOTE: this method will **not** produce stable hashes due to the standard
    /// `hash()` implementation's SipHash algorithm. Use `xxh3()` instead.
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.entry.hash(state);
        // TODO: should we hash the dirfd as well? If we do, we invalidate
        // the hash most likely if we close the directory and open it again.
        // Needs testing.
        //self.dirfd.hash(state);
    }
}

impl Xxh3Hashable for EntryExt {
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
    /// Directory count change (positive or negative)
    DirNum(i32),
    /// File count change (positive or negative)
    FileNum(i32),
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
#[derive(Clone, Hash)]
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
    First change seen is returned, so `DirNum` or `FileNum` change implies
    a `DirHash` or `FileHash` change, respectively.
    */
    pub fn change(&self, other: &Self) -> StateChange {
        if self.dirs != other.dirs {
            return StateChange::DirNum(self.dirs as i32 - other.dirs as i32);
        }
        if self.files != other.files {
            return StateChange::FileNum(self.files as i32 - other.files as i32);
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
type EntryVec = EnhVec<EntryExt>;

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

    /// Current state of the directory as a [DirectoryState] object.
    pub fn state_current(&mut self) -> DirectoryState {
        directory_state(self)
    }

    /// Whether the state has changed. Also updates the stored state if so.
    pub fn state_changed(&mut self) -> bool {
        let current: DirectoryState = self.state_current();
        let changed: StateChange = self.state.change(&current);
        self.state.update(current);
        !changed.is_same()
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
    */
    pub fn for_each<F>(&mut self, mut f: F)
    where
        F: FnMut(&Entry),
    {
        self.inner
            .iter()
            .filter_map(Result::ok)
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
    */
    pub fn iter(&mut self) -> DirHandleIter {
        DirHandleIter::new(self, false)
    }

    /// Same as `iter()`, but also explicitly `stat()`s each entry.
    pub fn iter_stat(&mut self) -> DirHandleIter {
        DirHandleIter::new(self, true)
    }

    /// Return the directory entries as a tuple of directories and files.
    /// The booleans specify whether to include directories and/or files.
    pub fn entries<'handle>(&'handle mut self, dirs: bool, files: bool) -> (EntryVec, EntryVec) {
        let mut d_vec: EntryVec = EnhVec::new();
        let mut f_vec: EntryVec = EnhVec::new();
        self.iter().for_each(|entry: EntryExt| {
            match entry.file_type() {
                Some(Type::Directory) => {
                    if dirs {
                        d_vec.push(entry)
                    }
                }
                Some(_) => {
                    if files {
                        f_vec.push(entry)
                    }
                }
                None => {} // ignore unknown file types
            }
        });
        (d_vec, f_vec)
    }

    /// Return the directory entries as sorted tuples of directories and files.
    pub fn entries_sorted<'handle>(&'handle mut self) -> (EntryVec, EntryVec) {
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
    pub fn iter_sorted(&mut self) -> DirHandleIterSorted {
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

// DirHandle can be Send, since the underlying nix::dir::Dir is Send as well.
unsafe impl Send for DirHandle {}

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
#[derive(Debug, Hash)]
struct BufDeque<T>(VecDeque<T>);

impl BufDeque<EntryExt> {
    pub fn new(capacity: usize) -> Self {
        Self(VecDeque::with_capacity(capacity))
    }

    /// Push an entry to the buffer. Directories to the front, rest to back.
    pub fn push(&mut self, entry: EntryExt) {
        if entry.is_dir() {
            self.push_front(entry);
        } else {
            self.push_back(entry);
        }
    }

    /// Any directory entries in the buffer?
    pub fn has_dir(&self) -> bool {
        self.iter().any(|entry: &EntryExt| entry.is_dir())
    }

    /**
    Try to pop a directory entry. The first directory found is chosen,
    but if none are in the buffer, we pop the oldest (front) entry.
    */
    pub fn try_pop_dir(&mut self) -> Option<EntryExt> {
        if self.is_empty() {
            return None;
        }
        self.iter()
            .position(|entry: &EntryExt| entry.is_dir())
            .map(|idx: usize| self.remove(idx))
            .unwrap_or_else(|| self.pop_front())
    }
}

/* --------------------------------- */

impl Default for BufDeque<EntryExt> {
    fn default() -> Self {
        Self::new(LOOKAHEAD_BUFFER_SIZE)
    }
}

impl<T> Deref for BufDeque<T> {
    type Target = VecDeque<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for BufDeque<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/* ######################################################################### */

/// The return type of [DirHandle::iter]
#[derive(Debug)]
pub struct DirHandleIter<'handle> {
    inner: Peekable<Iter<'handle>>,
    dirfd: DirFd,
    buf: BufDeque<EntryExt>,
    state: &'handle mut DirectoryState,
    stat: bool,
    /// clones of dir entries - used for hashing
    dirs: EntryVec,
    /// clones of file entries - used for hashing
    files: EntryVec,
    /// resettable hasher for calculating entry hashes
    xxh: Option<CustomXxh3Hasher>,
}

impl<'handle> DirHandleIter<'handle> {
    pub fn new(handle: &'handle mut DirHandle, stat: bool) -> Self {
        let update: bool = handle.state.when.is_none();
        Self {
            dirfd: handle.fd(),
            inner: handle.inner.iter().peekable(),
            buf: BufDeque::default(),
            state: &mut handle.state,
            stat,
            dirs: EntryVec::new(),
            files: EntryVec::new(),
            xxh: match update {
                true => Some(CustomXxh3Hasher::default()),
                false => None,
            },
        }
    }

    /// Shall we update the [DirectoryState] struct after iter is done?
    fn update(&mut self) -> bool {
        self.xxh.is_some()
    }

    /// Is the inner iterator done?
    #[inline]
    fn done(&mut self) -> bool {
        self.inner.peek().is_none()
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
                e.file_type()
                    .map_or(None, |t: Type| Some(t == Type::Directory))
            })
        })
    }

    /// Get one entry from the inner iterator.
    fn get_one(&mut self) -> Option<EntryExt> {
        if let Some(entry) = next(&mut self.inner, &self.dirfd, self.stat) {
            if self.update() {
                // store a clone of each entry for later use
                let ec: EntryExt = entry.clone();

                #[cfg(debug_assertions)]
                {
                    use custom_xxh3::hash_item;
                    let mut xxh: &mut CustomXxh3Hasher = self.xxh.as_mut().unwrap();
                    ec.xxh3(&mut xxh);
                    let vec: EntryVec = EnhVec::new_from(vec![entry.clone()]);
                    debug!(target: "get_one",
                "{:?} : xxh3(entry): 0x{:x}, vec_digest: 0x{:x}, hash_item(entry): 0x{:x}, hash_item(vec): 0x{:x}",
                entry.name(), xxh.reset(), vec.xxh3_digest(), hash_item(&entry), hash_item(&vec));
                } // END DEBUG

                match entry.file_type() {
                    Some(Type::Directory) => self.dirs.push(ec),
                    Some(_) => self.files.push(ec),
                    None => {} // unknown file type
                }
            }
            Some(entry)
        } else {
            None
        }
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
    type Item = EntryExt;

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
                                self.get_one().map(|extra: EntryExt| {
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
                                });
                            }
                        }
                    }
                }
            } else if self.done() {
                debug!(target: "DirHandleIter::next", "iter_done: {:?}", self.inner);
                if self.update() {
                    // hash the sorted entries and set DirHandle state
                    let mut xxh: CustomXxh3Hasher = self.xxh.take().unwrap();
                    xxh.reset(); // just in case...

                    self.dirs.as_sorted_asc().iter().for_each(|entry| {
                        entry.xxh3(&mut xxh);
                    });
                    self.state.hash_d = xxh.reset();

                    self.files.as_sorted_asc().iter().for_each(|entry| {
                        entry.xxh3(&mut xxh);
                    });
                    self.state.hash_f = xxh.finish();

                    self.state.dirs = self.dirs.len();
                    self.state.files = self.files.len();
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
pub struct DirHandleIterSorted<'handle>(EntryVec, PhantomData<&'handle DirHandle>);

impl<'handle> Iterator for DirHandleIterSorted<'handle> {
    type Item = EntryExt;

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
    pub fn open(&self, path: &Path) -> io::Result<CheckedOutHandle> {
        let handle: DirHandle = DirHandle::new(path)?;
        let fd: RawFd = handle.as_raw_fd();
        self.0.insert(fd, handle);
        Ok(self.checkout(fd).unwrap())
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
    pub fn get(&self, fd: RawFd) -> Option<CheckedOutHandle> {
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

// OpenHandles is thread-safe due to the internal DashMap being thread-safe.
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
    // Using `Rc` instead of `Arc` here is deliberate because we don't want to
    // share the callback between threads and this also disallows moving a
    // checked-out handle to another thread.
    close_callback: Rc<dyn Fn(RawFd) + 'a>,
}

impl<'a> CheckedOutHandle<'a> {
    /// Close this [DirHandle] and release its file descriptor.
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

/// Open a directory and return its handle.
fn get_dir_handle(path: &Path) -> io::Result<Dir> {
    Ok(Dir::open(path, OFlag::O_RDONLY, Mode::empty())?)
}

/// Open a file and return its handle.
fn get_file_handle(path: &Path) -> io::Result<File> {
    Ok(OpenOptions::new().read(true).open(path)?)
}

/// Return the state of a directory as a [DirectoryState] object.
#[instrument(level = "trace", skip_all, ret)]
fn directory_state(dir: &mut DirHandle) -> DirectoryState {
    let (dirs, files) = dir.entries_sorted();
    DirectoryState {
        dirs: dirs.len(),
        files: files.len(),
        hash_d: dirs.xxh3_digest(),
        hash_f: files.xxh3_digest(),
        when: TimeSinceEpoch::new().into(),
    }
}

/**
Return the next entry from the inner [nix::dir::Iter] as an [EntryExt],
skipping `.` and `..`. Also skips entries where the file type cannot be
determined (e.g. due to permission denied).

If `stat == true`, we also `stat()` the entry before returning it.
*/
#[instrument(level = "trace", skip(it))]
fn next(it: &mut Peekable<Iter>, dirfd: &DirFd, stat: bool) -> Option<EntryExt> {
    it.filter_map(Result::ok)
        .filter_map(|entry| {
            // Sadly it appears that we cannot rely on the special "." and ".."
            // entries being returned first by the libc `readdir` call, so to
            // filter them out we must match each name.
            if matches!(entry.file_name().to_bytes(), DOT1 | DOT2) {
                return None;
            }

            // convert the [nix::dir::Entry] to our `EntryExt`
            let entry: EntryExt = match stat {
                false => EntryExt::new(entry, dirfd.clone()),
                true => EntryExt::new_statted(entry, dirfd.clone()),
            };
            if let Some(_) = entry.file_type() {
                trace!(target: "name", "{:?} : {:?}", entry.name(), entry);
                Some(entry)
            } else {
                // we ignore the entry if we can't determine its type
                // (e.g. permission denied, unknown type)
                trace!(target: "name", "{:?} : {:?} (skipping, no type)", entry.name(), entry);
                return None;
            }
        })
        .next()
}
