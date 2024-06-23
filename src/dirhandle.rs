// Copyright (c) 2024 Mikko Tanner. All rights reserved.

#![allow(dead_code)]

use crate::enhvec::EnhVec;
use crate::hashing::{hash_item, CustomXxh3Hasher, Xxh3Hashable};
use crate::timesince::TimeSinceEpoch;
use libc;
use nix::{
    dir::{Dir, Entry, Iter, Type},
    fcntl::{openat2, AtFlags, OFlag, OpenHow, ResolveFlag},
    sys::stat::{fstatat, Mode},
};
use std::{
    cmp::Ordering,
    collections::VecDeque,
    fs::{read_link, File, OpenOptions},
    hash::{Hash, Hasher},
    io,
    iter::Peekable,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    os::fd::{AsRawFd, FromRawFd, RawFd},
    path::{Path, PathBuf},
    sync::OnceLock,
};
use tracing::{debug, instrument, trace, warn};

const DOT1: &[u8] = b".";
const DOT2: &[u8] = b"..";
const LOOKAHEAD_BUFFER_SIZE: usize = 64;

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
    dirfd: i32,
    stat: OnceLock<Option<libc::stat>>,
}

impl EntryExt {
    #[instrument(level = "trace")]
    pub fn new(entry: Entry, dirfd: i32) -> Self {
        Self {
            entry,
            dirfd,
            stat: OnceLock::new(),
        }
    }

    /**
    Return the [libc::stat] struct for the entry (may be cached).

    NOTE: If for some reason we cannot stat the entry, we return `None`.
    */
    pub fn stat(&self) -> Option<libc::stat> {
        *self.stat.get_or_init(|| {
            match fstatat(Some(self.dirfd), self.file_name(), AtFlags::AT_SYMLINK_NOFOLLOW) {
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
        let open_how: OpenHow = OpenHow::new()
            .flags(flags)
            .resolve(ResolveFlag::RESOLVE_BENEATH);
        let fd: i32 = openat2(self.dirfd, self.file_name(), open_how)?;
        Ok(unsafe { File::from_raw_fd(fd) })
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
        match read_link(format!("/proc/self/fd/{}", self.dirfd)) {
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

    pub fn is_file(&self) -> bool {
        matches!(self.file_type(), Some(Type::File))
    }
    pub fn is_dir(&self) -> bool {
        matches!(self.file_type(), Some(Type::Directory))
    }
    pub fn is_symlink(&self) -> bool {
        matches!(self.file_type(), Some(Type::Symlink))
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
        self.entry.hash(state);
    }

    fn xxh3_digest(&self) -> u64 {
        hash_item(&self.entry)
    }
}

/* ######################################################################### */

/**
The type of change detected in a directory, if any.

`DirectoryNum` or `FileNum` change also implies a change of `DirectoryHash`
or `FileHash`, respectively, but the reverse is not true. But since we first
check for `DirectoryNum` and `FileNum` changes, that doesn't matter.
*/
#[derive(Debug)]
pub enum StateChange {
    Unchanged,
    /// Directory count change (positive or negative)
    DirectoryNum(i32),
    /// File count change (positive or negative)
    FileNum(i32),
    /// Same directory count, but hash of dir entries has changed
    DirectoryHash,
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
#[derive(Debug)]
pub struct DirectoryState {
    num_dirs: usize,
    num_files: usize,
    hash_dirs: u64,
    hash_files: u64,
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

    /// Compare two states and return the type of change as a [StateChange] enum.
    pub fn change_type(&self, other: &Self) -> StateChange {
        if self.num_dirs != other.num_dirs {
            return StateChange::DirectoryNum(self.num_dirs as i32 - other.num_dirs as i32);
        }
        if self.num_files != other.num_files {
            return StateChange::FileNum(self.num_files as i32 - other.num_files as i32);
        }
        if self.hash_dirs != other.hash_dirs {
            return StateChange::DirectoryHash;
        }
        if self.hash_files != other.hash_files {
            return StateChange::FileHash;
        }
        StateChange::Unchanged
    }

    /// Combined hash of directory and file entry hashes.
    pub fn hash_all(&self) -> u64 {
        self.hash_dirs.rotate_left(32) ^ self.hash_files
    }
}

impl Default for DirectoryState {
    fn default() -> Self {
        Self {
            num_dirs: 0,
            num_files: 0,
            hash_dirs: 0,
            hash_files: 0,
            when: None,
        }
    }
}

impl Eq for DirectoryState {}

impl PartialEq for DirectoryState {
    fn eq(&self, other: &Self) -> bool {
        self.num_dirs == other.num_dirs
            && self.num_files == other.num_files
            && self.hash_dirs == other.hash_dirs
            && self.hash_files == other.hash_files
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
  - hashes of directory and file entry names are cached

NOTE: the counts and hashes are updated when the directory is iterated for
the first time, or when asked for explicitly, but they are **not** updated
automatically if the directory is changed externally.

NOTE: the [DirHandle] object is not thread-safe, and it is not intended
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

    /**
    This relies on proc filesystem being available due to the use of
    `/proc/self/fd` to resolve the path by file descriptor.
    */
    pub fn path(&self) -> io::Result<PathBuf> {
        Ok(read_link(format!("/proc/self/fd/{}", self.as_raw_fd()))?)
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
        let changed: StateChange = self.state.change_type(&current);
        self.state.update(current);
        !changed.is_same()
    }

    /// Return the inner [nix::dir::Iter] object and make it `Peekable`.
    pub fn raw_iter<'handle>(&'handle mut self) -> Peekable<Iter<'handle>> {
        self.inner.iter().peekable()
    }

    /**
    This iterator does the following:
    - skips the special `.` and `..` entries
    - rewinds after finishing
    - maintains a lookahead buffer to preferentially return directory
      entries before other entries (NOTE: not a guarantee)
    */
    pub fn iter<'handle>(&'handle mut self) -> DirHandleIter<'handle> {
        DirHandleIter::new(self)
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
        dirs.sort_by(|a, b| a.cmp(b));
        files.sort_by(|a, b| a.cmp(b));
        (dirs, files)
    }

    /**
    This iterator does the following:
    - skips the special `.` and `..` entries
    - returns directory entries before all other entries
    - sorts both entry lists alphabetically (separately)
    */
    pub fn iter_sorted<'handle>(&'handle mut self) -> DirHandleIterSorted<'handle> {
        let (mut dirs, mut files) = self.entries(true, true);

        // NOTE: we sort in reverse order to pop the entries in alphabetical order
        dirs.sort_by(|a, b| b.cmp(a));
        files.sort_by(|a, b| b.cmp(a));
        files.extend_from_slice(&dirs);
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
    The file descriptor continues to be owned by the `DirHandle`, so
    callers must not keep a `RawFd` after the `DirHandle` is dropped.
    */
    fn as_raw_fd(&self) -> RawFd {
        self.inner.as_raw_fd()
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
        self.iter().any(|entry| entry.is_dir())
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
            .position(|entry| entry.is_dir())
            .map(|idx| self.remove(idx))
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
    dirfd: RawFd,
    buf: BufDeque<EntryExt>,
    state: &'handle mut DirectoryState,
    /// dir entry names and corresponding hashes
    dirs: Vec<(String, u64)>,
    /// file entry names and corresponding hashes
    files: Vec<(String, u64)>,
    /// resettable hasher for calculating entry hashes
    xxh: CustomXxh3Hasher,
    update: bool,
}

impl<'handle> DirHandleIter<'handle> {
    pub fn new(handle: &'handle mut DirHandle) -> Self {
        Self {
            dirfd: handle.as_raw_fd(),
            inner: handle.inner.iter().peekable(),
            buf: BufDeque::default(),
            update: handle.state.when.is_none(),
            state: &mut handle.state,
            dirs: Vec::new(),
            files: Vec::new(),
            xxh: CustomXxh3Hasher::default(),
        }
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
        if let Some(entry) = next(&mut self.inner, self.dirfd) {
            if self.update {
                // store each entry name and its hash for later use
                entry.hash(&mut self.xxh);

                #[cfg(debug_assertions)]
                {
                    let tmp_hash = hash_item(&entry);
                    let tmp_vec: EntryVec = EnhVec::new_from(vec![entry.clone()]);
                    let vec_hash = hash_item(&tmp_vec);
                    debug!(target: "get_one",
                "{:?} : xxh_hash: {}, tmp_hash: {tmp_hash:?}, vec_hash: {vec_hash}, vec_xxh3: {}",
                entry.name(), self.xxh.finish(), tmp_vec.xxh3_digest());
                } // END DEBUG -- TODO: REMOVE

                match entry.file_type() {
                    Some(Type::Directory) => {
                        self.dirs.push((entry.name().to_string(), self.xxh.reset()))
                    }
                    Some(_) => self
                        .files
                        .push((entry.name().to_string(), self.xxh.reset())),
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
                if self.update {
                    // sort the data, use the hasher to build state from
                    // precalculated values and then set DirHandle state
                    self.dirs.sort_by(|a, b| a.0.cmp(&b.0));
                    self.files.sort_by(|a, b| a.0.cmp(&b.0));
                    self.state.num_dirs = self.dirs.len();
                    self.state.num_files = self.files.len();
                    self.state.when = TimeSinceEpoch::new().into();

                    self.xxh.reset(); // just in case...
                    self.dirs.iter().for_each(|(_, hash)| {
                        self.xxh.write_u64(*hash);
                    });
                    self.state.hash_dirs = self.xxh.reset();

                    self.files.iter().for_each(|(_, hash)| {
                        self.xxh.write_u64(*hash);
                    });
                    self.state.hash_files = self.xxh.reset();

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
in the Vec are only valid while the `DirHandle` object exists.
*/
#[derive(Debug)]
pub struct DirHandleIterSorted<'handle>(EntryVec, PhantomData<&'handle DirHandle>);

impl<'handle> Iterator for DirHandleIterSorted<'handle> {
    type Item = EntryExt;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.pop()
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
        num_dirs: dirs.len(),
        num_files: files.len(),
        hash_dirs: dirs.xxh3_digest(),
        hash_files: files.xxh3_digest(),
        when: TimeSinceEpoch::new().into(),
    }
}

/**
Return the next entry from the inner [nix::dir::Iter] as an `EntryExt`,
skipping `.` and `..`. Also skips entries where the file type cannot be
determined (e.g. due to permission denied).
*/
#[instrument(level = "trace", skip(it))]
fn next(it: &mut Peekable<Iter>, dirfd: RawFd) -> Option<EntryExt> {
    it.filter_map(Result::ok)
        .filter_map(|entry| {
            // Sadly it appears that we cannot rely on the special "." and ".."
            // entries being returned first by the libc `readdir` call, so to
            // filter them out we must match each name.
            if matches!(entry.file_name().to_bytes(), DOT1 | DOT2) {
                return None;
            }

            // convert the [nix::dir::Entry] to our `EntryExt`
            let entry: EntryExt = EntryExt::new(entry, dirfd);
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
