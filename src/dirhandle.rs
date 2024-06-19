// Copyright (c) 2024 Mikko Tanner. All rights reserved.

#![allow(dead_code)]

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

impl Hash for EntryExt {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.entry.hash(state);
        self.dirfd.hash(state);
    }
}

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

/// Open a directory and return its handle.
fn get_dir_handle(path: &Path) -> io::Result<Dir> {
    Ok(Dir::open(path, OFlag::O_RDONLY, Mode::empty())?)
}

/// Open a file and return its handle.
fn get_file_handle(path: &Path) -> io::Result<File> {
    Ok(OpenOptions::new().read(true).open(path)?)
}

/**
A wrapper around the raw file descriptor of a `DirHandle` object.
The lifetime parameter `'handle` is used to annotate that the raw file
descriptor is only valid while the `DirHandle` object exists.
*/
struct DirHandleRawFd<'handle>(RawFd, PhantomData<&'handle DirHandle>);

type EntryVec = Vec<EntryExt>;

/**
An open handle to a directory, internally a [nix::dir::Dir] object. The aim
is to be somewhat compatible with the [std::fs::ReadDir] API, with the
following notable differences / enhancements:

- `iter_sort()` sorts the returned entries alphabetically
- `iter()` rewinds after finishing, so it can be called multiple times
- `path()` returns the canonicalized path of the directory
  (relies on procfs being available and the inner Dir being open)
- change detection in the directory (or parts therein) is facilitated by:
  - numbers of directories and files are cached
  - hashes of directory and file entry names are cached

NOTE: the counts and hashes are updated when the directory is iterated,
or when asked for explicitly, but they are not updated automatically
if the directory is changed externally.

NOTE: the `DirHandle` object is not thread-safe, and it is not intended
to be shared between threads, see the following `readdir` manual:
https://www.gnu.org/software/libc/manual/html_node/Reading_002fClosing-Directory.html
Future versions of POSIX are likely to obsolete `readdir_r` and specify that it's
unsafe to call `readdir` simultaneously from multiple threads.
*/
#[derive(Debug, Eq)]
pub struct DirHandle {
    inner: Dir,
}

impl DirHandle {
    /// Open a directory by path and return its handle object.
    #[instrument(level = "trace")]
    pub fn new(path: &Path) -> io::Result<Self> {
        Ok(Self {
            inner: get_dir_handle(path)?,
        })
    }

    /// Return the raw file descriptor of the inner [nix::dir::Dir] object.
    fn raw_fd<'handle>(&'handle self) -> DirHandleRawFd<'handle> {
        DirHandleRawFd(self.as_raw_fd(), PhantomData)
    }

    /**
    This relies on proc filesystem being available due to the use of
    `/proc/self/fd` to resolve the path by file descriptor.
    */
    pub fn path(&self) -> io::Result<PathBuf> {
        Ok(read_link(format!("/proc/self/fd/{}", self.raw_fd().0))?)
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
        DirHandleIter {
            dirfd: self.as_raw_fd(),
            inner: self.inner.iter().peekable(),
            buf: BufDeque::default(),
        }
    }

    /**
    This iterator does the following:
    - skips the special `.` and `..` entries
    - returns directory entries before all other entries
    - sorts both entry lists alphabetically (separately)
    */
    pub fn iter_sorted<'handle>(&'handle mut self) -> DirHandleIterSorted<'handle> {
        let mut dirs: EntryVec = Vec::new();
        let mut files: EntryVec = Vec::new();
        self.iter().for_each(|entry: EntryExt| {
            match entry.file_type() {
                Some(Type::Directory) => dirs.push(entry),
                Some(_) => files.push(entry),
                None => {} // ignore unknown file types
            }
        });

        // NOTE: we sort in reverse order to pop the entries in alphabetical order
        dirs.sort_by(|a, b| b.cmp(a));
        files.sort_by(|a, b| b.cmp(a));
        DirHandleIterSorted(files.into_iter().chain(dirs.into_iter()).collect(), PhantomData)
    }
}

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

/// The return type of [DirHandle::iter]
#[derive(Debug)]
pub struct DirHandleIter<'handle> {
    inner: Peekable<Iter<'handle>>,
    dirfd: RawFd,
    buf: BufDeque<EntryExt>,
}

impl<'handle> DirHandleIter<'handle> {
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
    #[inline]
    fn get_one(&mut self) -> Option<EntryExt> {
        next(&mut self.inner, self.dirfd)
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
                return None;
            } else if self.buf.is_empty() {
                self.fill_buffer();
            }
        }
    }
}

/**
A sorted iterator over the entries in a directory.

The lifetime parameter `'handle` is used to annotate that the entries
in the Vec are only valid while the `DirHandle` object exists.
*/
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DirHandleIterSorted<'handle>(EntryVec, PhantomData<&'handle DirHandle>);

impl<'handle> Iterator for DirHandleIterSorted<'handle> {
    type Item = EntryExt;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.pop()
    }
}
