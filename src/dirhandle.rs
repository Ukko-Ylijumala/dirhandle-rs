// Copyright (c) 2024 Mikko Tanner. All rights reserved.

#![allow(dead_code)]

use core::ffi::CStr;
use libc;
use nix::{
    dir::{Dir, Entry, Iter, Type},
    fcntl::{openat2, AtFlags, OFlag, OpenHow, ResolveFlag},
    sys::stat::{fstatat, Mode},
};
use std::{
    fs::{read_link, File, OpenOptions},
    hash::{Hash, Hasher},
    io,
    marker::PhantomData,
    os::fd::{AsRawFd, FromRawFd, RawFd},
    path::{Path, PathBuf},
    sync::OnceLock,
};

const DOT1: &[u8] = b".";
const DOT2: &[u8] = b"..";

/// Since we cannot import `std::sys` directly (it's private), we need to
/// define our own `EntryType`, which is functionally a copy of the original
/// `std::sys::pal::unix::fs::FileType`.
#[derive(Debug, Clone, Copy, Hash)]
struct EntryType(libc::mode_t);

impl EntryType {
    pub fn is_dir(&self) -> bool {
        self.is(libc::S_IFDIR)
    }
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

    fn is(&self, mode: libc::mode_t) -> bool {
        self.masked() == mode
    }
    fn masked(&self) -> libc::mode_t {
        self.0 & libc::S_IFMT
    }
}

/// This struct extends the `nix::dir::Entry` struct with additional methods.
/// The aim is to be closely compatible with the `std::fs::DirEntry` API.
///
/// Notable differences:
/// - `metadata()` is replaced with `stat()`, and we return a `libc::stat` struct
/// - `file_type()` is replaced with a custom implementation, which uses `fstatat()`
///   if the file type is not available in the `dirent` struct
/// - the stat result is cached in a `OnceLock` to avoid calling `fstatat()`
///   multiple times for the same entry.
#[derive(Debug, Eq, Clone)]
pub struct EntryExt<'handle> {
    entry: Entry,
    dirfd: i32,
    stat: OnceLock<Option<libc::stat>>,
    phantom: PhantomData<&'handle DirHandle>,
}

impl<'handle> EntryExt<'handle> {
    pub fn new(entry: Entry, dirfd: i32) -> Self {
        Self {
            entry,
            dirfd,
            stat: OnceLock::new(),
            phantom: PhantomData,
        }
    }

    /// Return the `libc::stat` struct for the entry (may be cached).
    ///
    /// NOTE: If for some reason we cannot stat the entry, we return `None`.
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

    /// Return the size of the file, in bytes.
    ///
    /// Note that the size is returned as a `u64` to match the `std::fs::Metadata`
    /// API, even though the underlying `libc::stat` struct uses `i64` for the size.
    /// Also, if for some reason we cannot stat() the file, we return `0`.
    pub fn len(&self) -> u64 {
        match self.stat() {
            Some(stat) => stat.st_size as u64,
            None => 0,
        }
    }

    /// Return the mode of the file as a `libc::mode_t` value.
    pub fn mode(&self) -> Option<libc::mode_t> {
        match self.stat() {
            Some(stat) => Some(stat.st_mode),
            None => None,
        }
    }

    /// Returns the inode number (`d_ino`) of the underlying `dirent`.
    pub fn ino(&self) -> u64 {
        self.entry.ino()
    }

    /// Open a `std::fs::File` object from a raw file descriptor.
    fn open(&self, flags: OFlag) -> io::Result<File> {
        let open_how: OpenHow = OpenHow::new()
            .flags(flags)
            .resolve(ResolveFlag::RESOLVE_BENEATH);
        let fd: i32 = openat2(self.dirfd, self.file_name(), open_how)?;
        Ok(unsafe { File::from_raw_fd(fd) })
    }

    /// Open this entry for reading as a `std::fs::File` object.
    pub fn read(&self) -> io::Result<File> {
        self.open(OFlag::O_RDONLY)
    }

    /// Open this entry for read+write as a `std::fs::File` object.
    pub fn write(&self) -> io::Result<File> {
        self.open(OFlag::O_RDWR)
    }

    #[inline]
    pub fn file_name(&self) -> &CStr {
        self.entry.file_name()
    }
    pub fn name_as_bytes(&self) -> &[u8] {
        self.file_name().to_bytes()
    }
    pub fn name(&self) -> &str {
        self.file_name().to_str().unwrap()
    }

    /// This relies on proc filesystem being available due to the use of
    /// `/proc/self/fd` to resolve the parent directory by file descriptor.
    pub fn path(&self) -> io::Result<PathBuf> {
        match read_link(format!("/proc/self/fd/{}", self.dirfd)) {
            Ok(link) => Ok(link.join(self.name())),
            Err(e) => Err(e),
        }
    }

    /// Return the file type of the entry as a `nix::dir::Type` enum.
    pub fn file_type(&self) -> Option<Type> {
        if let Some(entry_type) = self.entry.file_type() {
            Some(entry_type)
        } else {
            match self.mode() {
                Some(mode) => {
                    let e_t: EntryType = EntryType(mode);
                    if e_t.is_dir() {
                        Some(Type::Directory)
                    } else if e_t.is_file() {
                        Some(Type::File)
                    } else if e_t.is_symlink() {
                        Some(Type::Symlink)
                    } else if e_t.is_fifo() {
                        Some(Type::Fifo)
                    } else if e_t.is_char() {
                        Some(Type::CharacterDevice)
                    } else if e_t.is_block() {
                        Some(Type::BlockDevice)
                    } else if e_t.is_sock() {
                        Some(Type::Socket)
                    } else {
                        None // unknown file type?!?
                    }
                }
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

impl<'handle> Hash for EntryExt<'handle> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.entry.hash(state);
        self.dirfd.hash(state);
    }
}

impl<'handle> PartialEq for EntryExt<'handle> {
    fn eq(&self, other: &Self) -> bool {
        self.entry == other.entry && self.dirfd == other.dirfd
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

/// A wrapper around the raw file descriptor of a `DirHandle` object.
/// The lifetime parameter `'handle` is used to annotate that the raw file
/// descriptor is only valid while the `DirHandle` object exists.
struct DirHandleRawFd<'handle>(RawFd, PhantomData<&'handle DirHandle>);

type EntryVec<'a> = Vec<EntryExt<'a>>;

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct DirHandle(Dir);

impl DirHandle {
    /// Open a directory by path and return its handle object.
    pub fn new(path: &Path) -> io::Result<Self> {
        Ok(Self(get_dir_handle(path)?))
    }

    /// Return the raw file descriptor of the inner `nix::dir::Dir` object.
    fn raw_fd<'handle>(&'handle self) -> DirHandleRawFd<'handle> {
        DirHandleRawFd(self.0.as_raw_fd(), PhantomData)
    }

    /// This relies on proc filesystem being available due to the use of
    /// `/proc/self/fd` to resolve the path by file descriptor.
    pub fn path(&self) -> io::Result<PathBuf> {
        Ok(read_link(format!("/proc/self/fd/{}", self.raw_fd().0))?)
    }

    /// Return the inner `nix::dir::Iter` object directly.
    pub fn raw_iter(&mut self) -> Iter {
        self.0.iter()
    }

    /// This iterator does the following:
    /// - skips the special `.` and `..` entries
    /// - returns directory entries before file entries
    /// - optionally sorts directory and file entries alphabetically (separately)
    pub fn iter<'handle>(&'handle mut self, sort: bool) -> DirectoryIterator<'handle> {
        let mut dirs: EntryVec = Vec::new();
        let mut files: EntryVec = Vec::new();
        self.gather_entries(&mut dirs, &mut files);

        if sort {
            // Note: we sort in reverse order to pop the entries in alphabetical order
            dirs.sort_by(|a, b| b.file_name().cmp(a.file_name()));
            files.sort_by(|a, b| b.file_name().cmp(a.file_name()));
        }
        DirectoryIterator(files.into_iter().chain(dirs.into_iter()).collect(), PhantomData)
    }

    /// Common code for `iter()` and `into_iter()`. Gather directory and file
    /// entries into the given vectors from the inner `nix::dir::Iter`.
    #[inline]
    fn gather_entries<'handle>(&'handle mut self, dirs: &mut EntryVec, files: &mut EntryVec) {
        let dirfd: i32 = self.raw_fd().0;
        self.raw_iter()
            .filter_map(Result::ok)
            .for_each(|entry: Entry| {
                if matches!(entry.file_name().to_bytes(), DOT1 | DOT2) {
                    return;
                }

                // convert the `nix::dir::Entry` to our `DirEntryExt`
                let entry: EntryExt = EntryExt::new(entry, dirfd);

                if let Some(entry_type) = entry.file_type() {
                    if entry_type == Type::Directory {
                        dirs.push(entry);
                    } else if entry_type == Type::File {
                        files.push(entry);
                    }
                } else {
                    // we ignore the entry if we can't determine its type
                    // (e.g. permission denied)
                    return;
                }
            });
    }
}

/// An iterator over the entries in a directory.
///
/// The lifetime parameter `'handle` is used to annotate that the entries
/// are only valid while the `DirHandle` object exists.
pub struct DirectoryIterator<'handle>(EntryVec<'handle>, PhantomData<&'handle DirHandle>);

impl<'handle> Iterator for DirectoryIterator<'handle> {
    type Item = EntryExt<'handle>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.pop()
    }
}

impl IntoIterator for DirHandle {
    type Item = EntryExt<'static>;
    type IntoIter = DirectoryIterator<'static>;

    fn into_iter(mut self) -> Self::IntoIter {
        let mut dirs: EntryVec = Vec::new();
        let mut files: EntryVec = Vec::new();
        self.gather_entries(&mut dirs, &mut files);
        DirectoryIterator(files.into_iter().chain(dirs.into_iter()).collect(), PhantomData)
    }
}

/* Obsolete / WIP code left here for reference
use std::ops::ControlFlow;

self.raw_iter()
.filter_map(Result::ok)
.for_each(|entry: Entry| {
    if let ControlFlow::Break(_) = process_entry(entry, dirfd, &mut dirs, &mut files) {
        return;
    }
});

#[inline]
fn process_entry(
    entry: Entry,
    dirfd: i32,
    dirs: &mut EntryVec,
    files: &mut EntryVec,
) -> ControlFlow<()> {
    if matches!(entry.file_name().to_bytes(), DOT1 | DOT2) {
        return ControlFlow::Break(());
    }

    // convert the `nix::dir::Entry` to our `EntryExt`
    let entry: EntryExt = EntryExt::new(entry, dirfd);

    if let Some(entry_type) = entry.file_type() {
        if entry_type == Type::Directory {
            dirs.push(entry);
        } else if entry_type == Type::File {
            files.push(entry);
        }
    } else {
        // we ignore the entry if we can't determine its type
        // (e.g. permission denied)
        return ControlFlow::Break(());
    }
    ControlFlow::Continue(())
}
*/
