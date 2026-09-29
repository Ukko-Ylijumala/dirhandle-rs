// Copyright (c) 2024-2026 Mikko Tanner. All rights reserved.

use custom_xxh3::{CustomXxh3Hasher, Xxh3Hashable};
use dashmap::{mapref::one::RefMut, DashMap};
use enhvec::{EnhVec, Sorting};
use miniutils::{ToDebug, ToDisplay};
use nix::{
    dir::{Dir, Entry, Iter, Type},
    errno::Errno,
    fcntl::{openat2, AtFlags, OFlag, OpenHow, ResolveFlag},
    sys::stat::{fstat, fstatat, Mode},
};
use std::{
    cmp::{Eq, Ord, Ordering, PartialEq, PartialOrd},
    collections::VecDeque,
    ffi::{CStr, OsStr},
    fmt::{self, Debug, Display, Formatter},
    fs::{read_link, File, OpenOptions},
    hash::{Hash, Hasher},
    io,
    iter::Peekable,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    os::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd, RawFd},
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
Slack for the `state_changed_fast()` mtime pre-check: filesystem
timestamp granularity (1s on older filesystems), f64 rounding and
minor clock skew. Timestamps within this window of the baseline are
treated as "maybe changed" and fall through to the full comparison.
*/
const MTIME_SLACK_SECS: f64 = 2.0;
/*
Stable typenum values for `EntryExt::typenum()`. They feed the
[DirectoryState] digests, so they are pinned to the kernel's `d_type`
ABI (`DT_*`) rather than derived from nix's [Type] discriminants, which
upstream could reorder and thereby silently change every stored hash.
NOTE: v0.4 used the nix enum order (0..6, unknown = 254); v0.5.0 switched
to `DT_*` together with the digest fold, so 0.4 and 0.5 hashes differ.
*/
const TYPENUM_FIFO:    u8 = libc::DT_FIFO;
const TYPENUM_CHR:     u8 = libc::DT_CHR;
const TYPENUM_DIR:     u8 = libc::DT_DIR;
const TYPENUM_BLK:     u8 = libc::DT_BLK;
const TYPENUM_REG:     u8 = libc::DT_REG;
const TYPENUM_LNK:     u8 = libc::DT_LNK;
const TYPENUM_SOCK:    u8 = libc::DT_SOCK;
const TYPENUM_UNKNOWN: u8 = libc::DT_UNKNOWN;
/*
Inline capacity of [EntryName] in bytes, including the trailing NUL. 39
keeps the enum at 48 bytes (payload + tag, 8-aligned because of the heap
variant's `Box`) while fitting names of up to 38 bytes inline - UUIDs
(36) included. Anything longer takes one heap allocation.
*/
const NAME_INLINE_CAP: usize = 39;
/*
Heap footprint of one open directory stream as allocated by glibc's
`opendir` / `fdopendir` (`sysdeps/posix/opendir.c`): a `struct __dirstream`
header (fd, lock, allocation / size / offset bookkeeping, ~48 bytes)
followed by the inline `getdents` buffer, which is `max(st_blksize, 32 KiB)`
capped at 1 MiB. 32 KiB covers every common filesystem. `nix::dir::Dir`
itself is only the `DIR*` and lives inline in [DirHandle]; the `dirent`
that `readdir_r` fills is stack-allocated per call, not heap.
*/
#[cfg(feature = "size_of")]
const DIR_STREAM_HEAP: usize = 32 * 1024 + 48;

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
- `DirFd >= 0` : Open file descriptor. `fd == 0` is a valid (open) fd -
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
    before or after - never in between the check and the store.
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
    - the directory has been deleted (`NotFound`; see [proc_fd_path])
    */
    pub fn path(&self) -> io::Result<PathBuf> {
        let fd: RawFd = self.fd();
        if fd == UNINIT_FD {
            return Err(io::Error::new(io::ErrorKind::NotFound, "no file descriptor"));
        }
        if fd < 0 {
            return Err(io::Error::new(io::ErrorKind::NotFound, "stale file descriptor"));
        }
        // `fd >= 0` was checked above, so `as_fd()` cannot panic here
        proc_fd_path(self)
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
        Some(self.cmp(other))
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
Owned, NUL-terminated entry name. Short names (the overwhelming majority)
live inline, longer ones on the heap; either way both the `&CStr` and the
`&[u8]` views are free - no `strlen` per call, unlike `dirent.d_name`.
*/
#[derive(Clone)]
enum EntryName {
    Inline { len: u8, buf: [u8; NAME_INLINE_CAP] },
    Heap(Box<[u8]>),
}

impl EntryName {
    fn new(name: &CStr) -> Self {
        let with_nul: &[u8] = name.to_bytes_with_nul();
        if with_nul.len() <= NAME_INLINE_CAP {
            let mut buf: [u8; NAME_INLINE_CAP] = [0; NAME_INLINE_CAP];
            buf[..with_nul.len()].copy_from_slice(with_nul);
            // `len` is the name length without the NUL (fits: cap - 1 < 256)
            Self::Inline { len: (with_nul.len() - 1) as u8, buf }
        } else {
            Self::Heap(with_nul.into())
        }
    }

    /// The name bytes including the trailing NUL.
    #[inline]
    fn with_nul(&self) -> &[u8] {
        match self {
            Self::Inline { len, buf } => &buf[..=*len as usize],
            Self::Heap(bytes) => bytes,
        }
    }

    #[inline]
    fn as_bytes(&self) -> &[u8] {
        let bytes: &[u8] = self.with_nul();
        &bytes[..bytes.len() - 1]
    }

    #[inline]
    fn as_cstr(&self) -> &CStr {
        /*
        SAFETY: `with_nul()` returns exactly the bytes of the `CStr` this
        name was built from (`new()` copies `to_bytes_with_nul()`), so the
        slice has no interior NUL and ends with one.
        */
        unsafe { CStr::from_bytes_with_nul_unchecked(self.with_nul()) }
    }
}

/**
A directory entry, aiming to be closely compatible with the
[std::fs::DirEntry] API. Built from a [nix::dir::Entry] but does **not**
keep one: the 280-byte `dirent` (256 of them the `d_name` array) is reduced
to an owned name, the inode and `d_type` at construction, which makes an
entry ~80 bytes instead of ~440 and keeps the lookahead buffer, `Vec`
sorts and the `Peekable` slot cheap to move through. `file_name()`,
`ino()` and `d_type()` cover what the old `Deref<Target = Entry>` exposed.

Notable differences to `std`:
- `metadata()` is replaced with `stat()`, and we return a [libc::stat] struct
- `file_type()` is replaced with a custom implementation, which uses `fstatat()`
  if the file type is not available in the `dirent` struct
- the stat result is cached in a `OnceLock` (boxed, to keep the entry
  compact) to avoid calling `fstatat()` multiple times for the same entry.

The `'h` lifetime ties each entry to the [DirHandle] that produced it via
a [BorrowedFd]: the parent handle's directory fd must remain open for as
long as the entry exists, which the borrow checker enforces. Collecting
entries into a `Vec` keeps the handle borrowed for the lifetime of the
vec, so use-after-close zombies are not constructible from safe code.
*/
#[derive(Clone)]
pub struct EntryExt<'h> {
    name: EntryName,
    ino: u64,
    /// `d_type` as reported by `readdir`; `None` when `DT_UNKNOWN`
    d_type: Option<Type>,
    dirfd: BorrowedFd<'h>,
    /// lazily cached `fstatat` result; `Some(None)` = stat failed
    stat: OnceLock<Option<Box<libc::stat>>>,
}

impl<'h> Eq for EntryExt<'h> {}

/// Hand-written so that the `name` and `stat` fields print usefully.
impl<'h> Debug for EntryExt<'h> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("EntryExt")
            .field("name", &self.file_name())
            .field("ino", &self.ino)
            .field("d_type", &self.d_type)
            .field("dirfd", &self.dirfd.as_raw_fd())
            // None: not statted yet; Some(false): stat() failed
            .field("statted", &self.stat.get().map(Option::is_some))
            .finish()
    }
}

impl<'h> EntryExt<'h> {
    /**
    Create an `EntryExt` from a [nix::dir::Entry] and its parent directory
    fd. Only the name, inode and `d_type` are copied out; the `Entry` is
    not retained (see the type docs).
    */
    #[instrument(level = "trace", skip(entry), fields(name = ?entry.file_name()))]
    pub fn new(entry: &Entry, dirfd: BorrowedFd<'h>) -> Self {
        Self {
            name: EntryName::new(entry.file_name()),
            ino: entry.ino(),
            d_type: entry.file_type(),
            dirfd,
            stat: OnceLock::new(),
        }
    }

    /// Create a new `EntryExt` and also `stat()` it before returning it.
    pub fn new_statted(entry: &Entry, dirfd: BorrowedFd<'h>) -> Self {
        let new_e: EntryExt<'h> = Self::new(entry, dirfd);
        new_e.stat();
        new_e
    }

    /// The entry name exactly as `readdir` returned it (no allocation).
    #[inline]
    pub fn file_name(&self) -> &CStr {
        self.name.as_cstr()
    }

    /// Inode number of the entry.
    #[inline]
    pub fn ino(&self) -> u64 {
        self.ino
    }

    /**
    File type as reported by `readdir` (`dirent.d_type`), **without** the
    `fstatat` fallback that `file_type()` applies. `None` means the
    filesystem reported `DT_UNKNOWN`.
    */
    #[inline]
    pub fn d_type(&self) -> Option<Type> {
        self.d_type
    }

    /**
    Return the [libc::stat] struct for the entry (may be cached).

    NOTE: If for some reason we cannot stat the entry, we return `None`.
    */
    pub fn stat(&self) -> Option<libc::stat> {
        self.stat
            .get_or_init(|| {
                fstatat(self.dirfd, self.file_name(), AtFlags::AT_SYMLINK_NOFOLLOW)
                    .ok()
                    .map(Box::new)
            })
            .as_deref()
            .copied()
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
        self.stat().map(|s: libc::stat| s.st_mode)
    }

    /// Whether the entry is a zero-length file (or stat() failed; see `len()`).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Owner uid of the entry, if it can be stat()ed.
    pub fn uid(&self) -> Option<libc::uid_t> {
        self.stat().map(|s: libc::stat| s.st_uid)
    }

    /// Owner gid of the entry, if it can be stat()ed.
    pub fn gid(&self) -> Option<libc::gid_t> {
        self.stat().map(|s: libc::stat| s.st_gid)
    }

    /// Number of hard links to the entry, if it can be stat()ed.
    // `nlink_t` is u64 on x86_64 but u32 on e.g. aarch64; `From` covers both,
    // and on the former clippy sees an identity conversion - by design.
    #[allow(clippy::useless_conversion)]
    pub fn nlink(&self) -> Option<u64> {
        self.stat().map(|s: libc::stat| u64::from(s.st_nlink))
    }

    /**
    Last modification time of the entry, if it can be stat()ed.

    NOTE: [TimeSinceEpoch] is `f64` seconds - roughly microsecond
    precision in the current era. Use `stat()` directly if you need the
    exact nanosecond timespec.
    */
    pub fn mtime(&self) -> Option<TimeSinceEpoch> {
        self.stat().map(|s: libc::stat| stat_time(s.st_mtime, s.st_mtime_nsec))
    }

    /// Last access time of the entry, if it can be stat()ed. See `mtime()`
    /// for the precision caveat.
    pub fn atime(&self) -> Option<TimeSinceEpoch> {
        self.stat().map(|s: libc::stat| stat_time(s.st_atime, s.st_atime_nsec))
    }

    /// Last status (inode) change time of the entry, if it can be stat()ed.
    /// See `mtime()` for the precision caveat.
    pub fn ctime(&self) -> Option<TimeSinceEpoch> {
        self.stat().map(|s: libc::stat| stat_time(s.st_ctime, s.st_ctime_nsec))
    }

    /// Open this entry as a [std::fs::File] with the given flags.
    /// `O_CLOEXEC` is always added so the fd does not leak across `exec`.
    fn open(&self, flags: OFlag) -> io::Result<File> {
        let open_how: OpenHow = OpenHow::new()
            .flags(flags | OFlag::O_CLOEXEC)
            .resolve(ResolveFlag::RESOLVE_BENEATH);
        let fd: OwnedFd = openat2(self.dirfd, self.file_name(), open_how)?;
        Ok(File::from(fd))
    }

    /// Open this entry for reading as a [std::fs::File] object.
    pub fn read(&self) -> io::Result<File> {
        self.open(OFlag::O_RDONLY)
    }

    /// Open this entry for read+write as a [std::fs::File] object.
    pub fn write(&self) -> io::Result<File> {
        self.open(OFlag::O_RDWR)
    }

    /**
    Open this entry as a new [DirHandle], if it is a directory.

    Like `open()`, this resolves via `openat2` with `RESOLVE_BENEATH`
    (plus `O_DIRECTORY`, `O_CLOEXEC` and `O_NONBLOCK`), so descending
    into a subdirectory needs neither procfs nor path re-resolution and
    is immune to rename races by construction - the natural primitive
    for recursive tree scans. Fails with `ENOTDIR` on non-directories.
    */
    pub fn open_dir(&self) -> io::Result<DirHandle> {
        let flags: OFlag =
            OFlag::O_RDONLY | OFlag::O_DIRECTORY | OFlag::O_CLOEXEC | OFlag::O_NONBLOCK;
        let open_how: OpenHow = OpenHow::new()
            .flags(flags)
            .resolve(ResolveFlag::RESOLVE_BENEATH);
        let fd: OwnedFd = openat2(self.dirfd, self.file_name(), open_how)?;
        DirHandle::from_fd(fd)
    }

    /// The entry name as raw bytes, without the trailing NUL (no allocation).
    #[inline]
    pub fn name_as_bytes(&self) -> &[u8] {
        self.name.as_bytes()
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
    Fails with `NotFound` if the parent directory has been deleted.
    */
    pub fn path(&self) -> io::Result<PathBuf> {
        let link: PathBuf = proc_fd_path(self.dirfd)?;
        Ok(link.join(OsStr::from_bytes(self.name_as_bytes())))
    }

    /// Return the file type of the entry as a [nix::dir::Type] enum,
    /// falling back to `stat()` when `d_type` is `DT_UNKNOWN`.
    pub fn file_type(&self) -> Option<Type> {
        if let Some(entry_type) = self.d_type {
            Some(entry_type)
        } else {
            match self.mode() {
                Some(mode) => EntryType(mode).entry_t(),
                None => None, // couldn't stat the entry
            }
        }
    }

    /**
    Return the file type of the entry as a `u8` typenum: the kernel's
    `DT_*` value (see the `TYPENUM_*` constants). This is part of the
    stable state digest, so it is an explicit mapping rather than nix's
    enum discriminant. Unknown is [TYPENUM_UNKNOWN] (`DT_UNKNOWN`).
    */
    #[rustfmt::skip]
    pub fn typenum(&self) -> u8 {
        match self.file_type() {
            Some(Type::Fifo)            => TYPENUM_FIFO,
            Some(Type::CharacterDevice) => TYPENUM_CHR,
            Some(Type::Directory)       => TYPENUM_DIR,
            Some(Type::BlockDevice)     => TYPENUM_BLK,
            Some(Type::File)            => TYPENUM_REG,
            Some(Type::Symlink)         => TYPENUM_LNK,
            Some(Type::Socket)          => TYPENUM_SOCK,
            None                        => TYPENUM_UNKNOWN,
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
NOTE: equality is defined over `(name, inode, parent dirfd)` - explicit
fields only. Historical note: this must never go back to comparing raw
`nix::dir::Entry`s: nix fills the dirent from `readdir_r` into a
`MaybeUninit` buffer and only `d_reclen` bytes get copied, while libc's
derived comparison reads the full 256-byte `d_name` array (plus
`d_off`/`d_reclen`) - i.e. uninitialized garbage. The same logical entry
read twice could compare unequal. Copying the fields out in `new()` is
what makes `EntryExt` immune to that.
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
    with [PartialEq] - `cmp() == Equal` if and only if `eq()`.
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

impl<'h> Hash for EntryExt<'h> {
    /**
    Hashes `(name, inode)` - a subset of the [PartialEq] fields, so the
    `Hash`/`Eq` contract holds. The dirfd is deliberately omitted: hashing
    it would invalidate hashes whenever the directory is reopened under a
    different fd.

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
        state.write_u64(self.ino);
        state.write_u8(self.typenum());
    }

    fn xxh3_digest(&self) -> u64 {
        let mut hasher: CustomXxh3Hasher = CustomXxh3Hasher::default();
        self.xxh3(&mut hasher);
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
        matches!(self, StateChange::Unchanged)
    }
}

/* --------------------------------- */

/// This struct holds the state of a directory for change detection.
#[derive(Clone, Default)]
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

    /**
    Construct a [DirHandle] from an already-open directory file descriptor
    (e.g. one returned by `openat` / `openat2`). Takes ownership: the fd
    is closed when the handle drops. Fails with `ENOTDIR` if the fd does
    not refer to a directory.
    */
    pub fn from_fd(fd: OwnedFd) -> io::Result<Self> {
        Ok(Self {
            inner: Dir::from_fd(fd)?,
            state: DirectoryState::default(),
        })
    }

    /// Our file descriptor as a [DirFd] object.
    pub fn fd(&self) -> DirFd {
        self.inner.as_raw_fd().into()
    }

    /**
    This relies on proc filesystem being available due to the use of
    `/proc/self/fd` to resolve the path by file descriptor. Fails with
    `NotFound` once the directory has been deleted (`rmdir`), even though
    the handle itself stays open and iterable (as empty).
    */
    pub fn path(&self) -> io::Result<PathBuf> {
        proc_fd_path(&self.inner)
    }

    /// Stored state of the directory as a [DirectoryState] object.
    pub fn state(&self) -> &DirectoryState {
        &self.state
    }

    /**
    Current state of the directory as a [DirectoryState] object.

    Does **not** touch the stored state - use `state_changed()` for that.
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
    Like `state_changed()`, but with a cheap timestamp pre-check.

    A directory's own mtime changes exactly when its entry list changes
    (add / remove / rename) - which is precisely what [DirectoryState]
    tracks. If both mtime and ctime of the directory are clearly older
    than the stored baseline, we report "unchanged" after a single
    `fstat` syscall instead of re-listing and re-hashing every entry.
    Timestamps within [MTIME_SLACK_SECS] of the baseline fall through
    to the full `state_changed()` comparison.

    The baseline `when` is the time the snapshot pass **started**, not
    when it finished: a change landing mid-pass may or may not have been
    seen by `readdir`, so it must compare as "not older" here and force
    the full check. Stamping the end of a pass longer than the slack
    would let such a change slip through - and since this pre-check never
    touches the stored state, it would keep slipping through on every
    later call until something else bumped the directory mtime.

    NOTE: backdating the directory mtime (`touch -d` / `utimensat`) does
    **not** defeat the pre-check, because those calls bump ctime, which
    we also consider (verified in the integration tests). Only direct
    clock manipulation or a filesystem with broken ctime semantics could
    produce a false "unchanged" here.
    */
    pub fn state_changed_fast(&mut self) -> io::Result<bool> {
        if let Some(when) = &self.state.when {
            let st: libc::stat = self.stat()?;
            let mtime: f64 = stat_time(st.st_mtime, st.st_mtime_nsec).get();
            let ctime: f64 = stat_time(st.st_ctime, st.st_ctime_nsec).get();
            if mtime.max(ctime) + MTIME_SLACK_SECS < when.get() {
                return Ok(false);
            }
        }
        self.state_changed()
    }

    /// `fstat()` the directory itself (not its entries).
    pub fn stat(&self) -> io::Result<libc::stat> {
        Ok(fstat(&self.inner)?)
    }

    /**
    Modification time of the directory itself. A directory's mtime
    changes when entries are added to, removed from or renamed within
    it (not when file contents change). See `EntryExt::mtime()` for
    the `f64` precision caveat.
    */
    pub fn mtime(&self) -> io::Result<TimeSinceEpoch> {
        let st: libc::stat = self.stat()?;
        Ok(stat_time(st.st_mtime, st.st_mtime_nsec))
    }

    /**
    Return the inner [nix::dir::Iter] object and make it `Peekable`.

    NOTE: this iterator will return the special `.` and `..` entries.

    ## Safety
    There is no memory-safety precondition here: the `&mut self` borrow
    already guarantees exclusive access to the underlying `DIR*` for the
    iterator's lifetime, and nix's `Iter` is `Send` for that reason. The
    `unsafe` marker is a deliberate speed bump - it forces callers to
    acknowledge that they are bypassing the `.`/`..` filtering, the error
    stickiness and the state tracking of the safe iterators, and that a
    `DIR*` stream must never be read from two places at once.
    */
    pub unsafe fn raw_iter<'handle>(&'handle mut self) -> Peekable<Iter<'handle>> {
        self.inner.iter().peekable()
    }

    /**
    Run a closure on each [nix::dir::Entry]. This allows lower-level but
    safe access to the inner [nix::dir::Iter] iterator.

    NOTE: the special `.` and `..` entries will be processed as well.

    NOTE: iteration stops at the first `readdir` error - skipping errors
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
        // everything gets partitioned (and usually sorted) afterwards, so
        // the dir-first lookahead of `iter()` would be wasted work here
        DirHandleIter::new(self, false).plain().for_each(|entry: EntryExt<'handle>| {
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
        DirHandleIterSorted(files)
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
rest of the fields are plain data), so no `unsafe impl` is needed - and
having one would silently mask a future non-Send field. This assertion
keeps the requirement checked at compile time: [OpenHandles] shares
handles across threads and needs `DirHandle: Send`.
*/
const _: () = {
    const fn assert_send<T: Send>() {}
    assert_send::<DirHandle>();
};

#[cfg(feature = "size_of")]
impl SizeOf for DirHandle {
    fn size_of_children(&self, context: &mut Context) {
        // the only heap child is glibc's directory stream; our own fields
        // (the `DIR*` and the DirectoryState) are inline and counted by
        // the caller via `size_of::<DirHandle>()`
        context.add(DIR_STREAM_HEAP).add_distinct_allocation();
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
    stays in sync because `is_dir()` is stable per entry - `d_type` is
    fixed and the stat fallback result is cached - and all mutation
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
    Pop the next entry, preferring directories. Since `push` is the only
    way in and it puts directories at the front and everything else at the
    back, the buffered directories always form a prefix of the queue - so
    "the first directory" is simply the front entry whenever `n_dirs > 0`,
    and no scan is needed. With no directories buffered, the oldest entry
    is popped instead.

    The returned flag says whether the popped entry is a directory, so the
    caller does not have to re-derive it via `is_dir()`.
    */
    pub fn try_pop_dir(&mut self) -> Option<(EntryExt<'h>, bool)> {
        let entry: EntryExt<'h> = self.q.pop_front()?;
        if self.n_dirs > 0 {
            debug_assert!(entry.is_dir(), "n_dirs > 0 but the front entry is not a directory");
            self.n_dirs -= 1;
            return Some((entry, true));
        }
        Some((entry, false))
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
    /**
    Plain mode: a straight pass over the stream without the dir-first
    lookahead buffer. Used by `entries()` / `directory_state()`, which
    partition or hash everything afterwards and gain nothing from the
    ordering heuristic. State tracking works exactly as in buffered mode.
    */
    plain: bool,
    /// running digest fold of dir entries - used for state hashing
    dirs: DigestFold,
    /// running digest fold of file entries - used for state hashing
    files: DigestFold,
    /**
    When this pass started (before the first `readdir`). Becomes the
    `when` of the finalized [DirectoryState] - see `state_changed_fast()`
    for why the start, and not the end, of the pass is the right stamp.
    */
    started: TimeSinceEpoch,
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
            plain: false,
            dirs: DigestFold::default(),
            files: DigestFold::default(),
            started: TimeSinceEpoch::new(),
        }
    }

    /// Switch this iterator to plain mode (see the `plain` field).
    fn plain(mut self) -> Self {
        self.plain = true;
        self
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
        // `Result<Entry, _>` is `Copy`, so matching on the peeked value by
        // value would memcpy the 280-byte dirent out of the slot each time;
        // `as_ref()` keeps it a borrow
        match self.inner.peek().map(Result::as_ref) {
            Some(Ok(e)) => {
                /*
                `.` and `..` are both directories, but `get_one()` filters
                them out - treating them as "next dir" here would wrongly
                defer the current entry only to have the dot skipped.
                */
                if matches!(e.file_name().to_bytes(), DOT1 | DOT2) {
                    return Some(false);
                }
                e.file_type().map(|t: Type| t == Type::Directory)
            }
            _ => Some(false),
        }
    }

    /**
    The inner iterator is exhausted or has hit a sticky error: finalize
    the [DirectoryState] if this pass was clean, complete and asked for
    it. Safe to call repeatedly; only the first call after a clean pass
    does any work.
    */
    fn finish(&mut self) {
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
            return;
        }
        if self.update {
            self.update = false;
            self.state.dirs = self.dirs.len();
            self.state.files = self.files.len();
            self.state.hash_d = self.dirs.finish();
            self.state.hash_f = self.files.finish();
            self.state.when = self.started.clone().into();
            trace!(target: "DirHandle.state", "{:?}", self.state);
        }
    }

    /// Get one entry from the inner iterator.
    fn get_one(&mut self) -> Option<EntryExt<'handle>> {
        let entry: EntryExt<'handle> = next(&mut self.inner, self.dirfd, self.stat)?;
        if self.update {
            // fold the entry's stable digest into the running state hash
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
        if self.plain {
            // straight pass: no lookahead, no reordering
            return match self.get_one() {
                Some(entry) => Some(entry),
                None => {
                    self.finish();
                    None
                }
            };
        }
        loop {
            if let Some((entry, is_dir)) = self.buf.try_pop_dir() {
                if is_dir {
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
                self.finish();
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

The `'handle` lifetime comes from the entries themselves (each holds a
`BorrowedFd<'handle>` of the parent [DirHandle]), so the vec cannot
outlive the handle - no extra marker needed.
*/
#[derive(Debug)]
pub struct DirHandleIterSorted<'handle>(EntryVec<'handle>);

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

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
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
        Some(self.wrap(self.0.get_mut(&fd)?))
    }

    /// Wrap an already write-locked map entry into a [CheckedOutHandle].
    fn wrap<'a>(&'a self, ref_handle: RefMut<'a, RawFd, DirHandle>) -> CheckedOutHandle<'a> {
        CheckedOutHandle {
            inner: ref_handle,
            pool: self,
            _not_send: PhantomData,
        }
    }

    /**
    Open a directory, insert its handle into the map and return it.

    **NOTE**: may deadlock if called while holding any kind of reference
    into this [OpenHandles] in the same thread.
    */
    pub fn open(&'_ self, path: &Path) -> io::Result<CheckedOutHandle<'_>> {
        let handle: DirHandle = DirHandle::new(path)?;
        let fd: RawFd = handle.as_raw_fd();
        /*
        `entry().insert()` inserts the handle and hands back the write-locked
        `RefMut` in one step under the shard lock, so there is no window in
        which another thread could `close(fd)` our freshly opened handle
        between the insert and the checkout (which the previous
        insert-then-get_mut sequence had to report as an io::Error).
        */
        Ok(self.wrap(self.0.entry(fd).insert(handle)))
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

    /// Remove a handle from the map and hand it back to the caller (the
    /// directory stays open - dropping the returned handle closes it).
    pub fn remove(&self, fd: RawFd) -> Option<DirHandle> {
        self.0.remove(&fd).map(|(_, handle)| handle)
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
!Sync (nix::dir::Dir is deliberately !Sync - readdir on a shared DIR*
races). Sharing &OpenHandles is still sound because DashMap's per-shard
RwLock makes &mut DirHandle access (checkout / for_each_mut) exclusive,
and concurrent shared access (for_each / iter) only reaches `&self`
methods of DirHandle - fd(), path(), state(), as_raw_fd(), Hash, Eq -
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
            /*
            DashMap keeps `(key, value)` pairs inline in its hashbrown
            tables, so the slot size is that tuple's size. Each handle's
            heap children (glibc's DIR stream) are added by the recursion
            below and must not be folded into the slot size as well - that
            double counted every handle in earlier versions.
            */
            let slot: usize = std::mem::size_of::<(RawFd, DirHandle)>();
            let used: usize = slot * self.0.len();
            let total: usize = slot * self.0.capacity();
            context
                .add(used)
                .add_excess(total - used)
                .add_distinct_allocation();

            self.0.iter().for_each(|itm| {
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

A checked-out handle is an active DashMap shard lock and must stay on
the thread that took it, so the type is deliberately `!Send`:

```compile_fail
use dirhandle::OpenHandles;
fn assert_send<T: Send>(_: &T) {}
let pool = OpenHandles::new();
let h = pool.open(std::path::Path::new("/")).unwrap();
assert_send(&h); // error: `Rc<()>` cannot be sent between threads safely
```
*/
pub struct CheckedOutHandle<'a> {
    inner: RefMut<'a, RawFd, DirHandle>,
    /// the pool this handle was checked out from - `close()` removes it there
    pool: &'a OpenHandles,
    /*
    Marker that makes the type `!Send` (and `!Sync`). Earlier versions got
    the same effect from an `Rc<dyn Fn(RawFd)>` close callback, which cost
    a heap allocation and a vtable call per checkout; a zero-sized `Rc`
    phantom keeps the property for free. Do not "fix" this to `Arc` or
    remove it - moving a live lock guard across threads violates DashMap's
    locking model (see doc/design/open-handles.md).
    */
    _not_send: PhantomData<Rc<()>>,
}

impl<'a> CheckedOutHandle<'a> {
    /**
    Close this [DirHandle] and release its file descriptor.

    NOTE: the internal lock must be released before removing the entry
    (see the deadlock caveat on [OpenHandles]), which opens a tiny
    window where another thread may close this fd and the kernel may
    recycle the number for a freshly opened handle - in that case the
    new entry gets evicted instead. Same fd-reuse caveat as documented
    on `OpenHandles::open`.
    */
    pub fn close(self) {
        let fd: RawFd = *self.inner.key();
        let pool: &'a OpenHandles = self.pool;
        drop(self.inner); // explicitly release the RefMut before touching the map
        pool.close(fd);
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
Resolve an open directory file descriptor to its path via
`/proc/self/fd/<fd>`.

NOTE: `/proc/self/fd` is itself a directory (its *entries* are symlinks),
so probing procfs availability must not `readlink()` the directory - that
fails with EINVAL even when procfs is mounted. We attempt the per-fd
resolution directly and only diagnose a missing procfs after a failure.

A directory that has been deleted while we hold it open still resolves
in procfs, to `"<old path> (deleted)"` - a path that does not exist and
that a caller would happily join entry names onto. We detect that case
by link count instead of by string: `rmdir` drops the inode's `st_nlink`
to 0, which `fstat` on the open fd reports, whereas a directory that is
merely *named* `"foo (deleted)"` keeps its links. Deleted directories
therefore fail with `NotFound`.
*/
fn proc_fd_path<Fd: AsFd>(fd: Fd) -> io::Result<PathBuf> {
    let raw: RawFd = fd.as_fd().as_raw_fd();
    let link: PathBuf = read_link(format!("{}/{}", PROC_FD_PATH, raw)).map_err(|e| {
        if e.kind() == io::ErrorKind::NotFound && !Path::new(PROC_FD_PATH).is_dir() {
            io::Error::new(io::ErrorKind::Unsupported, "procfs not available")
        } else {
            e
        }
    })?;
    if fstat(&fd)?.st_nlink == 0 {
        return Err(io::Error::new(io::ErrorKind::NotFound, "directory has been deleted"));
    }
    Ok(link)
}

/**
Convert a `(secs, nsecs)` timestamp pair from a [libc::stat] into a
[TimeSinceEpoch]. Inherently lossy: `f64` seconds carry roughly
microsecond precision in the current era, not nanoseconds.
*/
fn stat_time(secs: i64, nsecs: i64) -> TimeSinceEpoch {
    TimeSinceEpoch::new_from(secs as f64 + nsecs as f64 * 1e-9)
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
    OpenOptions::new().read(true).open(path)
}

/**
Order-independent accumulator of per-entry xxh3 digests - one per entry
class (dirs / files) of a [DirectoryState]. The per-entry digests cover
`(name, inode, typenum)`, so the same set of entries always folds to the
same combined digest regardless of `readdir` order.

Two commutative folds (wrapping sum and xor) plus the count are mixed
through xxh3 in `finish()`. Either fold alone would let trivially
constructed multisets collide (`{a, b}` vs `{c, d}` with `a + b == c + d`);
the pair requires simultaneous sum *and* xor equality. This is a
change-detection fingerprint, not a cryptographic commitment.

Replaces the v0.4 scheme (collect every digest, sort, hash the sequence):
O(1) memory and O(n) time per pass instead of an 8-bytes-per-entry `Vec`
and an `n log n` sort. Digests are therefore not comparable across the
0.4 / 0.5 boundary.

This is the single source of truth for [DirectoryState] hashing - both
the lazy in-iterator computation and `directory_state()` go through it,
which keeps the two paths comparable.
*/
#[derive(Debug, Default)]
struct DigestFold {
    sum: u64,
    xor: u64,
    n: usize,
}

impl DigestFold {
    #[inline]
    fn push(&mut self, digest: u64) {
        self.sum = self.sum.wrapping_add(digest);
        self.xor ^= digest;
        self.n += 1;
    }

    /// Number of digests folded in so far.
    #[inline]
    fn len(&self) -> usize {
        self.n
    }

    /// The combined digest.
    fn finish(&self) -> u64 {
        let mut xxh: CustomXxh3Hasher = CustomXxh3Hasher::default();
        xxh.write_u64(self.sum);
        xxh.write_u64(self.xor);
        xxh.write_u64(self.n as u64);
        xxh.finish()
    }
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
    // stamped before the first readdir - see `state_changed_fast()`
    let started: TimeSinceEpoch = TimeSinceEpoch::new();
    let mut d_fold: DigestFold = DigestFold::default();
    let mut f_fold: DigestFold = DigestFold::default();
    let mut iter: DirHandleIter = DirHandleIter::with_update(dir, false, false).plain();
    for entry in iter.by_ref() {
        match entry.file_type() {
            Some(Type::Directory) => d_fold.push(entry.xxh3_digest()),
            _ => f_fold.push(entry.xxh3_digest()),
        }
    }
    if let Some(errno) = iter.error() {
        return Err(io::Error::from_raw_os_error(errno as i32));
    }
    Ok(DirectoryState {
        dirs: d_fold.len(),
        files: f_fold.len(),
        hash_d: d_fold.finish(),
        hash_f: f_fold.finish(),
        when: started.into(),
    })
}

/**
Return the next entry from the inner [nix::dir::Iter] as an [EntryExt],
skipping `.` and `..`.

Entries whose file type cannot be determined (`d_type` is `DT_UNKNOWN`
and the `fstatat` fallback fails, e.g. due to permission denied) are
yielded too - their `file_type()` returns `None` and the caller decides
what to do. Silently dropping them would make a listable-but-unsearchable
directory iterate as empty on filesystems that don't populate `d_type`.

A `readdir` error ends the iteration: a persistent error (e.g. `ESTALE`
on NFS, `EIO`) would otherwise be skipped forever and spin this loop. The
error is deliberately left **unconsumed** in the [Peekable] slot, so it is
sticky - callers can observe it via `peek()` and repeated calls return
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

        /*
        Sadly it appears that we cannot rely on the special "." and ".."
        entries being returned first by the libc `readdir` call, so to
        filter them out we must match each name.
        */
        if matches!(entry.file_name().to_bytes(), DOT1 | DOT2) {
            continue;
        }

        // convert the [nix::dir::Entry] to our `EntryExt`
        let entry: EntryExt<'h> = match stat {
            false => EntryExt::new(&entry, dirfd),
            true => EntryExt::new_statted(&entry, dirfd),
        };
        trace!(target: "name", "{:?} : {:?}", entry.name(), entry);
        return Some(entry);
    }
}

/* ################################# TESTS ################################# */

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;

    fn siphash<T: Hash>(item: &T) -> u64 {
        let mut hasher: DefaultHasher = DefaultHasher::new();
        item.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn dirfd_state_machine() {
        let fd = DirFd::default();
        assert!(!fd.is_open());
        assert_eq!(fd.fd(), UNINIT_FD);
        assert!(fd.path().is_err(), "uninit fd must not resolve");

        assert_eq!(fd.set(5), Ok(5));
        assert!(fd.is_open());
        assert_eq!(fd.set(7), Err(5), "set() must refuse to overwrite an open fd");

        fd.clear();
        assert!(!fd.is_open());
        assert_eq!(fd.fd(), !5, "stale encoding is bitwise NOT of the old fd");
        assert!(fd.path().is_err(), "stale fd must not resolve");

        fd.clear();
        assert_eq!(fd.fd(), UNINIT_FD, "second clear() buries the stale fd");

        // fd 0 is a valid open fd and its stale marker (-1) stays distinct
        // from the uninitialized sentinel
        let zero = DirFd::from(0);
        assert!(zero.is_open());
        zero.clear();
        assert_eq!(zero.fd(), -1);
        assert!(!zero.is_open());
        assert_eq!(zero.set(3), Ok(3), "stale fd may be overwritten");
    }

    #[test]
    fn dirfd_as_fd_panics_when_not_open() {
        let fd = DirFd::default();
        let res = std::panic::catch_unwind(|| {
            let _ = fd.as_fd();
        });
        assert!(res.is_err(), "as_fd on a closed DirFd must panic, not be UB");
    }

    #[test]
    #[rustfmt::skip]
    fn entry_type_classification() {
        assert!(EntryType(libc::S_IFDIR | 0o755).is_dir());
        assert!(EntryType(libc::S_IFREG | 0o644).is_file());
        assert!(EntryType(libc::S_IFLNK | 0o777).is_symlink());
        assert_eq!(EntryType(libc::S_IFDIR).entry_t(),  Some(Type::Directory));
        assert_eq!(EntryType(libc::S_IFREG).entry_t(),  Some(Type::File));
        assert_eq!(EntryType(libc::S_IFLNK).entry_t(),  Some(Type::Symlink));
        assert_eq!(EntryType(libc::S_IFIFO).entry_t(),  Some(Type::Fifo));
        assert_eq!(EntryType(libc::S_IFSOCK).entry_t(), Some(Type::Socket));
        assert_eq!(EntryType(libc::S_IFBLK).entry_t(),  Some(Type::BlockDevice));
        assert_eq!(EntryType(libc::S_IFCHR).entry_t(),  Some(Type::CharacterDevice));
        assert_eq!(EntryType(0).entry_t(), None, "DT_UNKNOWN-ish mode has no type");
    }

    #[test]
    #[rustfmt::skip]
    fn typenums_are_pinned() {
        // the digest scheme depends on these exact numbers (the kernel's
        // d_type ABI) - they must never drift with nix's enum order
        assert_eq!(TYPENUM_UNKNOWN, 0);
        assert_eq!(TYPENUM_FIFO,    1);
        assert_eq!(TYPENUM_CHR,     2);
        assert_eq!(TYPENUM_DIR,     4);
        assert_eq!(TYPENUM_BLK,     6);
        assert_eq!(TYPENUM_REG,     8);
        assert_eq!(TYPENUM_LNK,    10);
        assert_eq!(TYPENUM_SOCK,   12);
    }

    #[test]
    fn file_type_falls_back_to_stat_without_d_type() {
        /*
        Filesystems that report DT_UNKNOWN (XFS without ftype, some NFS)
        are not available here, so simulate one: rebuild each entry with
        `d_type: None` and an empty stat cache, and check that the
        `fstatat` fallback resolves the same type as `d_type` did.
        */
        let dir: PathBuf = std::env::temp_dir()
            .join(format!("dirhandle-unit-{}-dtype", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("plain"), b"x").unwrap();
        std::fs::create_dir(dir.join("sub")).unwrap();
        std::os::unix::fs::symlink("plain", dir.join("link")).unwrap();

        let mut h = DirHandle::new(&dir).unwrap();
        let mut seen: usize = 0;
        for e in h.iter() {
            let via_dtype: Option<Type> = e.file_type();
            assert!(via_dtype.is_some(), "tmpfs/ext4 must report d_type for {e:?}");
            let untyped = EntryExt {
                d_type: None,
                stat: OnceLock::new(),
                ..e.clone()
            };
            assert!(!untyped.is_statted());
            assert_eq!(untyped.d_type(), None);
            assert_eq!(untyped.file_type(), via_dtype, "fallback must agree for {e:?}");
            assert!(untyped.is_statted(), "fallback must go through stat()");
            assert_eq!(untyped.typenum(), e.typenum());
            assert_eq!(untyped.is_dir(), e.is_dir());
            assert_eq!(untyped.is_symlink(), e.is_symlink());
            seen += 1;
        }
        assert_eq!(seen, 3);

        // d_type unknown AND stat failing: yielded as "type unknown", counted
        // as a non-directory, typenum = DT_UNKNOWN
        let dirfd: BorrowedFd = h.inner.as_fd();
        let ghost = EntryExt {
            name: EntryName::new(c"does-not-exist"),
            ino: 0,
            d_type: None,
            dirfd,
            stat: OnceLock::new(),
        };
        assert_eq!(ghost.file_type(), None);
        assert_eq!(ghost.typenum(), TYPENUM_UNKNOWN);
        assert!(!ghost.is_dir());
        assert_eq!(ghost.len(), 0);
        assert!(ghost.is_statted(), "the failed stat must be cached too");

        drop(h);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn entry_ext_is_compact() {
        // the whole point of v0.5.0's EntryExt: ~80 bytes, not ~440
        assert_eq!(std::mem::size_of::<EntryName>(), 48);
        assert!(std::mem::size_of::<EntryExt>() <= 80, "{}", std::mem::size_of::<EntryExt>());
    }

    #[test]
    fn entry_name_inline_and_heap() {
        let short: &CStr = c"short.txt";
        let exact: &CStr = c"12345678901234567890123456789012345678"; // 38 = cap - 1
        let long: &CStr = c"123456789012345678901234567890123456789"; // 39 -> heap
        for (cs, inline) in [(short, true), (exact, true), (long, false)] {
            let n: EntryName = EntryName::new(cs);
            assert_eq!(matches!(n, EntryName::Inline { .. }), inline, "{cs:?}");
            assert_eq!(n.as_cstr(), cs);
            assert_eq!(n.as_bytes(), cs.to_bytes());
            assert_eq!(n.with_nul(), cs.to_bytes_with_nul());
        }
        let empty: EntryName = EntryName::new(c"");
        assert_eq!(empty.as_bytes(), b"");
        assert_eq!(empty.as_cstr(), c"");
    }

    #[test]
    fn state_changed_fast_short_circuits_on_old_mtime() {
        let dir: PathBuf = std::env::temp_dir()
            .join(format!("dirhandle-unit-{}-fast-path", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("f1"), b"1").unwrap();

        let mut h = DirHandle::new(&dir).unwrap();
        assert!(!h.state_changed_fast().unwrap(), "baseline");

        // a modification that is real, but whose dir mtime/ctime are
        // "clearly older" than the (forged) baseline, must be answered by
        // the fstat pre-check alone - i.e. the early-return branch
        std::fs::write(dir.join("f2"), b"2").unwrap();
        let far_future: f64 = TimeSinceEpoch::new().get() + 1e6;
        h.state.when = Some(TimeSinceEpoch::new_from(far_future));
        assert!(!h.state_changed_fast().unwrap(), "pre-check must short-circuit");
        assert_eq!(h.state.files, 1, "short-circuit must not touch the stored state");

        // and with an honest baseline the full path sees the change
        h.state.when = Some(TimeSinceEpoch::new_from(0.0));
        assert!(h.state_changed_fast().unwrap(), "full comparison must run");
        assert_eq!(h.state.files, 2);

        drop(h);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn digest_fold_is_order_independent() {
        let fold = |ds: &[u64]| -> u64 {
            let mut f: DigestFold = DigestFold::default();
            ds.iter().for_each(|d: &u64| f.push(*d));
            f.finish()
        };
        let a: u64 = fold(&[1, 2, 3]);
        assert_eq!(a, fold(&[3, 1, 2]), "same set must digest equal regardless of order");
        assert_ne!(a, fold(&[1, 2]), "different sets must differ");
        assert_eq!(fold(&[]), fold(&[]));
        assert_ne!(fold(&[]), a);
        // a sum-only fold would collide here (1 + 4 == 2 + 3); xor differs
        assert_ne!(fold(&[1, 4]), fold(&[2, 3]));
        // sum and xor both 0, only the count tells these apart
        assert_ne!(fold(&[0]), fold(&[]));
        assert_ne!(fold(&[7, 7]), fold(&[]));
    }

    #[test]
    fn stat_time_conversion() {
        let t: f64 = stat_time(1_700_000_000, 500_000_000).get();
        assert!((t - 1_700_000_000.5).abs() < 1e-3, "got {t}");
        assert_eq!(stat_time(0, 0).get(), 0.0);
    }

    #[test]
    fn state_change_precedence_and_deltas() {
        let base = DirectoryState {
            dirs: 2,
            files: 10,
            hash_d: 0x1111,
            hash_f: 0x2222,
            when: None,
        };
        assert!(base.change(&base.clone()).is_same());

        // count changes win over hash changes, deltas are signed i64
        let mut more = base.clone();
        more.dirs = 5;
        more.hash_d = 0x9999;
        match base.change(&more) {
            StateChange::DirNum(d) => assert_eq!(d, 3),
            c => panic!("expected DirNum(3), got {c:?}"),
        }

        let mut fewer = base.clone();
        fewer.files = 4;
        match base.change(&fewer) {
            StateChange::FileNum(d) => assert_eq!(d, -6),
            c => panic!("expected FileNum(-6), got {c:?}"),
        }

        // same counts, different hash
        let mut hashed = base.clone();
        hashed.hash_f = 0xdead;
        assert!(matches!(base.change(&hashed), StateChange::FileHash));
        let mut hashed_d = base.clone();
        hashed_d.hash_d = 0xbeef;
        assert!(matches!(base.change(&hashed_d), StateChange::DirHash));
    }

    #[test]
    fn directory_state_eq_and_hash_ignore_when() {
        let a = DirectoryState {
            dirs: 1,
            files: 2,
            hash_d: 3,
            hash_f: 4,
            when: None,
        };
        let mut b = a.clone();
        b.when = Some(TimeSinceEpoch::new());
        assert_eq!(a, b, "Eq must ignore `when`");
        assert_eq!(siphash(&a), siphash(&b), "Hash must agree with Eq");
        assert_eq!(a.hash_all(), b.hash_all());
    }
}
