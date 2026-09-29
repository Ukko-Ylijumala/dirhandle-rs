// Copyright (c) 2026 Mikko Tanner. All rights reserved.
//
// Integration tests for the public dirhandle API. Each test works in its
// own unique temp directory (tests run in parallel), cleaned up on drop.

use dirhandle::{DirHandle, DirectoryState, EntryExt, OpenHandles, StateChange};
use std::collections::hash_map::DefaultHasher;
use std::ffi::{CString, OsStr};
use std::fs;
use std::hash::{Hash, Hasher};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

/* ############################### HELPERS ################################# */

/// Self-cleaning unique temp directory for one test.
struct TestDir(PathBuf);

impl TestDir {
    fn new(tag: &str) -> Self {
        let dir: PathBuf =
            std::env::temp_dir().join(format!("dirhandle-test-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        TestDir(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }

    fn file(&self, name: &str, content: &[u8]) -> PathBuf {
        let p: PathBuf = self.0.join(name);
        fs::write(&p, content).unwrap();
        p
    }

    fn subdir(&self, name: &str) -> PathBuf {
        let p: PathBuf = self.0.join(name);
        fs::create_dir(&p).unwrap();
        p
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn siphash<T: Hash>(item: &T) -> u64 {
    let mut hasher: DefaultHasher = DefaultHasher::new();
    item.hash(&mut hasher);
    hasher.finish()
}

fn has_cloexec(fd: i32) -> bool {
    let flags: i32 = unsafe { libc::fcntl(fd, libc::F_GETFD) };
    assert!(flags >= 0, "F_GETFD failed");
    flags & libc::FD_CLOEXEC != 0
}

/// Whether the raw fd is open *and* refers to `path` - robust against the
/// number having been reused by a parallel test after a close.
fn fd_refers_to(fd: i32, path: &Path) -> bool {
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    if unsafe { libc::fstat(fd, &mut st) } != 0 {
        return false;
    }
    let md: fs::Metadata = fs::metadata(path).unwrap();
    st.st_dev == md.dev() && st.st_ino == md.ino()
}

/* ############################ PATH RESOLUTION ############################ */

#[test]
fn path_resolution_via_procfs() {
    let td = TestDir::new("path-resolution");
    td.file("plain.txt", b"hi");

    let mut h = DirHandle::new(td.path()).unwrap();
    let p: PathBuf = h.path().expect("path() must work with procfs mounted");
    assert_eq!(p, td.path().canonicalize().unwrap());

    for e in h.iter() {
        let ep: PathBuf = e.path().expect("entry path() must resolve");
        assert!(ep.symlink_metadata().is_ok(), "bad entry path {ep:?}");
    }
}

#[test]
fn non_utf8_names_round_trip() {
    // try filesystems in order; some (e.g. ext4 casefold dirs, ZFS
    // utf8only) reject non-UTF-8 names with EILSEQ
    let raw: &OsStr = OsStr::from_bytes(b"bad\xff\xfename");
    let mut chosen: Option<PathBuf> = None;
    for base in [std::env::temp_dir(), PathBuf::from("/dev/shm")] {
        let dir: PathBuf = base.join(format!("dirhandle-test-{}-nonutf8", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        if fs::create_dir_all(&dir).is_err() {
            continue;
        }
        if fs::write(dir.join(raw), b"raw").is_ok() {
            chosen = Some(dir);
            break;
        }
        let _ = fs::remove_dir_all(&dir);
    }
    let Some(dir) = chosen else {
        eprintln!("SKIP: no available filesystem accepts non-UTF-8 names");
        return;
    };

    let mut h = DirHandle::new(&dir).unwrap();
    let mut seen: bool = false;
    for e in h.iter() {
        if e.name_as_bytes() == raw.as_bytes() {
            let ep: PathBuf = e.path().unwrap();
            assert_eq!(ep.file_name().unwrap().as_bytes(), raw.as_bytes());
            assert!(ep.symlink_metadata().is_ok(), "resolved path must exist");
            seen = true;
        }
    }
    assert!(seen, "non-UTF-8 entry must be yielded");
    drop(h);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn path_of_deleted_directory_is_not_found() {
    let td = TestDir::new("deleted-path");
    let sub: PathBuf = td.subdir("gone");
    let mut h = DirHandle::new(&sub).unwrap();
    assert_eq!(h.path().unwrap(), sub.canonicalize().unwrap());

    fs::remove_dir(&sub).unwrap();
    // the handle stays open and iterates as empty ...
    assert_eq!(h.iter().count(), 0);
    // ... but procfs would now resolve it to "<path> (deleted)", which must
    // not leak out as a path; both resolution entry points agree
    for res in [h.path(), h.fd().path()] {
        let err = res.expect_err("deleted directory must not resolve to a path");
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound, "{err}");
        assert!(!err.to_string().contains("(deleted)"), "procfs artefact leaked: {err}");
    }
}

/* ############################## OPEN FLAGS ############################### */

#[test]
fn fds_are_cloexec() {
    let td = TestDir::new("cloexec");
    td.file("f.txt", b"x");

    let mut h = DirHandle::new(td.path()).unwrap();
    assert!(has_cloexec(h.as_raw_fd()), "directory fd must be CLOEXEC");

    for e in h.iter() {
        if e.is_file() {
            let f = e.read().unwrap();
            assert!(has_cloexec(f.as_raw_fd()), "entry file fd must be CLOEXEC");
        }
    }
}

#[test]
fn fifo_is_rejected_fast() {
    let td = TestDir::new("fifo");
    let fifo: PathBuf = td.path().join("pipe");
    let c_path: CString = CString::new(fifo.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(c_path.as_ptr(), 0o644) }, 0);

    // without O_DIRECTORY this would block waiting for a writer
    let t0 = std::time::Instant::now();
    let res = DirHandle::new(&fifo);
    assert!(res.is_err(), "opening a FIFO as a directory must fail");
    assert!(t0.elapsed().as_secs() < 1, "FIFO open must not block");
}

/* ############################## ITERATION ################################ */

#[test]
fn iteration_basics() {
    let td = TestDir::new("iteration");
    td.file("b.txt", b"b");
    td.file("a.txt", b"a");
    td.subdir("zdir");
    td.subdir("adir");

    let mut h = DirHandle::new(td.path()).unwrap();

    // dots are filtered, everything else is yielded
    let names: Vec<String> = h.iter().map(|e| e.name()).collect();
    assert_eq!(names.len(), 4);
    assert!(!names.iter().any(|n| n == "." || n == ".."));

    // repeated iteration works (rewind), same entries each pass
    let pass2: Vec<String> = h.iter().map(|e| e.name()).collect();
    assert_eq!(names, pass2);

    // early drop must not leave the stream mid-way for the next pass
    {
        let mut it = h.iter();
        let _ = it.next();
        let _ = it.next();
    } // dropped early
    assert_eq!(h.iter().count(), 4, "full count after early drop");

    // iter_sorted: dirs first (alphabetical), then files (alphabetical)
    let sorted: Vec<String> = h.iter_sorted().map(|e| e.name()).collect();
    assert_eq!(sorted, ["adir", "zdir", "a.txt", "b.txt"]);

    // entries() filtering
    let (d, f) = h.entries(true, true);
    assert_eq!((d.len(), f.len()), (2, 2));
    let (d, f) = h.entries(true, false);
    assert_eq!((d.len(), f.len()), (2, 0));
    let (d, f) = h.entries(false, true);
    assert_eq!((d.len(), f.len()), (0, 2));

    // for_each operates on the raw stream, including `.` and `..`
    let mut raw_count: usize = 0;
    h.for_each(|_| raw_count += 1);
    assert_eq!(raw_count, 6, "4 entries + . + ..");
}

#[test]
fn readdir_error_ends_pass_without_finalizing() {
    /*
    Regression: nix 0.30 drove readdir_r(3), which reports errors through
    its return value, but nix only checked for -1 - so every readdir error
    looked like a clean end of stream, and the partial (here: empty)
    listing was finalized as the directory's state. Provoke a real
    getdents64 failure (ENOTDIR) by swapping a regular file in under the
    open DIR* stream.
    */
    let td = TestDir::new("readdir-error");
    let file: PathBuf = td.file("f.txt", b"x");
    td.subdir("sub");

    let mut h = DirHandle::new(td.path()).unwrap();
    let f = fs::File::open(&file).unwrap();
    assert!(unsafe { libc::dup2(f.as_raw_fd(), h.as_raw_fd()) } >= 0, "dup2 failed");

    let mut it = h.iter();
    assert_eq!(it.by_ref().count(), 0);
    assert_eq!(it.error().map(|e| e as i32), Some(libc::ENOTDIR), "error must surface");
    drop(it);
    assert_eq!(h.state(), &DirectoryState::default(), "a failed pass must not finalize");

    let err = h.state_current().expect_err("a partial listing must not pass as state");
    assert_eq!(err.raw_os_error(), Some(libc::ENOTDIR), "{err}");
    assert!(h.state_changed().is_err());
    assert_eq!(h.state(), &DirectoryState::default(), "baseline must stay untouched");
}

/* ############################ ENTRY IDENTITY ############################# */

#[test]
fn entry_identity_and_hashing() {
    let td = TestDir::new("identity");
    td.file("one.txt", b"1");
    td.file("two.txt", b"2");
    td.subdir("sub");

    // two independent handles to the same directory
    let mut ha = DirHandle::new(td.path()).unwrap();
    let mut hb = DirHandle::new(td.path()).unwrap();
    let mut va: Vec<EntryExt> = ha.iter().collect();
    let mut vb: Vec<EntryExt> = hb.iter().collect();
    va.sort();
    vb.sort();
    assert_eq!(va.len(), vb.len());

    for (a, b) in va.iter().zip(vb.iter()) {
        assert_eq!(a.name_as_bytes(), b.name_as_bytes());
        // std Hash covers (name, ino) and excludes the dirfd, so the same
        // entry hashes identically across handles and across reads — and
        // never depends on uninitialized dirent bytes
        assert_eq!(siphash(a), siphash(b), "hash must be stable across handles");
        // Eq includes the parent dirfd: same file via different handles
        // is deliberately not eq
        assert_ne!(a, b, "different dirfd implies not equal");
        // Ord is consistent with Eq (tie-breaks on ino + dirfd)
        assert_ne!(a.cmp(b), std::cmp::Ordering::Equal);
        // identity within one handle
        assert_eq!(a, &a.clone());
        assert_eq!(a.cmp(&a.clone()), std::cmp::Ordering::Equal);
        assert_eq!(siphash(a), siphash(&a.clone()));
    }
}

/* ############################ STATE TRACKING ############################# */

#[test]
fn state_tracking_lifecycle() {
    let td = TestDir::new("state");
    td.file("f1.txt", b"1");
    td.subdir("d1");

    let mut h = DirHandle::new(td.path()).unwrap();

    // state_current() must not touch the stored state
    let current: DirectoryState = h.state_current().unwrap();
    assert_ne!(h.state(), &current, "stored state must remain uninitialized");
    assert_eq!(h.state(), &DirectoryState::default());

    // the first complete pass populates the stored state lazily, and the
    // in-iterator hashing path must agree with directory_state()
    h.iter().for_each(|_| {});
    assert_eq!(h.state(), &current, "both hashing paths must agree");

    // baseline + change detection
    let mut h2 = DirHandle::new(td.path()).unwrap();
    assert!(!h2.state_changed().unwrap(), "first call establishes baseline");
    assert!(!h2.state_changed().unwrap(), "no change yet");

    let before: DirectoryState = h2.state().clone();
    td.file("f2.txt", b"2");
    match before.change(&h2.state_current().unwrap()) {
        StateChange::FileNum(d) => assert_eq!(d, 1),
        c => panic!("expected FileNum(1), got {c:?}"),
    }
    assert!(h2.state_changed().unwrap(), "added file must be detected");

    td.subdir("d2");
    let current: DirectoryState = h2.state_current().unwrap();
    match h2.state().change(&current) {
        StateChange::DirNum(d) => assert_eq!(d, 1),
        c => panic!("expected DirNum(1), got {c:?}"),
    }
    assert!(h2.state_changed().unwrap(), "added dir must be detected");

    // rename: same counts, different entry hash
    fs::rename(td.path().join("f2.txt"), td.path().join("f2-renamed.txt")).unwrap();
    let current: DirectoryState = h2.state_current().unwrap();
    match h2.state().change(&current) {
        StateChange::FileHash => {}
        c => panic!("expected FileHash, got {c:?}"),
    }
    assert!(h2.state_changed().unwrap(), "rename must be detected");
    assert!(!h2.state_changed().unwrap(), "stable after update");

    // early-dropped iterator must not finalize state
    let mut h3 = DirHandle::new(td.path()).unwrap();
    {
        let mut it = h3.iter();
        let _ = it.next();
    }
    assert_eq!(h3.state(), &DirectoryState::default(), "no partial finalize");
}

#[test]
fn state_changed_fast_lifecycle() {
    let td = TestDir::new("state-fast");
    td.file("f1.txt", b"1");

    let mut h = DirHandle::new(td.path()).unwrap();
    assert!(!h.state_changed_fast().unwrap(), "first call = baseline");
    assert!(!h.state_changed_fast().unwrap(), "no change");

    td.file("f2.txt", b"2");
    assert!(h.state_changed_fast().unwrap(), "added file must be detected");
    assert!(!h.state_changed_fast().unwrap(), "stable again");

    // backdating the directory mtime does NOT defeat the pre-check,
    // because utimensat bumps ctime (which the pre-check also considers)
    td.file("f3.txt", b"3");
    let c_path: CString = CString::new(td.path().as_os_str().as_bytes()).unwrap();
    let past = libc::timespec { tv_sec: 1_000_000, tv_nsec: 0 };
    let times: [libc::timespec; 2] = [past, past];
    let rc: i32 =
        unsafe { libc::utimensat(libc::AT_FDCWD, c_path.as_ptr(), times.as_ptr(), 0) };
    assert_eq!(rc, 0, "utimensat failed");
    assert!(h.state_changed_fast().unwrap(), "ctime guard must catch backdated mtime");
}

#[test]
fn state_when_is_stamped_at_pass_start() {
    /*
    Regression: `when` used to be stamped at the *end* of a pass. A change
    landing early in a pass longer than MTIME_SLACK_SECS then had an mtime
    "clearly older" than the baseline, so `state_changed_fast()` reported
    unchanged forever, even though the listing had never seen the entry.
    */
    let td = TestDir::new("state-when");
    for i in 0..5 {
        td.file(&format!("f{i}"), b"x");
    }

    let mut h = DirHandle::new(td.path()).unwrap();
    {
        // lazy first pass: the first next() makes glibc slurp the whole
        // (small) directory into its buffer, so the file created below is
        // not seen by this pass
        let mut it = h.iter();
        let _ = it.next();
        td.file("late", b"x"); // dir mtime = now
        std::thread::sleep(std::time::Duration::from_millis(2500)); // > slack
        let n: usize = 1 + it.by_ref().count();
        assert_eq!(n, 5, "the late file must not have been listed");
    } // clean exhaustion finalizes the state
    assert_ne!(h.state(), &DirectoryState::default(), "state must be populated");
    assert!(h.state_changed_fast().unwrap(), "mid-pass change must be detected");
}

/* ########################## v0.4.1 ADDITIONS ############################# */

#[test]
fn open_dir_descends_and_stays_beneath() {
    let td = TestDir::new("open-dir");
    let sub: PathBuf = td.subdir("sub");
    fs::write(sub.join("inner.txt"), b"x").unwrap();
    td.file("plain.txt", b"p");
    // a symlink escaping the tree, and one staying beneath
    std::os::unix::fs::symlink("/", td.path().join("escape")).unwrap();
    std::os::unix::fs::symlink("sub", td.path().join("benign")).unwrap();

    let mut h = DirHandle::new(td.path()).unwrap();
    let entries: Vec<EntryExt> = h.iter().collect();

    let by_name = |n: &[u8]| entries.iter().find(|e| e.name_as_bytes() == n).unwrap();

    // descend into a real subdir: no procfs, no path re-resolution
    let mut sub_h: DirHandle = by_name(b"sub").open_dir().expect("open_dir must work");
    let names: Vec<String> = sub_h.iter().map(|e| e.name()).collect();
    assert_eq!(names, ["inner.txt"]);
    assert!(has_cloexec(sub_h.as_raw_fd()), "open_dir fd must be CLOEXEC");

    // non-directories are rejected
    assert!(by_name(b"plain.txt").open_dir().is_err(), "ENOTDIR expected");

    // RESOLVE_BENEATH: escaping symlinks are rejected by the kernel,
    // in-tree symlinks resolve fine
    assert!(by_name(b"escape").open_dir().is_err(), "escape must be blocked");
    let mut benign: DirHandle = by_name(b"benign").open_dir().expect("in-tree symlink");
    assert_eq!(benign.iter().count(), 1);
}

#[test]
fn from_fd_and_dir_stat() {
    let td = TestDir::new("from-fd");
    td.file("f.txt", b"x");

    // a directory fd from the std File API works
    let owned: OwnedFd = fs::File::open(td.path()).unwrap().into();
    let mut h = DirHandle::from_fd(owned).unwrap();
    assert_eq!(h.iter().count(), 1);

    // a non-directory fd is rejected - and closed, not leaked
    let file: PathBuf = td.path().join("f.txt");
    let file_fd: OwnedFd = fs::File::open(&file).unwrap().into();
    let raw: i32 = file_fd.as_raw_fd();
    let err = DirHandle::from_fd(file_fd).expect_err("ENOTDIR expected");
    assert_eq!(err.raw_os_error(), Some(libc::ENOTDIR), "{err}");
    assert!(!fd_refers_to(raw, &file), "rejected fd must be closed");

    // so is an O_PATH fd, which passes the type check but cannot be listed
    let c_path: CString = CString::new(td.path().as_os_str().as_bytes()).unwrap();
    let raw: i32 =
        unsafe { libc::open(c_path.as_ptr(), libc::O_PATH | libc::O_DIRECTORY | libc::O_CLOEXEC) };
    assert!(raw >= 0, "O_PATH open failed");
    let err = DirHandle::from_fd(unsafe { OwnedFd::from_raw_fd(raw) }).expect_err("EBADF expected");
    assert_eq!(err.raw_os_error(), Some(libc::EBADF), "{err}");
    assert!(!fd_refers_to(raw, td.path()), "rejected O_PATH fd must be closed");

    // stat/mtime of the directory itself
    let st: libc::stat = h.stat().unwrap();
    assert_eq!(st.st_mode & libc::S_IFMT, libc::S_IFDIR);
    let mtime: f64 = h.mtime().unwrap().get();
    assert!(mtime > 1.5e9 && mtime < 4.0e9, "sane dir mtime: {mtime}");
}

#[test]
fn entry_stat_accessors() {
    let td = TestDir::new("accessors");
    td.file("full.txt", b"some bytes");
    td.file("empty.txt", b"");

    let mut h = DirHandle::new(td.path()).unwrap();
    for e in h.iter() {
        match e.name_as_bytes() {
            b"full.txt" => {
                assert!(!e.is_empty());
                assert_eq!(e.len(), 10);
            }
            b"empty.txt" => {
                assert!(e.is_empty());
                assert_eq!(e.len(), 0);
            }
            other => panic!("unexpected entry {other:?}"),
        }
        assert_eq!(e.uid().unwrap(), unsafe { libc::getuid() });
        assert_eq!(e.gid().unwrap(), unsafe { libc::getgid() });
        assert_eq!(e.nlink().unwrap(), 1);
        for t in [e.mtime(), e.atime(), e.ctime()] {
            let secs: f64 = t.unwrap().get();
            assert!(secs > 1.5e9 && secs < 4.0e9, "sane timestamp: {secs}");
        }
    }
}

/* ############################ OPEN HANDLES ############################### */

#[test]
fn open_handles_pool() {
    let td = TestDir::new("pool");
    td.file("f.txt", b"x");
    let sub: PathBuf = td.subdir("sub");

    let pool = OpenHandles::new();
    let fd_a: i32 = {
        let mut a = pool.open(td.path()).unwrap();
        assert_eq!(a.iter().count(), 2);
        a.as_raw_fd()
    }; // checkout dropped: handle stays in the pool
    assert_eq!(pool.len(), 1);
    assert!(pool.contains(fd_a));

    // re-checkout and explicitly close: removed from pool, fd released
    let again = pool.get(fd_a).expect("handle must still be in the pool");
    again.close();
    assert!(!pool.contains(fd_a));
    assert_eq!(pool.len(), 0);

    // multiple handles + for_each
    let fd_1: i32 = pool.open(td.path()).unwrap().as_raw_fd();
    let _fd_2: i32 = pool.open(&sub).unwrap().as_raw_fd();
    assert_eq!(pool.len(), 2);
    let mut seen: usize = 0;
    pool.for_each(|_| seen += 1);
    assert_eq!(seen, 2);
    pool.close(fd_1);
    assert_eq!(pool.len(), 1);
    pool.close_all();
    assert_eq!(pool.len(), 0);
}

#[test]
fn open_handles_across_threads() {
    let td = TestDir::new("pool-threads");
    for i in 0..8 {
        let sub: PathBuf = td.subdir(&format!("sub{i}"));
        fs::write(sub.join("f.txt"), b"x").unwrap();
    }

    let pool = OpenHandles::new();
    std::thread::scope(|scope| {
        for i in 0..8 {
            let pool = &pool;
            let dir: PathBuf = td.path().join(format!("sub{i}"));
            scope.spawn(move || {
                let mut h = pool.open(&dir).unwrap();
                assert_eq!(h.iter().count(), 1);
                assert!(!h.state_changed().unwrap());
            });
        }
    });
    assert_eq!(pool.len(), 8);
    pool.close_all();
    assert_eq!(pool.len(), 0);
}
