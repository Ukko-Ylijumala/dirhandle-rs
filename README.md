# DirHandle - A Rust Directory Handling Utility

## Overview
**DirHandle** is a Rust library that provides efficient and thread-safe directory handling, entry iteration, and state tracking. It extends `nix::dir::Entry` to offer additional metadata caching, hashing, and sorting functionality.

## Features
- **Thread-safe File Descriptor Wrapper (`DirFd`)**
- **Extended Directory Entry Handling (`EntryExt`)**
  - Cached `stat()` results using `OnceLock`
  - Supports file type determination (`is_dir()`, `is_file()`, etc.)
- **State Tracking (`DirectoryState`)**
  - Monitors file and directory counts with `xxh3()` hashing
- **Optimized Iterators (`DirHandleIter`)**
  - Lookahead buffering for prioritized directory traversal
- **Thread-safe Handle Storage (`OpenHandles`)**
  - Uses `DashMap` for concurrent access
  - Provides checked-out handles with automatic cleanup

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
dirhandle = { git = "https://github.com/Ukko-Ylijumala/dirhandle-rs" }
```

## Usage
### Open a Directory
```rust
use std::path::Path;
use dirhandle::DirHandle;

fn main() {
    let path = Path::new("/some/directory");
    let mut handle = DirHandle::new(path).expect("Failed to open directory");
    
    for entry in handle.iter() {
        println!("Found: {}", entry.name());
    }
}
```

### Track Directory Changes
```rust
if handle.state_changed() {
    println!("Directory contents changed!");
}
```

### Open Handles Thread-Safely
```rust
use dirhandle::OpenHandles;

let handles = OpenHandles::new();
if let Ok(mut handle) = handles.open(&path) {
    for entry in handle.iter_sorted() {
        println!("Sorted entry: {}", entry.name());
    }
}
```

## License

Copyright (c) 2024-2025 Mikko Tanner. All rights reserved.

License: MIT OR Apache-2.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Version History

- 0.3.5: Initial library version
    - Filter DirHandle code to a separate crate

This library started its life as a component of a larger application, but at some point it made more sense to separate the code into its own little project and here we are.
