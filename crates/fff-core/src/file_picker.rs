//! Core file picker: filesystem indexing, background watching, and fuzzy search.
//!
//! [`FilePicker`] is the central component of fff-search. It:
//!
//! 1. **Indexes** a directory tree in a background thread, collecting every
//!    non-ignored file into a path-sorted `Vec<FileItem>`.
//! 2. **Watches** the filesystem via the `notify` crate, applying
//!    create/modify/delete events to the index in real time.
//! 3. **Owns files**: Provides a values for search and provides a good entry point for
//!    fuzzy search and live grep
//!
//! # Lifecycle
//!
//! ```text
//!   new_with_shared_state()
//!     │
//!     ├─> background scan thread ──> populates SharedPicker
//!     └─> file-system watcher    ──> live updates SharedPicker
//!
//!   search()         <── borrows &self, delegates to fuzzy_search
//!   grep()           <── static, borrows &[FileItem] (live content search)
//!   trigger_rescan() <── synchronous re-index
//!   cancel()         <── shuts down background work
//! ```
//!
//! # Thread Safety
//!
//! `FilePicker` itself is **not** `Sync`!
//! all concurrent access goes through [`SharedPicker`](crate::SharedPicker) .
//! The background scanner and watcher acquire write locks only when mutating
//! the file index, so read-heavy search workloads rarely contend.

use crate::FFFStringStorage;
use crate::background_watcher::{BackgroundWatcher, is_git_file};
use crate::bigram_filter::{BigramFilter, BigramOverlay};
use crate::error::Error;
use crate::frecency::FrecencyTracker;
use crate::git::GitStatusCache;
use crate::grep::{GrepResult, GrepSearchOptions, grep_search, multi_grep_search};
use crate::ignore::non_git_repo_overrides;
use crate::query_tracker::QueryTracker;
use crate::scan::{ScanConfig, ScanJob, ScanSignals};
use crate::score::fuzzy_match_and_score_files;
use crate::shared::{SharedFrecency, SharedPicker};
use crate::simd_path::{ArenaPtr, PATH_BUF_SIZE};
use crate::types::{
    ContentCacheBudget, DirItem, DirSearchResult, FileItem, MixedItemRef, MixedSearchResult,
    PaginationArgs, Score, ScoringContext, SearchResult,
};
use fff_query_parser::FFFQuery;
use git2::{Repository, Status};
use rayon::prelude::*;
use std::fmt::Debug;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, LazyLock,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};
use std::time::SystemTime;
use tracing::{Level, debug, error, info, warn};

/// Dedicated thread pool for background work (scan, warmup, bigram build).
/// Uses fewer threads than the global rayon pool so Neovim's event loop
/// and search queries can still get CPU time.
pub(crate) static BACKGROUND_THREAD_POOL: LazyLock<rayon::ThreadPool> = LazyLock::new(|| {
    let total = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);
    let bg_threads = total.saturating_sub(2).max(1);
    info!(
        "Background pool: {} threads (system has {})",
        bg_threads, total
    );
    rayon::ThreadPoolBuilder::new()
        .num_threads(bg_threads)
        .thread_name(|i| format!("fff-bg-{i}"))
        .start_handler(|_| {
            // Pin workers to the USER_INITIATED QoS class on macOS so the
            // scheduler keeps them on P-cores. Without this the kernel is
            // free to drift them to E-cores, which are ~2× slower for the
            // bigram scan and per-file syscalls.
            #[cfg(target_os = "macos")]
            unsafe {
                let _ = libc::pthread_set_qos_class_self_np(
                    libc::qos_class_t::QOS_CLASS_USER_INITIATED,
                    0,
                );
            }
        })
        .build()
        .expect("failed to create background rayon pool")
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FFFMode {
    #[default]
    Neovim,
    Ai,
}

impl FFFMode {
    pub fn is_ai(self) -> bool {
        self == FFFMode::Ai
    }
}

/// Configuration for a single fuzzy search invocation.
///
/// Passed to [`FilePicker::search`] to control threading, pagination,
/// and scoring behavior.
#[derive(Debug, Clone, Copy, Default)]
pub struct FuzzySearchOptions<'a> {
    pub max_threads: usize,
    pub current_file: Option<&'a str>,
    pub project_path: Option<&'a Path>,
    pub combo_boost_score_multiplier: i32,
    pub min_combo_count: u32,
    pub pagination: PaginationArgs,
}

#[derive(Debug, Clone)]
pub(crate) struct FileSync {
    pub(crate) git_workdir: Option<PathBuf>,
    /// Base files laid out in two partitions, each internally sorted by
    /// (parent_dir, filename):
    ///   `files[..indexable_count]`        — bigram-indexable (text-by-ext,
    ///                                        0 < size <= max_mmap_file_size)
    ///   `files[indexable_count..base_count]` — known-unindexable (binary,
    ///                                           zero-sized, or too large)
    ///   `files[base_count..]`              — overflow (created since the
    ///                                        last full reindex)
    ///
    /// The `indexable_count` split lets the bigram builder allocate column
    /// bitsets sized to the eligible subset instead of the full base, saving
    /// ~20% of peak RSS on directories with many non-code files.
    files: Vec<FileItem>,
    /// Number of bigram-indexable files at the head of `files`. Must satisfy
    /// `indexable_count <= base_count <= files.len()`. Files at
    /// `[..indexable_count]` are the only ones referenced by the bigram
    /// inverted index.
    indexable_count: usize,
    /// Number of base files (from the last full reindex). Overflow files
    /// live at `files[base_count..]`, each with its own `ChunkedPathStore`
    /// kept alive in `overflow_stores`.
    base_count: usize,
    /// Sorted directory table. Each entry is a unique parent directory of at
    /// least one file in `files`. Sorted by absolute path for O(log n) lookup.
    /// Built during `walk_filesystem` and used for directory picker mode,
    /// per-directory stats, and as a fast replacement for `extract_watch_dirs`.
    dirs: Vec<DirItem>,
    /// Shared builder for overflow file paths. Each overflow file's ChunkedString
    /// uses `arena_override` pointing into this builder's arena. The builder
    /// grows incrementally — no per-file store allocation. Dropped on rescan.
    overflow_builder: Option<crate::simd_path::ChunkedPathStoreBuilder>,
    /// Compressed bigram inverted index built during the post-scan phase.
    /// Lives here so that replacing `FileSync` on rescan automatically drops
    /// the stale index (bigram file indices are positions in `files`).
    bigram_index: Option<Arc<BigramFilter>>,
    /// Overlay tracking file mutations since the bigram index was built.
    bigram_overlay: Option<Arc<parking_lot::RwLock<BigramOverlay>>>,
    /// Chunk-level deduped path store for zero-copy SIMD matching.
    /// Each file's relative path is pre-chunked into 16-byte aligned blocks
    /// with content-based deduplication across files.
    chunked_paths: Option<crate::simd_path::ChunkedPathStore>,
}

impl FileSync {
    fn new() -> Self {
        Self {
            files: Vec::new(),
            indexable_count: 0,
            base_count: 0,
            dirs: Vec::new(),
            overflow_builder: None,
            git_workdir: None,
            bigram_index: None,
            bigram_overlay: None,
            chunked_paths: None,
        }
    }

    /// Arena for base files (from the last full scan).
    #[inline]
    fn arena_base_ptr(&self) -> ArenaPtr {
        self.chunked_paths
            .as_ref()
            .map(|s| s.as_arena_ptr())
            .unwrap_or(ArenaPtr::null())
    }

    /// Arena for overflow files (added after the last full scan).
    #[inline]
    fn overflow_arena_ptr(&self) -> ArenaPtr {
        self.overflow_builder
            .as_ref()
            .map(|b| b.as_arena_ptr())
            .unwrap_or(self.arena_base_ptr())
    }

    /// Resolve the correct arena for a given file (base vs overflow).
    #[inline]
    fn arena_for_file(&self, file: &FileItem) -> ArenaPtr {
        if file.is_overflow() {
            self.overflow_arena_ptr()
        } else {
            self.arena_base_ptr()
        }
    }

    /// Get all files (base + overflow). The base portion `[..base_count]` is
    /// sorted by path; the overflow tail is unsorted.
    #[inline]
    fn files(&self) -> &[FileItem] {
        &self.files
    }

    /// Get the overflow portion (files added since last full reindex).
    #[inline]
    fn overflow_files(&self) -> &[FileItem] {
        &self.files[self.base_count..]
    }

    /// Get mutable file at index (works for base files only).
    #[inline]
    fn get_file_mut(&mut self, index: usize) -> Option<&mut FileItem> {
        self.files.get_mut(index)
    }

    /// Find file index by path using binary search on the sorted base portion.
    /// `path` must be an absolute path under `base_path`.
    #[inline]
    fn find_file_index(&self, path: &Path, base_path: &Path) -> Result<usize, usize> {
        let arena = self.arena_base_ptr();

        // Strip base_path prefix to get the relative path.
        let rel_path = match path.strip_prefix(base_path) {
            Ok(r) => r.to_string_lossy(),
            Err(_) => return Err(0),
        };

        // Split into directory (with trailing '/') and filename.
        let parent_end = rel_path
            .rfind(std::path::is_separator)
            .map(|i| i + 1)
            .unwrap_or(0);
        let dir_rel = &rel_path[..parent_end];
        let filename = &rel_path[parent_end..];

        // Binary search dirs to find the parent directory index.
        // Dir items store the relative path including trailing '/' (e.g. "src/components/").
        let mut dir_buf = [0u8; crate::simd_path::PATH_BUF_SIZE];
        let dir_idx = match self
            .dirs
            .binary_search_by(|d| d.read_relative_path(arena, &mut dir_buf).cmp(dir_rel))
        {
            Ok(idx) => idx as u32,
            Err(_) => return Err(0), // directory not found
        };

        // Binary search files by (parent_dir, filename). Base files live in
        // two internally-sorted partitions — indexable first, then
        // unindexable — so we try each half in turn. Two O(log n) searches
        // with short-circuit on the first hit.
        let cmp_key = |f: &FileItem| {
            f.parent_dir_index().cmp(&dir_idx).then_with(|| {
                let fname = f.file_name(arena);
                fname.as_str().cmp(filename)
            })
        };

        if self.indexable_count > 0
            && let Ok(pos) = self.files[..self.indexable_count].binary_search_by(cmp_key)
        {
            return Ok(pos);
        }

        if self.indexable_count < self.base_count
            && let Ok(rel_pos) =
                self.files[self.indexable_count..self.base_count].binary_search_by(cmp_key)
        {
            return Ok(self.indexable_count + rel_pos);
        }

        Err(0)
    }

    /// Find a file in the overflow portion by relative path (linear scan).
    /// Returns the absolute index into `files` (i.e. `base_count + position`).
    fn find_overflow_index(&self, rel_path: &str) -> Option<usize> {
        let overflow_arena = self.overflow_arena_ptr();
        self.files[self.base_count..]
            .iter()
            .position(|f| f.relative_path_eq(overflow_arena, rel_path))
            .map(|pos| self.base_count + pos)
    }

    /// Insert a file at position. Simple - no HashMap to maintain!
    fn insert_file(&mut self, position: usize, file: FileItem) {
        self.files.insert(position, file);
    }

    fn retain_files_with_arena<F>(&mut self, mut predicate: F) -> usize
    where
        F: FnMut(&FileItem, ArenaPtr) -> bool,
    {
        let base_arena = self.arena_base_ptr();
        let overflow_arena = self.overflow_arena_ptr();

        let indexable_count = self.indexable_count;
        let base_count = self.base_count;
        let initial_len = self.files.len();

        let indexable_retained = self.files[..indexable_count]
            .iter()
            .filter(|f| predicate(f, base_arena))
            .count();
        let base_retained = self.files[indexable_count..base_count]
            .iter()
            .filter(|f| predicate(f, base_arena))
            .count()
            + indexable_retained;

        self.files.retain(|f| {
            predicate(
                f,
                if f.is_overflow() {
                    overflow_arena
                } else {
                    base_arena
                },
            )
        });

        self.indexable_count = indexable_retained;
        self.base_count = base_retained;
        initial_len - self.files.len()
    }

    /// Insert a file in sorted order (by path).
    /// Returns true if inserted, false if file already exists.
    ///
    /// TODO: `find_file_index` no longer returns a meaningful insertion
    /// point (`Err(0)` is a sentinel, not a binary-search bucket). This
    /// function now always inserts at the end of the indexable partition.
    /// It's only reachable via `add_file_sorted`, which is unused. If it
    /// ever gets a caller, re-derive the insertion point explicitly.
    fn insert_file_sorted(&mut self, file: FileItem, base_path: &Path) -> bool {
        let arena = self.arena_base_ptr();
        let abs_path = file.absolute_path(arena, base_path);
        match self.find_file_index(&abs_path, base_path) {
            Ok(_) => false, // File already exists
            Err(_) => {
                let position = self.indexable_count;
                self.insert_file(position, file);
                self.indexable_count += 1;
                self.base_count += 1;
                true
            }
        }
    }
}

impl FileItem {
    pub fn new(path: PathBuf, base_path: &Path, git_status: Option<Status>) -> (Self, String) {
        let metadata = std::fs::metadata(&path).ok();
        Self::new_with_metadata(path, base_path, git_status, metadata.as_ref())
    }

    /// Create a FileItem using pre-fetched metadata to avoid a redundant stat syscall.
    /// Returns `(FileItem, relative_path)`. The FileItem's `path` field is
    /// empty; callers must populate it via `set_path` or `build_chunked_path_store_and_assign`.
    fn new_with_metadata(
        path: PathBuf,
        base_path: &Path,
        git_status: Option<Status>,
        metadata: Option<&std::fs::Metadata>,
    ) -> (Self, String) {
        let path_buf = pathdiff::diff_paths(&path, base_path).unwrap_or_else(|| path.clone());
        let relative_path = path_buf.to_string_lossy().into_owned();

        let (size, modified) = match metadata {
            Some(metadata) => {
                let size = metadata.len();
                let modified = metadata
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                    .map_or(0, |d| d.as_secs());

                (size, modified)
            }
            None => (0, 0),
        };

        let is_binary = is_known_binary_extension(&path);

        let filename_start = relative_path
            .rfind(std::path::is_separator)
            .map(|i| i + 1)
            .unwrap_or(0) as u16;

        let item = Self::new_raw(filename_start, size, modified, git_status, is_binary);
        (item, relative_path)
    }

    /// Create a FileItem with an empty ChunkedString from a path on disk.
    ///
    /// Returns `(file_item, relative_path_string)`. The relative path must be
    /// kept alongside the FileItem until `build_chunked_path_store_and_assign`
    /// populates each item's `path` field from the shared arena.
    pub fn new_from_walk(
        path: &Path,
        base_path: &Path,
        git_status: Option<Status>,
        metadata: Option<&std::fs::Metadata>,
    ) -> (Self, String) {
        let (size, modified) = match metadata {
            Some(metadata) => {
                let size = metadata.len();
                let modified = metadata
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                    .map_or(0, |d| d.as_secs());
                (size, modified)
            }
            None => (0, 0),
        };

        let is_binary = is_known_binary_extension(path);

        let rel = pathdiff::diff_paths(path, base_path).unwrap_or_else(|| path.to_path_buf());
        let rel_str = rel.to_string_lossy().into_owned();
        let fname_offset = rel_str
            .rfind(std::path::is_separator)
            .map(|i| i + 1)
            .unwrap_or(0) as u16;

        let item = Self::new_raw(fname_offset, size, modified, git_status, is_binary);
        (item, rel_str)
    }

    pub(crate) fn update_frecency_scores(
        &mut self,
        tracker: &FrecencyTracker,
        arena: ArenaPtr,
        base_path: &Path,
        mode: FFFMode,
    ) -> Result<(), Error> {
        let mut abs_buf = [0u8; crate::simd_path::PATH_BUF_SIZE];
        let abs = self.write_absolute_path(arena, base_path, &mut abs_buf);
        self.access_frecency_score = tracker.get_access_score(abs, mode) as i16;
        self.modification_frecency_score =
            tracker.get_modification_score(self.modified, self.git_status, mode) as i16;

        Ok(())
    }
}

/// Options for creating a [`FilePicker`].
pub struct FilePickerOptions {
    pub base_path: String,
    /// Pre-populate mmap caches for top-frecency files after the initial scan.
    pub enable_mmap_cache: bool,
    /// Build content index after the initial scan for faster content-aware filtering.
    pub enable_content_indexing: bool,
    /// Mode of the picker impact the way file watcher events are handled and the scoring logic
    pub mode: FFFMode,
    /// Explicit cache budget. When `None`, the budget is auto-computed from
    /// the repo size after the initial scan completes.
    pub cache_budget: Option<ContentCacheBudget>,
    /// When `false`, `new_with_shared_state` skips the background file watcher.
    pub watch: bool,
}

impl Default for FilePickerOptions {
    fn default() -> Self {
        Self {
            base_path: ".".into(),
            enable_mmap_cache: false,
            enable_content_indexing: false,
            mode: FFFMode::default(),
            cache_budget: None,
            watch: true,
        }
    }
}

pub struct FilePicker {
    pub mode: FFFMode,
    pub base_path: PathBuf,
    signals: ScanSignals,
    sync_data: FileSync,
    cache_budget: Arc<ContentCacheBudget>,
    has_explicit_cache_budget: bool,
    scanned_files_count: Arc<AtomicUsize>,
    background_watcher: Option<BackgroundWatcher>,
    enable_mmap_cache: bool,
    enable_content_indexing: bool,
    watch: bool,
}

impl std::fmt::Debug for FilePicker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FilePicker")
            .field("base_path", &self.base_path)
            .field("sync_data", &self.sync_data)
            .field(
                "is_scanning",
                &self.signals.scanning.load(Ordering::Relaxed),
            )
            .field(
                "scanned_files_count",
                &self.scanned_files_count.load(Ordering::Relaxed),
            )
            .finish_non_exhaustive()
    }
}

impl FFFStringStorage for &FilePicker {
    #[inline]
    fn arena_for(&self, file: &FileItem) -> crate::simd_path::ArenaPtr {
        self.sync_data.arena_for_file(file)
    }

    #[inline]
    fn base_arena(&self) -> crate::simd_path::ArenaPtr {
        self.sync_data.arena_base_ptr()
    }

    #[inline]
    fn overflow_arena(&self) -> crate::simd_path::ArenaPtr {
        self.sync_data.overflow_arena_ptr()
    }
}

impl FilePicker {
    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    /// Convert an absolute path to a relative path string (relative to base_path).
    /// Returns None if the path doesn't start with base_path.
    fn to_relative_path<'a>(&self, path: &'a Path) -> Option<&'a str> {
        path.strip_prefix(&self.base_path)
            .ok()
            .and_then(|p| p.to_str())
    }

    pub fn need_enable_mmap_cache(&self) -> bool {
        self.enable_mmap_cache
    }

    pub fn need_enable_content_indexing(&self) -> bool {
        self.enable_content_indexing
    }

    pub fn need_watch(&self) -> bool {
        self.watch
    }

    pub fn mode(&self) -> FFFMode {
        self.mode
    }

    pub fn cache_budget(&self) -> &ContentCacheBudget {
        &self.cache_budget
    }

    pub fn bigram_index(&self) -> Option<&BigramFilter> {
        self.sync_data.bigram_index.as_deref()
    }

    pub fn bigram_overlay(&self) -> Option<&parking_lot::RwLock<BigramOverlay>> {
        self.sync_data.bigram_overlay.as_deref()
    }

    pub fn get_file_mut(&mut self, index: usize) -> Option<&mut FileItem> {
        self.sync_data.get_file_mut(index)
    }

    pub fn set_bigram_index(&mut self, index: BigramFilter, overlay: BigramOverlay) {
        self.sync_data.bigram_index = Some(Arc::new(index));
        self.sync_data.bigram_overlay = Some(Arc::new(parking_lot::RwLock::new(overlay)));
    }

    /// Absolute path to the repository root if the indexed tree lives
    /// inside a git working directory. `None` for non-git bases.
    pub fn git_root(&self) -> Option<&Path> {
        self.sync_data.git_workdir.as_deref()
    }

    pub fn has_explicit_cache_budget(&self) -> bool {
        self.has_explicit_cache_budget
    }

    pub fn set_cache_budget(&mut self, budget: ContentCacheBudget) {
        self.cache_budget = Arc::new(budget);
    }

    pub(crate) fn cache_budget_arc(&self) -> Arc<ContentCacheBudget> {
        Arc::clone(&self.cache_budget)
    }

    /// Bundle the atomic flags the scan orchestrator needs. One method
    /// instead of four separate getters so every callsite passes a
    /// single `ScanSignals` value.
    pub(crate) fn scan_signals(&self) -> crate::scan::ScanSignals {
        crate::scan::ScanSignals {
            scanning: Arc::clone(&self.signals.scanning),
            watcher_ready: Arc::clone(&self.signals.watcher_ready),
            cancelled: Arc::clone(&self.signals.cancelled),
            post_scan_busy: Arc::clone(&self.signals.post_scan_busy),
            rescan_pending: Arc::clone(&self.signals.rescan_pending),
        }
    }

    /// Clone of the `scanned_files_count` atomic counter. The walker
    /// bumps it per-file so the UI progress indicator can poll it via
    /// `get_scan_progress` without holding any lock.
    pub(crate) fn scanned_files_counter(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.scanned_files_count)
    }

    /// Reference to `sync_data.files()` exposed with arena base ptr and
    /// `indexable_count` so the post-scan orchestrator can snapshot a
    /// raw slice for off-lock work. SAFETY is upheld externally via
    /// the `post_scan_busy` flag.
    pub(crate) fn sync_data_snapshot(&self) -> (&[FileItem], usize, ArenaPtr) {
        (
            self.sync_data.files(),
            self.sync_data.indexable_count,
            self.sync_data.arena_base_ptr(),
        )
    }

    pub(crate) fn install_background_watcher(&mut self, watcher: BackgroundWatcher) {
        self.background_watcher = Some(watcher);
    }

    pub(crate) fn commit_new_sync(&mut self, sync: FileSync) {
        self.sync_data = sync;
        self.cache_budget.reset();
    }

    /// Get all indexed files sorted by path.
    /// Note: Files are stored sorted by PATH for efficient insert/remove.
    /// For frecency-sorted results, use search() which sorts matched results.
    pub fn get_files(&self) -> &[FileItem] {
        self.sync_data.files()
    }

    pub fn get_overflow_files(&self) -> &[FileItem] {
        self.sync_data.overflow_files()
    }

    /// Get the directory table (sorted by path).
    pub fn get_dirs(&self) -> &[DirItem] {
        &self.sync_data.dirs
    }

    /// Actual heap bytes used: (chunked_path_store, 0, 0).
    /// The second element is 0 because leaked overflow stores aren't tracked.
    pub fn arena_bytes(&self) -> (usize, usize, usize) {
        let chunked = self
            .sync_data
            .chunked_paths
            .as_ref()
            .map_or(0, |s| s.heap_bytes());
        (chunked, 0, 0)
    }

    #[tracing::instrument(level = "debug", skip_all)]
    pub(crate) fn for_each_dir(&self, mut f: impl FnMut(&Path) -> ControlFlow<()>) {
        let dir_table = &self.sync_data.dirs;
        let base = self.base_path.as_path();

        if !dir_table.is_empty() {
            let arena = self.arena_base_ptr();
            let mut path_buf = PathBuf::with_capacity(crate::simd_path::PATH_BUF_SIZE);
            let mut prev_relative_path = String::new();

            let mut scratch_buf = [0u8; crate::simd_path::PATH_BUF_SIZE];
            for dir_item in dir_table {
                let full_relative_path = dir_item.read_relative_path(arena, &mut scratch_buf);
                let relative_path = full_relative_path.trim_end_matches(std::path::is_separator);

                if relative_path.is_empty() {
                    // Files directly under base_path
                    prev_relative_path.clear();
                    continue;
                }

                let mut i = common_dir_prefix_len(&prev_relative_path, relative_path);
                // If we stopped on a separator, skip it — we want to start
                // emitting at the first unseen segment, not re-emit the
                // already-emitted prefix path.
                if i < relative_path.len()
                    && std::path::is_separator(relative_path.as_bytes()[i] as char)
                {
                    i += 1;
                }

                // Walk the suffix of `relative_path` one segment at a time, emitting
                // each previously unseen ancestor up to and including `relative_path`.
                while i < relative_path.len() {
                    let next_sep = relative_path[i..]
                        .find(std::path::is_separator)
                        .map(|off| i + off)
                        .unwrap_or(relative_path.len());
                    let ancestor_rel = &relative_path[..next_sep];

                    path_buf.clear();
                    path_buf.push(base);
                    path_buf.push(ancestor_rel);

                    // we can't really emit iterator here unfortunately
                    if matches!(f(path_buf.as_path()), ControlFlow::Break(())) {
                        return;
                    }

                    i = next_sep + 1;
                }

                prev_relative_path.clear();
                prev_relative_path.push_str(relative_path);
            }
            return;
        }

        // fallback that should never be happening, but it is possible to get the file
        // path from the absolute path using componenets api as well:
        let files = self.sync_data.files();
        let arena = self.arena_base_ptr();
        let mut current = self.base_path.clone();
        let mut path_buf = [0u8; PATH_BUF_SIZE];

        for file in files {
            let abs = file.write_absolute_path(arena, base, &mut path_buf);
            let Some(parent) = abs.parent() else {
                continue;
            };
            if parent == current.as_path() {
                continue;
            }

            while current.as_path() != base && !parent.starts_with(&current) {
                current.pop();
            }

            let Ok(remainder) = parent.strip_prefix(&current) else {
                continue;
            };
            for component in remainder.components() {
                current.push(component);
                if matches!(f(current.as_path()), ControlFlow::Break(())) {
                    return;
                }
            }
        }
    }

    /// Lower bound on the number of directories `for_each_watch_dir` will
    /// emit. Equals `sync_data.dirs.len()` — cheap to compute without the
    /// walk. Used to decide between per-dir NonRecursive watches and a
    /// single Recursive watch on macOS, where FSEvents streams are capped
    /// per process. The true count can be slightly higher (pure-ancestor
    /// dirs aren't in `sync_data.dirs`), but for any realistic repo the
    /// gap is small compared to the threshold's headroom.
    pub fn watch_dirs_count_hint(&self) -> usize {
        self.sync_data.dirs.len()
    }

    /// Create a new FilePicker from options.
    /// Always prefer new_with_shared_state for the consumer application, use this only if you know
    /// what you are doing. This won't spawn the backgraound watcher and won't walk the file tree.
    pub fn new(options: FilePickerOptions) -> Result<Self, Error> {
        let path = PathBuf::from(&options.base_path);
        if !path.exists() {
            error!("Base path does not exist: {}", options.base_path);
            return Err(Error::InvalidPath(path));
        }
        if path.parent().is_none() {
            error!("Refusing to index filesystem root: {}", path.display());
            return Err(Error::FilesystemRoot(path));
        }

        let has_explicit_budget = options.cache_budget.is_some();
        let initial_budget = options.cache_budget.unwrap_or_default();

        Ok(FilePicker {
            background_watcher: None,
            base_path: path,
            cache_budget: Arc::new(initial_budget),
            has_explicit_cache_budget: has_explicit_budget,
            signals: crate::scan::ScanSignals::default(),
            mode: options.mode,
            scanned_files_count: Arc::new(AtomicUsize::new(0)),
            sync_data: FileSync::new(),
            enable_mmap_cache: options.enable_mmap_cache,
            enable_content_indexing: options.enable_content_indexing,
            watch: options.watch,
        })
    }

    /// Create a picker, place it into the shared handle, and spawn background
    /// indexing + file-system watcher. This is the default entry point.
    pub fn new_with_shared_state(
        shared_picker: SharedPicker,
        shared_frecency: SharedFrecency,
        options: FilePickerOptions,
    ) -> Result<(), Error> {
        let picker = Self::new(options)?;

        info!(
            "Spawning background threads: base_path={}, warmup={}, content_indexing={}, mode={:?}",
            picker.base_path.display(),
            picker.enable_mmap_cache,
            picker.enable_content_indexing,
            picker.mode,
        );

        let warmup = picker.enable_mmap_cache;
        let content_indexing = picker.enable_content_indexing;
        let watch = picker.watch;
        let mode = picker.mode;

        let signals = picker.scan_signals();
        let scanned_files_counter = picker.scanned_files_counter();
        let path = picker.base_path.clone();

        // Flip scanning=true *before* we publish the picker and spawn
        // the worker, so any caller that calls `wait_for_scan`
        // immediately after `new_with_shared_state` sees the scan as
        // in-progress. Otherwise the waiter can poll between publish
        // and worker-start, see `false`, and return early before any
        // file has been indexed.
        signals.scanning.store(true, Ordering::Release);

        {
            let mut guard = shared_picker.write()?;
            *guard = Some(picker);
        }

        ScanJob::new_initial(
            shared_picker,
            shared_frecency,
            path,
            mode,
            signals,
            scanned_files_counter,
            ScanConfig {
                warmup,
                content_indexing,
                watch,
                auto_cache_budget: true,
                install_watcher: true,
            },
        )
        .spawn();

        Ok(())
    }

    /// Synchronous filesystem scan — populates `self` with indexed files.
    ///
    /// Use this when you need direct access to the picker without shared state:
    /// ```ignore
    /// let mut picker = FilePicker::new(options)?;
    /// picker.collect_files()?;
    /// // picker.get_files() is now populated
    /// ```
    pub fn collect_files(&mut self) -> Result<(), Error> {
        self.signals.scanning.store(true, Ordering::Relaxed);
        self.scanned_files_count.store(0, Ordering::Relaxed);

        let empty_frecency = SharedFrecency::default();
        let walk = walk_filesystem(
            &self.base_path,
            &self.scanned_files_count,
            &empty_frecency,
            self.mode,
        )?;

        self.sync_data = walk.sync;

        // Recalculate cache budget based on actual file count (unless
        // the caller provided an explicit budget via FilePickerOptions).
        if !self.has_explicit_cache_budget {
            let file_count = self.sync_data.files().len();
            self.cache_budget = Arc::new(ContentCacheBudget::new_for_repo(file_count));
        } else {
            self.cache_budget.reset();
        }

        // Apply git status synchronously.
        if let Ok(Some(git_cache)) = walk.git_handle.join() {
            let arena = self.arena_base_ptr();
            for file in self.sync_data.files.iter_mut() {
                file.git_status =
                    git_cache.lookup_status(&file.absolute_path(arena, &self.base_path));
            }
        }

        self.signals.scanning.store(false, Ordering::Relaxed);
        Ok(())
    }

    /// Start the background file-system watcher.
    ///
    /// The picker must already be placed into `shared_picker` (the watcher
    /// needs the shared handle to apply live updates). Call after
    /// [`collect_files`](Self::collect_files) or after an initial scan.
    pub fn spawn_background_watcher(
        &mut self,
        shared_picker: &SharedPicker,
        shared_frecency: &SharedFrecency,
    ) -> Result<(), Error> {
        let git_workdir = self.sync_data.git_workdir.clone();
        let watcher = BackgroundWatcher::new(
            self.base_path.clone(),
            git_workdir,
            shared_picker.clone(),
            shared_frecency.clone(),
            self.mode,
        )?;
        self.background_watcher = Some(watcher);
        self.signals.watcher_ready.store(true, Ordering::Release);
        Ok(())
    }

    /// Perform fuzzy search on files with a pre-parsed query.
    ///
    /// The query should be parsed using [`FFFQuery`]::parse() before calling
    /// this function. If a [`QueryTracker`] is provided, the search will
    /// automatically look up the last selected file for this query and apply
    /// combo-boost scoring.
    ///
    pub fn fuzzy_search<'q>(
        &self,
        query: &'q FFFQuery<'q>,
        query_tracker: Option<&QueryTracker>,
        options: FuzzySearchOptions<'q>,
    ) -> SearchResult<'_> {
        let files = self.get_files();
        let max_threads = if options.max_threads == 0 {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        } else {
            options.max_threads
        };

        debug!(
            raw_query = ?query.raw_query,
            pagination = ?options.pagination,
            ?max_threads,
            current_file = ?options.current_file,
            "Fuzzy search",
        );

        let total_files = files.len();
        let location = query.location;

        // Get effective query for max_typos calculation (without location suffix)
        let effective_query = match &query.fuzzy_query {
            fff_query_parser::FuzzyQuery::Text(t) => *t,
            fff_query_parser::FuzzyQuery::Parts(parts) if !parts.is_empty() => parts[0],
            _ => query.raw_query.trim(),
        };

        // small queries with a large number of results can match absolutely everything
        let max_typos = (effective_query.len() as u16 / 4).clamp(2, 6);
        // Look up the last file selected for this query (combo-boost scoring)
        let last_same_query_entry =
            query_tracker
                .zip(options.project_path)
                .and_then(|(tracker, project_path)| {
                    tracker
                        .get_last_query_entry(
                            query.raw_query,
                            project_path,
                            options.min_combo_count,
                        )
                        .ok()
                        .flatten()
                });

        let context = ScoringContext {
            query,
            max_typos,
            max_threads,
            project_path: options.project_path,
            current_file: options.current_file,
            last_same_query_match: last_same_query_entry,
            combo_boost_score_multiplier: options.combo_boost_score_multiplier,
            min_combo_count: options.min_combo_count,
            pagination: options.pagination,
        };

        let time = std::time::Instant::now();

        let base_arena = self.sync_data.arena_base_ptr();
        let overflow_arena = self
            .sync_data
            .overflow_builder
            .as_ref()
            .map(|b| b.as_arena_ptr())
            .unwrap_or(base_arena);

        let (items, scores, total_matched) = fuzzy_match_and_score_files(
            files,
            &context,
            self.sync_data.base_count,
            base_arena,
            overflow_arena,
        );

        info!(
            ?query,
            completed_in = ?time.elapsed(),
            total_matched,
            returned_count = items.len(),
            pagination = ?options.pagination,
            "Fuzzy search completed",
        );

        SearchResult {
            items,
            scores,
            total_matched,
            total_files,
            location,
        }
    }

    /// Perform fuzzy search on indexed directories.
    ///
    /// Returns directories ranked by fuzzy match quality + frecency.
    pub fn fuzzy_search_directories<'q>(
        &self,
        query: &'q FFFQuery<'q>,
        options: FuzzySearchOptions<'q>,
    ) -> DirSearchResult<'_> {
        let dirs = self.get_dirs();
        let max_threads = if options.max_threads == 0 {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        } else {
            options.max_threads
        };

        let total_dirs = dirs.len();

        let effective_query = match &query.fuzzy_query {
            fff_query_parser::FuzzyQuery::Text(t) => *t,
            fff_query_parser::FuzzyQuery::Parts(parts) if !parts.is_empty() => parts[0],
            _ => query.raw_query.trim(),
        };

        let max_typos = (effective_query.len() as u16 / 4).clamp(2, 6);

        let context = ScoringContext {
            query,
            max_typos,
            max_threads,
            project_path: options.project_path,
            current_file: options.current_file,
            last_same_query_match: None,
            combo_boost_score_multiplier: 0,
            min_combo_count: 0,
            pagination: options.pagination,
        };

        let arena = self.sync_data.arena_base_ptr();
        let time = std::time::Instant::now();

        let (items, scores, total_matched) =
            crate::score::fuzzy_match_and_score_dirs(dirs, &context, arena);

        info!(
            ?query,
            completed_in = ?time.elapsed(),
            total_matched,
            returned_count = items.len(),
            "Directory search completed",
        );

        DirSearchResult {
            items,
            scores,
            total_matched,
            total_dirs,
        }
    }

    /// Perform a mixed fuzzy search across both files and directories.
    ///
    /// Returns a single flat list where files and directories are interleaved
    /// by total score in descending order.
    ///
    /// If the raw query ends with a path separator (`/`), only directories
    /// are searched — files are skipped entirely. The caller should parse the
    /// query with `DirSearchConfig` so that trailing `/` is kept as fuzzy
    /// text instead of becoming a `PathSegment` constraint.
    pub fn fuzzy_search_mixed<'q>(
        &self,
        query: &'q FFFQuery<'q>,
        query_tracker: Option<&QueryTracker>,
        options: FuzzySearchOptions<'q>,
    ) -> MixedSearchResult<'_> {
        let location = query.location;
        let page_offset = options.pagination.offset;
        let page_limit = if options.pagination.limit > 0 {
            options.pagination.limit
        } else {
            100
        };

        let dirs_only =
            query.raw_query.ends_with(std::path::MAIN_SEPARATOR) || query.raw_query.ends_with('/');

        // Run file search and dir search with no pagination (we merge then paginate).
        let internal_limit = page_offset.saturating_add(page_limit).saturating_mul(2);

        let dir_options = FuzzySearchOptions {
            pagination: PaginationArgs {
                offset: 0,
                limit: internal_limit,
            },
            ..options
        };
        let dir_results = self.fuzzy_search_directories(query, dir_options);

        if dirs_only {
            let total_matched = dir_results.total_matched;
            let total_dirs = dir_results.total_dirs;

            let mut merged: Vec<(MixedItemRef<'_>, Score)> =
                Vec::with_capacity(dir_results.items.len());
            for (dir, score) in dir_results.items.into_iter().zip(dir_results.scores) {
                merged.push((MixedItemRef::Dir(dir), score));
            }

            if page_offset >= merged.len() {
                return MixedSearchResult {
                    items: vec![],
                    scores: vec![],
                    total_matched,
                    total_files: self.sync_data.files().len(),
                    total_dirs,
                    location,
                };
            }

            let end = (page_offset + page_limit).min(merged.len());
            let page = merged.drain(page_offset..end);
            let (items, scores): (Vec<_>, Vec<_>) = page.unzip();

            return MixedSearchResult {
                items,
                scores,
                total_matched,
                total_files: self.sync_data.files().len(),
                total_dirs,
                location,
            };
        }

        let file_options = FuzzySearchOptions {
            pagination: PaginationArgs {
                offset: 0,
                limit: internal_limit,
            },
            ..options
        };
        let file_results = self.fuzzy_search(query, query_tracker, file_options);

        // Merge by score descending.
        let total_matched = file_results.total_matched + dir_results.total_matched;
        let total_files = file_results.total_files;
        let total_dirs = dir_results.total_dirs;

        let mut merged: Vec<(MixedItemRef<'_>, Score)> =
            Vec::with_capacity(file_results.items.len() + dir_results.items.len());

        for (file, score) in file_results.items.into_iter().zip(file_results.scores) {
            merged.push((MixedItemRef::File(file), score));
        }
        for (dir, score) in dir_results.items.into_iter().zip(dir_results.scores) {
            merged.push((MixedItemRef::Dir(dir), score));
        }

        // Sort merged results by total score descending.
        merged.sort_unstable_by_key(|b| std::cmp::Reverse(b.1.total));

        // Paginate.
        if page_offset >= merged.len() {
            return MixedSearchResult {
                items: vec![],
                scores: vec![],
                total_matched,
                total_files,
                total_dirs,
                location,
            };
        }

        let end = (page_offset + page_limit).min(merged.len());
        let page = merged.drain(page_offset..end);
        let (items, scores): (Vec<_>, Vec<_>) = page.unzip();

        MixedSearchResult {
            items,
            scores,
            total_matched,
            total_files,
            total_dirs,
            location,
        }
    }

    /// Perform a live grep search across indexed files.
    ///
    /// If `options.abort_signal` is set it overrides the picker's internal
    /// cancellation flag, giving the caller full control over when to stop.
    pub fn grep(&self, query: &FFFQuery<'_>, options: &GrepSearchOptions) -> GrepResult<'_> {
        let overlay_guard = self.sync_data.bigram_overlay.as_ref().map(|o| o.read());
        let arena = self.arena_base_ptr();
        let overflow_arena = self.sync_data.overflow_arena_ptr();
        let cancel = options
            .abort_signal
            .as_deref()
            .unwrap_or(&self.signals.cancelled);

        grep_search(
            self.get_files(),
            query,
            options,
            self.cache_budget(),
            self.sync_data.bigram_index.as_deref(),
            overlay_guard.as_deref(),
            cancel,
            &self.base_path,
            arena,
            overflow_arena,
        )
    }

    /// Multi-pattern grep search across indexed files.
    pub fn multi_grep(
        &self,
        patterns: &[&str],
        constraints: &[fff_query_parser::Constraint<'_>],
        options: &GrepSearchOptions,
    ) -> GrepResult<'_> {
        let overlay_guard = self.sync_data.bigram_overlay.as_ref().map(|o| o.read());
        let arena = self.arena_base_ptr();
        let overflow_arena = self.sync_data.overflow_arena_ptr();
        let cancel = options
            .abort_signal
            .as_deref()
            .unwrap_or(&self.signals.cancelled);

        multi_grep_search(
            self.get_files(),
            patterns,
            constraints,
            options,
            self.cache_budget(),
            self.sync_data.bigram_index.as_deref(),
            overlay_guard.as_deref(),
            cancel,
            &self.base_path,
            arena,
            overflow_arena,
        )
    }

    /// Like [`grep`](Self::grep) but ignores the bigram overlay.
    pub fn grep_without_overlay(
        &self,
        query: &FFFQuery<'_>,
        options: &GrepSearchOptions,
    ) -> GrepResult<'_> {
        let arena = self.arena_base_ptr();
        let overflow_arena = self.sync_data.overflow_arena_ptr();
        let cancel = options
            .abort_signal
            .as_deref()
            .unwrap_or(&self.signals.cancelled);

        grep_search(
            self.get_files(),
            query,
            options,
            self.cache_budget(),
            self.sync_data.bigram_index.as_deref(),
            None,
            cancel,
            &self.base_path,
            arena,
            overflow_arena,
        )
    }

    // Returns an ongoing or finisshed scan progress
    pub fn get_scan_progress(&self) -> ScanProgress {
        let scanned_count = self.scanned_files_count.load(Ordering::Relaxed);
        let is_scanning = self.signals.scanning.load(Ordering::Relaxed);
        ScanProgress {
            scanned_files_count: scanned_count,
            is_scanning,
            is_watcher_ready: self.signals.watcher_ready.load(Ordering::Relaxed),
            is_warmup_complete: self.sync_data.bigram_index.is_some(),
        }
    }

    /// Update git statuses for files, using the provided shared frecency tracker.
    pub fn update_git_statuses(
        &mut self,
        status_cache: GitStatusCache,
        shared_frecency: &SharedFrecency,
    ) -> Result<(), Error> {
        debug!(
            statuses_count = status_cache.statuses_len(),
            "Updating git status",
        );

        let mode = self.mode;
        let bp = self.base_path.clone();
        let arena = self.arena_base_ptr();
        let frecency = shared_frecency.read()?;
        status_cache
            .into_iter()
            .try_for_each(|(path, status)| -> Result<(), Error> {
                if let Some(file) = self.get_mut_file_by_path(&path) {
                    file.git_status = Some(status);
                    if let Some(ref f) = *frecency {
                        file.update_frecency_scores(f, arena, &bp, mode)?;
                    }
                    // Update parent dir frecency inline.
                    let score = file.access_frecency_score as i32;
                    let dir_idx = file.parent_dir_index() as usize;
                    if let Some(dir) = self.sync_data.dirs.get_mut(dir_idx) {
                        dir.update_frecency_if_larger(score);
                    }
                } else {
                    error!(?path, "Couldn't update the git status for path");
                }
                Ok(())
            })?;

        Ok(())
    }

    pub fn update_single_file_frecency(
        &mut self,
        file_path: impl AsRef<Path>,
        frecency_tracker: &FrecencyTracker,
    ) -> Result<(), Error> {
        let path = file_path.as_ref();
        let arena = self.arena_base_ptr();
        let rel = self.to_relative_path(path).unwrap_or("");
        let index = self
            .sync_data
            .find_file_index(path, &self.base_path)
            .ok()
            .or_else(|| self.sync_data.find_overflow_index(rel));
        if let Some(index) = index
            && let Some(file) = self.sync_data.get_file_mut(index)
        {
            file.update_frecency_scores(frecency_tracker, arena, &self.base_path, self.mode)?;

            // Update parent dir frecency inline (only if larger).
            let score = file.access_frecency_score as i32;
            let dir_idx = file.parent_dir_index() as usize;
            if let Some(dir) = self.sync_data.dirs.get_mut(dir_idx) {
                dir.update_frecency_if_larger(score);
            }
        }

        Ok(())
    }

    pub fn get_file_by_path(&self, path: impl AsRef<Path>) -> Option<&FileItem> {
        self.sync_data
            .find_file_index(path.as_ref(), &self.base_path)
            .ok()
            .and_then(|index| self.sync_data.files().get(index))
    }

    pub fn get_mut_file_by_path(&mut self, path: impl AsRef<Path>) -> Option<&mut FileItem> {
        let path = path.as_ref();
        let rel = self.to_relative_path(path).unwrap_or("");
        let index = self
            .sync_data
            .find_file_index(path, &self.base_path)
            .ok()
            .or_else(|| self.sync_data.find_overflow_index(rel));
        index.and_then(|i| self.sync_data.get_file_mut(i))
    }

    /// Add a file to the picker's files in sorted order (used by background watcher)
    pub fn add_file_sorted(&mut self, file: FileItem) -> Option<&FileItem> {
        let arena = self.arena_base_ptr();
        let path = file.absolute_path(arena, &self.base_path);

        if self.sync_data.insert_file_sorted(file, &self.base_path) {
            // File was inserted, look it up
            self.sync_data
                .find_file_index(&path, &self.base_path)
                .ok()
                .and_then(|idx| self.sync_data.get_file_mut(idx))
                .map(|file_mut| &*file_mut) // Convert &mut to &
        } else {
            // File already exists
            warn!(
                "Trying to insert a file that already exists: {}",
                path.display()
            );
            self.sync_data
                .find_file_index(&path, &self.base_path)
                .ok()
                .and_then(|idx| self.sync_data.get_file_mut(idx))
                .map(|file_mut| &*file_mut) // Convert &mut to &
        }
    }

    #[tracing::instrument(skip(self), name = "timing_update", level = Level::DEBUG)]
    pub fn on_create_or_modify(&mut self, path: impl AsRef<Path> + Debug) -> Option<&FileItem> {
        let path = path.as_ref();
        let overlay = self.sync_data.bigram_overlay.as_ref().map(Arc::clone);

        if let Ok(pos) = self.sync_data.find_file_index(path, &self.base_path) {
            let file = self.sync_data.get_file_mut(pos)?;

            if file.is_deleted() {
                // Resurrect tombstoned file.
                file.set_deleted(false);
                debug!(
                    "on_create_or_modify: resurrected tombstoned file at index {}",
                    pos
                );
            }

            debug!(
                "on_create_or_modify: file EXISTS at index {}, updating metadata",
                pos
            );

            let modified = match std::fs::metadata(path) {
                Ok(metadata) => metadata
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok()),
                Err(e) => {
                    error!("Failed to get metadata for {}: {}", path.display(), e);
                    None
                }
            };

            if let Some(modified) = modified {
                let modified = modified.as_secs();
                if file.modified < modified {
                    file.modified = modified;
                    file.invalidate_mmap(&self.cache_budget);
                }
            }

            // Update the bigram overlay for this modified file — but only
            // if it lives in the indexable region. The overlay's bitsets are
            // sized to `base_file_count == indexable_count`; modifying a
            // file past that boundary would write a key the filter can't
            // resolve back to a column.
            if let Some(ref overlay) = overlay {
                let in_indexable = {
                    let guard = overlay.read();
                    pos < guard.base_file_count()
                };
                if in_indexable && let Ok(content) = std::fs::read(path) {
                    overlay.write().modify_file(pos, &content);
                }
            }

            return Some(&*file);
        }

        // Check overflow for existing added files.
        let rel_path = self.to_relative_path(path).unwrap_or("");
        if let Some(abs_idx) = self.sync_data.find_overflow_index(rel_path) {
            let file = self.sync_data.get_file_mut(abs_idx)?;
            let modified = std::fs::metadata(path)
                .ok()
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok());
            if let Some(modified) = modified {
                let modified = modified.as_secs();
                if file.modified < modified {
                    file.modified = modified;
                    file.invalidate_mmap(&self.cache_budget);
                }
            }
            return Some(&*file);
        }

        // New file — append to overflow (preserves base indices for bigram).
        debug!(
            "on_create_or_modify: file NEW, appending to overflow (base: {}, overflow: {})",
            self.sync_data.base_count,
            self.sync_data.overflow_files().len(),
        );

        let (mut file_item, rel_path) = FileItem::new(path.to_path_buf(), &self.base_path, None);

        // Lazily create the shared overflow builder on first use.
        let builder = self
            .sync_data
            .overflow_builder
            .get_or_insert_with(|| crate::simd_path::ChunkedPathStoreBuilder::new(64));

        let cs = builder.add_file_immediate(&rel_path, file_item.path.filename_offset);
        file_item.set_path(cs);
        file_item.set_overflow(true);
        self.sync_data.files.push(file_item);
        self.sync_data.files.last()
    }

    /// Tombstone a file instead of removing it, keeping base indices stable.
    pub fn remove_file_by_path(&mut self, path: impl AsRef<Path>) -> bool {
        let path = path.as_ref();
        match self.sync_data.find_file_index(path, &self.base_path) {
            Ok(index) => {
                let file = &mut self.sync_data.files[index];
                file.set_deleted(true);
                // Clear any cached git status — the tombstone no longer
                // corresponds to a real worktree file, so any previously
                // cached status (e.g. `WT_MODIFIED` from before the
                // delete) is actively misleading. All user-facing search
                // paths filter `is_deleted()` so this is invisible today,
                // but keeping the invariant "tombstone ⇒ git_status=None"
                // means a new reader that forgets the filter can't leak
                // stale data.
                file.git_status = None;
                file.invalidate_mmap(&self.cache_budget);
                if let Some(ref overlay) = self.sync_data.bigram_overlay {
                    overlay.write().delete_file(index);
                }
                true
            }
            Err(_) => {
                // Check overflow for added files — these can be removed directly
                // since they aren't in the base bigram index.
                let rel = self.to_relative_path(path).unwrap_or("");
                if let Some(abs_pos) = self.sync_data.find_overflow_index(rel) {
                    self.sync_data.files.remove(abs_pos);
                    true
                } else {
                    false
                }
            }
        }
    }

    // TODO make this O(n)
    pub fn remove_all_files_in_dir(&mut self, dir: impl AsRef<Path>) -> usize {
        let dir_path = dir.as_ref();
        let relative_dir = self.to_relative_path(dir_path).unwrap_or("").to_string();

        let dir_prefix = if relative_dir.is_empty() {
            String::new()
        } else {
            format!("{}{}", relative_dir, std::path::MAIN_SEPARATOR)
        };

        self.sync_data.retain_files_with_arena(|file, arena| {
            !file.relative_path_starts_with(arena, &dir_prefix)
        })
    }

    /// Use this to prevent any substantial background threads from acquiring the locks
    pub fn cancel(&self) {
        self.signals.cancelled.store(true, Ordering::Release);
    }

    pub fn stop_background_monitor(&mut self) {
        if let Some(mut watcher) = self.background_watcher.take() {
            watcher.stop();
        }
    }

    /// Take the background watcher without joining its owner thread. The
    /// caller is responsible for dropping the returned value, which joins
    /// the owner thread via the watcher's `Drop` impl.
    ///
    /// Use this from code paths that hold a lock the watcher's threads
    /// may try to re-acquire (e.g. the global `FILE_PICKER` write lock).
    /// Calling the blocking `stop()` under such a lock deadlocks: the
    /// watcher's debounced-event handler or owner thread may itself try
    /// to `.write()` the picker, and `handle.join()` will wait forever
    /// for a guard the caller is still holding.
    ///
    /// Returns an opaque type-erased value; all the caller needs to know
    /// is to `drop()` it outside the lock.
    #[must_use = "dropping the returned handle outside the lock is required to avoid deadlock"]
    pub fn detach_background_monitor(&mut self) -> Option<Box<dyn Send + 'static>> {
        self.background_watcher
            .take()
            .map(|w| Box::new(w) as Box<dyn Send + 'static>)
    }

    #[inline]
    pub(crate) fn arena_base_ptr(&self) -> ArenaPtr {
        self.sync_data.arena_base_ptr()
    }

    pub fn trigger_rescan(&mut self, shared_frecency: &SharedFrecency) -> Result<(), Error> {
        if self.signals.scanning.load(Ordering::Relaxed) {
            debug!("Scan already in progress, skipping trigger_rescan");
            return Ok(());
        }

        // The post-scan warmup + bigram phase holds a raw pointer into the
        // current files Vec. Replacing sync_data now would free that memory.
        // Skip — the background watcher will retry on the next event.
        if self.signals.post_scan_busy.load(Ordering::Acquire) {
            debug!("Post-scan bigram build in progress, skipping rescan");
            return Ok(());
        }

        self.signals.scanning.store(true, Ordering::Relaxed);
        self.scanned_files_count.store(0, Ordering::Relaxed);

        let walk_result = walk_filesystem(
            &self.base_path,
            &self.scanned_files_count,
            shared_frecency,
            self.mode,
        );

        match walk_result {
            Ok(walk) => {
                eprintln!(
                    "TRIGGER_RESCAN: walk produced {} files, base_count={}, indexable_count={}",
                    walk.sync.files.len(),
                    walk.sync.base_count,
                    walk.sync.indexable_count
                );
                eprintln!(
                    "TRIGGER_RESCAN: BEFORE replace: files={}, base_count={}",
                    self.sync_data.files.len(),
                    self.sync_data.base_count
                );
                info!(
                    "Filesystem rescan completed: found {} files",
                    walk.sync.files.len()
                );

                self.sync_data = walk.sync;
                self.cache_budget.reset();
                eprintln!(
                    "TRIGGER_RESCAN: AFTER replace: files={}, base_count={}",
                    self.sync_data.files.len(),
                    self.sync_data.base_count
                );

                // Apply git status synchronously for rescan (typically fast).
                if let Ok(Some(git_cache)) = walk.git_handle.join() {
                    let frecency = shared_frecency.read().ok();
                    let frecency_ref = frecency.as_ref().and_then(|f| f.as_ref());
                    let mode = self.mode;
                    let bp = &self.base_path;
                    let arena = self.arena_base_ptr();

                    // Reset dir frecency before recomputation.
                    for dir in self.sync_data.dirs.iter() {
                        dir.reset_frecency();
                    }

                    let files = &mut self.sync_data.files;
                    let dirs = &self.sync_data.dirs;
                    BACKGROUND_THREAD_POOL.install(|| {
                        files.par_iter_mut().for_each(|file| {
                            file.git_status =
                                git_cache.lookup_status(&file.absolute_path(arena, bp));
                            if let Some(frecency) = frecency_ref {
                                let _ = file.update_frecency_scores(frecency, arena, bp, mode);
                            }
                            let score = file.access_frecency_score as i32;
                            if score > 0 {
                                let dir_idx = file.parent_dir_index() as usize;
                                if let Some(dir) = dirs.get(dir_idx) {
                                    dir.update_frecency_if_larger(score);
                                }
                            }
                        });
                    });
                }

                // Warmup is deferred to the post-rescan bigram rebuild thread
                // (spawned by trigger_full_rescan) which does warmup + bigram
                // in one pass, matching the initial scan's post-scan phase.
            }
            Err(error) => error!(?error, "Failed to scan file system"),
        }

        self.signals.scanning.store(false, Ordering::Relaxed);
        Ok(())
    }

    /// Quick way to check if scan is going without acquiring a lock for [Self::get_scan_progress]
    pub fn is_scan_active(&self) -> bool {
        self.signals.scanning.load(Ordering::Relaxed)
    }

    /// Return a clone of the scanning flag so callers can poll it without
    /// holding a lock on the picker.
    pub fn scan_signal(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.signals.scanning)
    }

    /// Return a clone of the watcher-ready flag so callers can poll it without
    /// holding a lock on the picker.
    pub fn watcher_signal(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.signals.watcher_ready)
    }
}

/// A point-in-time snapshot of the file-scanning progress.
///
/// Returned by [`FilePicker::get_scan_progress`]. Useful for displaying
/// a progress indicator while the initial scan is running.
#[derive(Debug, Clone)]
pub struct ScanProgress {
    pub scanned_files_count: usize,
    pub is_scanning: bool,
    pub is_watcher_ready: bool,
    pub is_warmup_complete: bool,
}

/// Pre-populate mmap caches for the most valuable files so the first grep
/// search doesn't pay the mmap creation + page fault cost.
///
/// All files are collected once, then an O(n) `select_nth_unstable_by`
/// partitions the top [`MAX_CACHED_CONTENT_FILES`] highest-frecency eligible
/// files to the front (binary / empty files are pushed to the end by the
/// comparator). The selected prefix is warmed in parallel via rayon.
///
/// Files beyond the budget are still available via temporary mmaps on first
/// grep access, so correctness is unaffected.
#[tracing::instrument(skip(files), name = "warmup_mmaps", level = Level::DEBUG)]
pub(crate) fn warmup_mmaps(
    files: &[FileItem],
    budget: &ContentCacheBudget,
    base_path: &Path,
    arena: ArenaPtr,
) {
    let max_files = budget.max_files;
    let max_bytes = budget.max_bytes;
    let max_file_size = budget.max_file_size;

    // Single collect — no pre-filter. The comparator in select_nth pushes
    // ineligible files (binary, empty) to the tail automatically.
    let mut all: Vec<&FileItem> = files.iter().collect();

    // O(n) partial sort: top max_files eligible-by-frecency files land in
    // all[..max_files]. Ineligible files compare as "lowest priority" so
    // they naturally sink past the partition boundary.
    if all.len() > max_files {
        all.select_nth_unstable_by(max_files, |a, b| {
            let a_ok = !a.is_binary() && a.size > 0;
            let b_ok = !b.is_binary() && b.size > 0;
            match (a_ok, b_ok) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                (false, false) => std::cmp::Ordering::Equal,
                (true, true) => b.total_frecency_score().cmp(&a.total_frecency_score()),
            }
        });
    }

    let to_warm = &all[..all.len().min(max_files)];

    let warmed_bytes = AtomicU64::new(0);
    let budget_exhausted = AtomicBool::new(false);

    BACKGROUND_THREAD_POOL.install(|| {
        to_warm.par_iter().for_each(|file| {
            if budget_exhausted.load(Ordering::Relaxed) {
                return;
            }

            if file.is_binary() || file.size == 0 || file.size > max_file_size {
                return;
            }

            // Byte budget.
            let prev_bytes = warmed_bytes.fetch_add(file.size, Ordering::Relaxed);
            if prev_bytes + file.size > max_bytes {
                budget_exhausted.store(true, Ordering::Relaxed);
                return;
            }

            if let Some(content) = file.get_content(arena, base_path, budget) {
                let _ = std::hint::black_box(content.first());
            }
        });
    });
}

/// Result of the fast walk phase — files are searchable immediately,
/// git status arrives later via the join handle.
pub(crate) struct WalkResult {
    pub(crate) sync: FileSync,
    pub(crate) git_handle: std::thread::JoinHandle<Option<GitStatusCache>>,
}

/// Returns files immediately (searchable) and a handle to the in-progress
/// git status computation. This avoids blocking on `git status` which can
/// take 10+ seconds on very large repos (e.g. chromium).
pub(crate) fn walk_filesystem(
    base_path: &Path,
    synced_files_count: &Arc<AtomicUsize>,
    shared_frecency: &SharedFrecency,
    mode: FFFMode,
) -> Result<WalkResult, Error> {
    use ignore::WalkBuilder;

    let scan_start = std::time::Instant::now();
    info!("SCAN: Starting filesystem walk and git status (async)");

    // Discover git root (fast — just walks up looking for .git/)
    let git_workdir = Repository::discover(base_path)
        .ok()
        .and_then(|repo| repo.workdir().map(Path::to_path_buf));

    if let Some(ref git_dir) = git_workdir {
        debug!("Git repository found at: {}", git_dir.display());
    } else {
        debug!("No git repository found for path: {}", base_path.display());
    }

    // Spawn git status on a detached thread — we won't wait for it here.
    let git_workdir_for_status = git_workdir.clone();
    let git_handle = std::thread::spawn(move || {
        GitStatusCache::read_git_status(
            git_workdir_for_status.as_deref(),
            &mut crate::git::default_status_options(),
        )
    });

    // Walk files (the fast part, typically 2-3s even on huge repos).
    let is_git_repo = git_workdir.is_some();
    let bg_threads = BACKGROUND_THREAD_POOL.current_num_threads();

    let mut walk_builder = WalkBuilder::new(base_path);
    walk_builder
        // this is a very important guard for the user opening ~/ or other root non-git dir
        .hidden(!is_git_repo)
        .git_ignore(true)
        .git_exclude(true)
        .git_global(true)
        .ignore(true)
        .follow_links(false)
        .threads(bg_threads);

    if !is_git_repo && let Some(overrides) = non_git_repo_overrides(base_path) {
        walk_builder.overrides(overrides);
    }

    let walker = walk_builder.build_parallel();
    let walker_start = std::time::Instant::now();
    debug!("SCAN: Starting file walker");

    // Walk: collect (FileItem, rel_path) pairs. Keep the walk fast —
    // no chunking, no HashMap, just Vec::push under the Mutex.
    let pairs = parking_lot::Mutex::new(Vec::<(FileItem, String)>::new());

    walker.run(|| {
        let pairs = &pairs;
        let counter = Arc::clone(synced_files_count);
        let base_path = base_path.to_path_buf();

        Box::new(move |result| {
            let Ok(entry) = result else {
                return ignore::WalkState::Continue;
            };

            if entry.file_type().is_some_and(|ft| ft.is_file()) {
                let path = entry.path();

                // Ignore walkers sometimes surface files inside `.git/`
                // when the base is itself a git repo — skip them.
                if is_git_file(path) {
                    return ignore::WalkState::Continue;
                }

                if !is_git_repo && is_known_binary_extension(path) {
                    return ignore::WalkState::Continue;
                }

                let metadata = entry.metadata().ok();
                let (file_item, rel_path) =
                    FileItem::new_from_walk(path, &base_path, None, metadata.as_ref());

                pairs.lock().push((file_item, rel_path));
                counter.fetch_add(1, Ordering::Relaxed);
            }
            ignore::WalkState::Continue
        })
    });

    let mut pairs = pairs.into_inner();

    info!(
        "SCAN: File walking completed in {:?} for {} files",
        walker_start.elapsed(),
        pairs.len(),
    );

    // Sort by (dir_part, filename). This groups files by their directory
    // into contiguous runs so the linear dir-extraction pass below can
    // dedupe by comparing only against the previous dir.
    //
    // An earlier version sorted by full relative path. That ordering works
    // for uniform depths but breaks for mixed trees: root-level files with
    // names that sort between two subdirectories end up interleaved,
    // causing the consecutive-dedup to emit the root dir multiple times
    // (e.g. seed order `README.md`, `d_foo/a.rs`, `f_root.rs`, `src/x.rs`
    // produced `dirs = ["", "d_foo/", "", "src/"]`). The duplicate broke
    // `find_file_index`'s binary search — files with `parent_dir=0` became
    // unreachable when the search resolved `dir_rel=""` to the duplicate
    // at index 2.
    BACKGROUND_THREAD_POOL.install(|| {
        pairs.par_sort_unstable_by(|(a, path_a), (b, path_b)| {
            let a_off = a.path.filename_offset as usize;
            let b_off = b.path.filename_offset as usize;
            // SAFETY note: `filename_offset` is a byte offset into the
            // UTF-8 relative path that the walker recorded. It always
            // lies on a character boundary.
            let (a_dir, a_name) = path_a.split_at(a_off);
            let (b_dir, b_name) = path_b.split_at(b_off);
            a_dir.cmp(b_dir).then_with(|| a_name.cmp(b_name))
        });
    });

    // Build ChunkedPathStore + extract dirs + assign parent_dir in one pass.
    // Files are sorted by relative path, so dir changes happen in order.
    // add_file_immediate returns a ChunkedString with null arena_base;
    // we fixup arena_base after the arena is frozen.
    let mut files: Vec<FileItem> = Vec::with_capacity(pairs.len());
    let mut dirs: Vec<DirItem> = Vec::new();
    let mut builder = crate::simd_path::ChunkedPathStoreBuilder::new(pairs.len());
    // Reusable buffer for the previously emitted dir path. `prev_dir_valid`
    // tracks whether `prev_dir` has been populated (so the first iteration
    // always enters the dir branch, even if `dir_part == ""`).
    let mut prev_dir: String = String::new();
    let mut prev_dir_valid: bool = false;
    let mut current_dir_idx: u32 = 0;

    for (mut file, rel) in pairs {
        let fname_offset = file.path.filename_offset as usize;
        let dir_part = &rel[..fname_offset];

        if !prev_dir_valid || prev_dir.as_str() != dir_part {
            let dir_cs = builder.add_dir_immediate(dir_part);
            // Compute last-segment offset: for "src/components/" -> 4 (points to "components/")
            let last_seg = if dir_part.is_empty() {
                0
            } else {
                let trimmed = dir_part.trim_end_matches(std::path::is_separator);
                trimmed
                    .rfind(std::path::is_separator)
                    .map(|i| i + 1)
                    .unwrap_or(0) as u16
            };
            dirs.push(DirItem::new(dir_cs, last_seg));
            current_dir_idx = (dirs.len() - 1) as u32;
            prev_dir.clear();
            prev_dir.push_str(dir_part);
            prev_dir_valid = true;
        }

        let cs = builder.add_file_immediate(&rel, file.path.filename_offset);
        file.set_path(cs);
        file.set_parent_dir(current_dir_idx);
        files.push(file);
    }
    let chunked_paths = builder.finish();
    let arena = chunked_paths.as_arena_ptr();

    // Apply frecency scores (access-based only — git status not yet available).
    // DirItem.max_access_frecency is AtomicI32, so parallel threads write directly.
    let frecency = shared_frecency
        .read()
        .map_err(|_| Error::AcquireFrecencyLock)?;
    if let Some(frecency) = frecency.as_ref() {
        let dirs_ref = &dirs;
        BACKGROUND_THREAD_POOL.install(|| {
            files.par_iter_mut().for_each(|file| {
                let _ = file.update_frecency_scores(frecency, arena, base_path, mode);
                let score = file.access_frecency_score as i32;
                if score > 0 {
                    let dir_idx = file.parent_dir_index() as usize;
                    if let Some(dir) = dirs_ref.get(dir_idx) {
                        dir.update_frecency_if_larger(score);
                    }
                }
            });
        });
    }
    drop(frecency);

    // Re-sort by (indexable-first, parent_dir, filename). Indexable base
    // files come first so the bigram builder can size its column bitsets to
    // just the indexable subset. Within each partition files stay sorted by
    // (parent_dir, filename) — `find_file_index` does two binary searches
    // (one per partition) to preserve O(log n) lookups.
    //
    // "Indexable" = can possibly contribute bigrams: not binary-by-extension,
    // non-zero size, not larger than the bigram/mmap cap. The cap matches
    // `ContentCacheBudget::max_file_size` default (10 MB) — any file above
    // that is skipped by `build_bigram_index` anyway.
    const BIGRAM_ELIGIBLE_MAX_SIZE: u64 = 10 * 1024 * 1024;
    let is_indexable =
        |f: &FileItem| !f.is_binary() && f.size > 0 && f.size <= BIGRAM_ELIGIBLE_MAX_SIZE;
    BACKGROUND_THREAD_POOL.install(|| {
        files.par_sort_unstable_by(|a, b| {
            // Sort indexables first (true < false when we invert with !).
            (!is_indexable(a))
                .cmp(&!is_indexable(b))
                .then_with(|| a.parent_dir_index().cmp(&b.parent_dir_index()))
                .then_with(|| a.file_name(arena).cmp(&b.file_name(arena)))
        });
    });
    let indexable_count = files.partition_point(is_indexable);

    // Ask the allocator to return freed pages to the OS.
    hint_allocator_collect();

    let file_item_size = std::mem::size_of::<FileItem>();
    let files_vec_bytes = files.len() * file_item_size;
    let dir_table_bytes = dirs.len() * std::mem::size_of::<DirItem>()
        + dirs
            .iter()
            .map(|d| d.relative_path(arena).len())
            .sum::<usize>();

    let total_time = scan_start.elapsed();
    info!(
        "SCAN: Walk completed in {:?} ({} files, {} dirs, \
         chunked_store={:.2}MB, files_vec={:.2}MB, dirs={:.2}MB, FileItem={}B)",
        total_time,
        files.len(),
        dirs.len(),
        chunked_paths.heap_bytes() as f64 / 1_048_576.0,
        files_vec_bytes as f64 / 1_048_576.0,
        dir_table_bytes as f64 / 1_048_576.0,
        file_item_size,
    );

    let base_count = files.len();

    Ok(WalkResult {
        sync: FileSync {
            files,
            indexable_count,
            base_count,
            dirs,
            overflow_builder: None,
            git_workdir,
            bigram_index: None,
            bigram_overlay: None,
            chunked_paths: Some(chunked_paths),
        },
        git_handle,
    })
}

pub(crate) fn apply_git_status_and_frecency(
    shared_picker: &SharedPicker,
    shared_frecency: &SharedFrecency,
    git_handle: std::thread::JoinHandle<Option<GitStatusCache>>,
    mode: FFFMode,
) {
    let join_start = std::time::Instant::now();
    let git_cache = match git_handle.join() {
        Ok(cache) => cache,
        Err(_) => {
            error!("Git status thread panicked");
            return;
        }
    };
    info!("SCAN: Git status ready in {:?}", join_start.elapsed());

    let Some(git_cache) = git_cache else { return };

    // Take a snapshot of the raw pointers + metadata under a brief read
    // lock, then drop the guard BEFORE running the rayon loop. Previously
    // we held the picker write lock for the whole loop (multi-second:
    // 500k files × LMDB read per file), which froze every FFI caller on
    // the main nvim thread — searches, progress polls, BufEnter tracking.
    //
    // SAFETY: the scan thread is the only writer that replaces
    // `sync_data` during post-scan. `post_scan_busy` blocks rescans; git
    // status is applied exactly once per scan, on this thread, before
    // anything else races for the files slice. The overflow watcher only
    // appends to `files[base_count..]` — we only mutate `files[..]` here,
    // and we deliberately do not touch or dereference the overflow tail.
    #[allow(clippy::type_complexity)]
    let snapshot: Option<(
        *mut FileItem,
        usize,
        *const crate::types::DirItem,
        usize,
        PathBuf,
        ArenaPtr,
    )> = shared_picker.read().ok().and_then(|guard| {
        guard.as_ref().map(|picker| {
            let files = &picker.sync_data.files;
            let dirs = &picker.sync_data.dirs;
            (
                files.as_ptr() as *mut FileItem,
                files.len(),
                dirs.as_ptr(),
                dirs.len(),
                picker.base_path.clone(),
                picker.arena_base_ptr(),
            )
        })
    });

    let Some((files_ptr, files_len, dirs_ptr, dirs_len, bp, arena)) = snapshot else {
        return;
    };

    let frecency = shared_frecency.read().ok();
    let frecency_ref = frecency.as_ref().and_then(|f| f.as_ref());

    // SAFETY: the scan thread is the sole replacer of `sync_data`; it
    // won't run again until this function returns. See module-level
    // comments on the post-scan phase for the full ordering argument.
    let files: &mut [FileItem] = unsafe { std::slice::from_raw_parts_mut(files_ptr, files_len) };
    let dirs: &[crate::types::DirItem] = unsafe { std::slice::from_raw_parts(dirs_ptr, dirs_len) };

    // Reset dir frecency before recomputation.
    for dir in dirs.iter() {
        dir.reset_frecency();
    }

    BACKGROUND_THREAD_POOL.install(|| {
        files.par_iter_mut().for_each(|file| {
            let mut buf = [0u8; crate::simd_path::PATH_BUF_SIZE];
            let absolute_path = file.write_absolute_path(arena, &bp, &mut buf);

            file.git_status = git_cache.lookup_status(absolute_path);
            if let Some(frecency) = frecency_ref {
                let _ = file.update_frecency_scores(frecency, arena, &bp, mode);
            }

            let score = file.access_frecency_score as i32;
            if score > 0 {
                let dir_idx = file.parent_dir_index() as usize;
                if let Some(dir) = dirs.get(dir_idx) {
                    dir.update_frecency_if_larger(score);
                }
            }
        });
    });
    drop(frecency);

    info!(
        "SCAN: Applied git status to {} files ({} dirty)",
        files_len,
        git_cache.statuses_len(),
    );
}

/// Fast extension-based binary detection. Avoids opening files during scan.
/// Covers the vast majority of binary files in typical repositories.
#[inline]
fn is_known_binary_extension(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
        return false;
    };

    matches!(
        ext,
        // Images
        "png" | "jpg" | "jpeg" | "gif" | "bmp" | "ico" | "webp" | "tiff" | "tif" | "avif" |
        "heic" | "psd" | "icns" | "cur" | "raw" | "cr2" | "nef" | "dng" |
        // Video/Audio
        "mp4" | "avi" | "mov" | "wmv" | "mkv" | "mp3" | "wav" | "flac" | "ogg" | "m4a" |
        "aac" | "webm" | "flv" | "mpg" | "mpeg" | "wma" | "opus" | "pcm" | "reapeaks" |
        // Compressed/Archives
        "zip" | "tar" | "gz" | "bz2" | "xz" | "7z" | "rar" | "zst" | "lz4" | "lzma" |
        "cab" | "cpio" | "jsonlz4" |
        // Packages/Installers
        "deb" | "rpm" | "apk" | "dmg" | "msi" | "iso" | "nupkg" | "whl" | "egg" |
        "snap" | "appimage" | "flatpak" | "crx" | "pak" |
        // Executables/Libraries
        "exe" | "dll" | "so" | "dylib" | "o" | "a" | "lib" | "bin" | "elf" |
        // Documents
        "pdf" | "doc" | "docx" | "xls" | "xlsx" | "ppt" | "pptx" |
        // Databases
        "db" | "sqlite" | "sqlite3" | "mdb" |
        // SQLite / LevelDB auxiliary files
        "sqlite-wal" | "sqlite-shm" | "sqlite3-wal" | "sqlite3-shm" |
        "db-wal" | "db-shm" | "ldb" |
        // Fonts
        "ttf" | "otf" | "woff" | "woff2" | "eot" |
        // Compiled/Runtime
        "class" | "pyc" | "pyo" | "wasm" | "dex" | "jar" | "war" |
        // OCaml / Swift / Objective-C build artefacts
        "cmi" | "cmt" | "cmti" | "cmx" | "cof" | "cot" | "cop" | "nib" |
        "swiftdeps" | "swiftdeps~" | "swiftdoc" | "swiftmodule" | "swiftsourceinfo" |
        // ML/Data Science
        "npy" | "npz" | "pkl" | "pickle" | "h5" | "hdf5" | "pt" | "pth" | "onnx" |
        "safetensors" | "tfrecord" |
        // 3D/Game
        "glb" | "fbx" | "blend" | "blp" | "tga" |
        // Game engines / Unity-Unreal side-files
        "meta" | "dat" | "tfx" | "dia" | "journal" | "toc" | "thm" | "pfl" |
        "shadow" | "scan" | "flm" | "bcmap" | "userinfo" |
        // Data/serialized
        "parquet" | "arrow" | "pb" |
        // IDE/OS metadata
        "DS_Store" | "suo"
    )
}

/// Detect binary content by checking for NUL bytes in the first 512 bytes.
/// Called lazily when file content is first loaded, not during initial scan.
#[inline]
pub(crate) fn detect_binary_content(content: &[u8]) -> bool {
    let check_len = content.len().min(512);
    content[..check_len].contains(&0)
}

/// Length of the longest shared directory prefix of two relative dir
/// paths (without a trailing separator), measured as the number of bytes
/// up to and including the last shared separator — plus the full shorter
/// path when it is itself a directory prefix of the longer one.
///
/// Examples:
///   `"src/components"` vs `"src/routes"`   → 4  (`"src/"` emitted once)
///   `"lib/deep/nested"` vs `"lib/deep"`   → 8  (`"lib/deep"` is a prefix)
///   `"lib/deep"` vs `"lib/deeper"`        → 4  (only `"lib/"` is shared)
///   `"lib"` vs `"src"`                    → 0
///
/// Used by [`FilePicker::for_each_watch_dir`] to avoid re-emitting
/// ancestors that were already yielded for the previous (sorted) sibling.
fn common_dir_prefix_len(a: &str, b: &str) -> usize {
    let max = a.len().min(b.len());
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    let mut last_sep = 0;
    let mut i = 0;
    while i < max && a_bytes[i] == b_bytes[i] {
        if std::path::is_separator(a_bytes[i] as char) {
            last_sep = i + 1;
        }
        i += 1;
    }
    // If one string is a prefix of the other and the next byte in the
    // longer one is a separator, the full shorter path is a shared dir.
    if i == max && i > 0 {
        let longer = if a.len() > b.len() { a_bytes } else { b_bytes };
        if i < longer.len() && std::path::is_separator(longer[i] as char) {
            return i;
        }
    }
    last_sep
}

/// Ask the global allocator to return freed pages to the OS.
/// Enabled via the `mimalloc-collect` feature (set by fff-nvim).
/// No-op when the feature is off (tests, system allocator).
pub(crate) fn hint_allocator_collect() {
    #[cfg(feature = "mimalloc-collect")]
    {
        // Collect BACKGROUND_THREAD_POOL workers — that's where the bigram
        // builder allocated memory. `rayon::broadcast` would target the global
        // pool, which is the wrong set of threads.
        BACKGROUND_THREAD_POOL.broadcast(|_| unsafe { libmimalloc_sys::mi_collect(true) });

        // Main thread too.
        unsafe { libmimalloc_sys::mi_collect(true) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::SharedFrecency;

    /// The overflow arena (ChunkedPathStoreBuilder) grows every time a new
    /// file is created after the initial scan and is never reclaimed on
    /// removal. `trigger_rescan` replaces `sync_data`, dropping the builder
    /// and resetting the overflow count to 0 — the only path that reclaims
    /// the arena memory.
    #[test]
    fn trigger_rescan_reclaims_overflow() {
        let dir = tempfile::tempdir().unwrap();

        // Initial scan with one existing file. The base index locks in at
        // size 1; everything added after this lives in overflow.
        std::fs::write(dir.path().join("seed.txt"), b"seed").unwrap();

        let mut picker = FilePicker::new(FilePickerOptions {
            base_path: dir.path().to_str().unwrap().into(),
            watch: false,
            ..Default::default()
        })
        .unwrap();
        picker.collect_files().unwrap();
        assert_eq!(picker.get_files().len(), 1);
        assert_eq!(picker.get_overflow_files().len(), 0);

        // Create several new files and push them through the same code path
        // the background watcher uses. Each call appends to the overflow
        // builder's arena and chunk_dedup HashMap.
        for i in 0..5 {
            let path = dir.path().join(format!("new_{i}.txt"));
            std::fs::write(&path, b"content").unwrap();
            assert!(picker.on_create_or_modify(&path).is_some());
        }
        assert_eq!(picker.get_overflow_files().len(), 5);

        // Remove two of them: `remove_file_by_path` drops the FileItem via
        // `Vec::remove` but does NOT shrink the builder arena.
        for i in 0..2 {
            let path = dir.path().join(format!("new_{i}.txt"));
            std::fs::remove_file(&path).unwrap();
            assert!(picker.remove_file_by_path(&path));
        }
        assert_eq!(picker.get_overflow_files().len(), 3);

        // Full rescan: replaces sync_data, drops the builder, and re-indexes
        // from disk. All surviving files become base files again.
        let frecency = SharedFrecency::default();
        picker.trigger_rescan(&frecency).unwrap();
        assert_eq!(picker.get_overflow_files().len(), 0);
        // 1 seed + 3 surviving new_*.txt = 4 base files after rescan.
        assert_eq!(picker.get_files().len(), 4);
    }

    /// The watcher must watch every ancestor directory up to `base_path`,
    /// not just the immediate parents of indexed files. Intermediate dirs
    /// that contain only subdirectories (no direct files) are NOT in
    /// `sync_data.dirs` — yet they must still appear in `extract_watch_dirs`
    /// so Create events on new subdirectories below them fire.
    ///
    /// Correctness regression guard for any refactor that replaces the
    /// ancestor walk with a direct `sync_data.dirs` iteration.
    #[test]
    fn extract_watch_dirs_includes_pure_ancestor_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path();

        // Tree:
        //   base/src/components/button.txt    (src/components has a file)
        //   base/src/routes/home.txt          (src/routes has a file)
        //   base/lib/deep/nested/util.txt     (lib and lib/deep have no files)
        //
        // `sync_data.dirs` will only contain:
        //   src/components/
        //   src/routes/
        //   lib/deep/nested/
        //
        // But the watcher also needs:
        //   src/       (pure ancestor — no direct files)
        //   lib/       (pure ancestor)
        //   lib/deep/  (pure ancestor)
        // otherwise new siblings like `src/NewDir/x.txt` are missed.
        for rel in [
            "src/components/button.txt",
            "src/routes/home.txt",
            "lib/deep/nested/util.txt",
        ] {
            let path = base.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"x").unwrap();
        }

        let mut picker = FilePicker::new(FilePickerOptions {
            base_path: base.to_str().unwrap().into(),
            watch: false,
            ..Default::default()
        })
        .unwrap();
        picker.collect_files().unwrap();

        let mut watch_dirs: Vec<PathBuf> = Vec::new();
        picker.for_each_dir(|p| {
            watch_dirs.push(p.to_path_buf());
            std::ops::ControlFlow::Continue(())
        });
        let watch_set: std::collections::HashSet<PathBuf> = watch_dirs.iter().cloned().collect();

        // Immediate parents (in sync_data.dirs) must be present.
        for rel in ["src/components", "src/routes", "lib/deep/nested"] {
            assert!(
                watch_set.contains(&base.join(rel)),
                "expected immediate parent {rel} in watch dirs, got {watch_set:?}",
            );
        }

        // Pure-ancestor dirs (NOT in sync_data.dirs) must also be present.
        for rel in ["src", "lib", "lib/deep"] {
            assert!(
                watch_set.contains(&base.join(rel)),
                "expected pure-ancestor {rel} in watch dirs, got {watch_set:?}",
            );
        }

        // No duplicates — streaming dedup must not emit the same dir twice.
        assert_eq!(
            watch_dirs.len(),
            watch_set.len(),
            "duplicate watch dir emitted: {watch_dirs:?}",
        );

        // Base path itself is NOT walked into the result — the walker stops
        // at `current == base`. The outer `debouncer.watch(base_path, ...)`
        // call in create_debouncer covers it separately.
        assert!(
            !watch_set.contains(base),
            "base path must not be in watch dirs (covered by the top-level watch call)",
        );
    }

    #[test]
    fn common_dir_prefix_len_cases() {
        assert_eq!(common_dir_prefix_len("", ""), 0);
        assert_eq!(common_dir_prefix_len("", "src"), 0);
        assert_eq!(common_dir_prefix_len("lib", "src"), 0);
        assert_eq!(common_dir_prefix_len("src/components", "src/routes"), 4);
        assert_eq!(common_dir_prefix_len("lib/deep/nested", "lib/deep"), 8);
        assert_eq!(common_dir_prefix_len("lib/deep", "lib/deep/nested"), 8);
        assert_eq!(common_dir_prefix_len("lib/deep", "lib/deeper"), 4);
        assert_eq!(common_dir_prefix_len("src", "src"), 0);
        // "src" is emitted-as-dir; "src/x" extends it — full "src" is shared.
        assert_eq!(common_dir_prefix_len("src", "src/x"), 3);
    }
}
