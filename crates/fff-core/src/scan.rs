//! Unified scan-phase orchestrator.
//!
//! Every (re)index code path — initial scan, FFI-triggered rescan,
//! watcher overflow rescan — goes through [`ScanJob::run`]. The
//! orchestrator owns the *sequence* of a scan:
//!
//!   1. walk filesystem off-lock
//!   2. swap `sync_data` under a brief write
//!   3. apply git status + frecency off-lock
//!   4. (optional, initial scan only) spawn the filesystem watcher
//!   5. (optional) post-scan: auto-size cache budget, warmup, bigram
//!
//! The picker write lock is held only in step 2 and step 5's index
//! install — both O(µs-ms), never seconds. Every other FFI caller on
//! the nvim main thread keeps running.
//!
//! ## Entry points
//!
//! - [`ScanJob::spawn`] — fire-and-forget from `SharedPicker` state.
//!   Used by the watcher overflow path and by FFI (`scan_files`).
//! - [`ScanJob::spawn_initial`] — same, but takes explicit config for
//!   the very first scan, before the `FilePicker` struct lives inside
//!   the shared handle.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use tracing::{error, info};

use crate::background_watcher::BackgroundWatcher;
use crate::bigram_filter::BigramOverlay;
use crate::error::Error;
use crate::bigram_filter::build_bigram_index;
use crate::file_picker::{self, FFFMode, warmup_mmaps};
use crate::shared::{SharedFrecency, SharedPicker};
use crate::types::ContentCacheBudget;

/// Shared atomic flags surfaced by the picker for the scan worker to
/// signal its progress. Grouped so every callsite passes one value,
/// not four.
#[derive(Clone, Default)]
pub struct ScanSignals {
    /// `true` while any scan phase is running. Readers (progress FFI,
    /// `wait_for_scan`) poll this.
    pub scanning: Arc<AtomicBool>,
    /// Set to `true` once the filesystem watcher has been installed
    /// (initial scan only). Watcher-dependent callers block on it.
    pub watcher_ready: Arc<AtomicBool>,
    /// Short-circuits a scan that's racing a `FilePicker` replacement.
    /// Owned by `FilePicker` and flipped by `FilePicker::cancel`.
    pub cancelled: Arc<AtomicBool>,
    /// `true` while the raw-pointer snapshot used by warmup + bigram
    /// build is live. Concurrent rescans skip themselves until this
    /// clears so `sync_data` can't be freed under the snapshot.
    pub post_scan_busy: Arc<AtomicBool>,
}

/// Which optional phases a scan should run.
#[derive(Clone, Copy, Default)]
pub(crate) struct ScanConfig {
    pub(crate) warmup: bool,
    pub(crate) content_indexing: bool,
    pub(crate) watch: bool,
    pub(crate) auto_cache_budget: bool,
    pub(crate) install_watcher: bool,
}

/// A fully-configured scan job ready to run on a background thread.
///
/// Build with [`ScanJob::from_picker`] (reads all state from the
/// current `FilePicker`) or [`ScanJob::initial`] (for the bootstrap
/// scan, before the picker is published to `SharedPicker`).
pub(crate) struct ScanJob {
    shared_picker: SharedPicker,
    shared_frecency: SharedFrecency,
    base_path: PathBuf,
    mode: FFFMode,
    signals: ScanSignals,
    config: ScanConfig,
    /// Walker-maintained counter backing `get_scan_progress` on the UI
    /// side. Reset to 0 at scan start, incremented per-file by the
    /// walker. Shared `Arc` so the UI polls the same atomic.
    scanned_files_counter: Arc<AtomicUsize>,
}

impl ScanJob {
    pub fn new(
        shared_picker: &SharedPicker,
        shared_frecency: &SharedFrecency,
        install_watcher: bool,
    ) -> Result<Option<Self>, Error> {
        let guard = shared_picker.read()?;
        let picker = guard.as_ref().ok_or(Error::FilePickerMissing)?;

        if picker.is_scan_active() {
            return Ok(None);
        }

        let signals = picker.scan_signals();
        if signals.post_scan_busy.load(Ordering::Acquire) {
            return Ok(None);
        }

        Ok(Some(Self {
            shared_picker: shared_picker.clone(),
            shared_frecency: shared_frecency.clone(),
            base_path: picker.base_path().to_path_buf(),
            mode: picker.mode(),
            signals,
            scanned_files_counter: picker.scanned_files_counter(),
            config: ScanConfig {
                warmup: picker.need_enable_mmap_cache(),
                content_indexing: picker.need_enable_content_indexing(),
                watch: picker.need_watch(),
                auto_cache_budget: !picker.has_explicit_cache_budget(),
                install_watcher,
            },
        }))
    }


    /// Same as [`new`] but without reading from the picker — caller
    /// supplies the base path / mode / flags directly. Used by the
    /// bootstrap scan before the `FilePicker` is published to
    /// `SharedPicker`.
    pub fn new_initial(
        shared_picker: SharedPicker,
        shared_frecency: SharedFrecency,
        base_path: PathBuf,
        mode: FFFMode,
        signals: ScanSignals,
        scanned_files_counter: Arc<AtomicUsize>,
        config: ScanConfig,
    ) -> Self {
        Self {
            shared_picker,
            shared_frecency,
            base_path,
            mode,
            signals,
            scanned_files_counter,
            config,
        }
    }

    /// Spawn the job on a dedicated OS thread. Returns immediately.
    pub fn spawn(self) -> std::thread::JoinHandle<()> {
        std::thread::Builder::new()
            .name("fff-scan".into())
            .spawn(move || self.run())
            .expect("failed to spawn fff-scan thread")
    }

    fn run(self) {
        let Self {
            shared_picker,
            shared_frecency,
            base_path,
            mode,
            signals,
            scanned_files_counter,
            config,
        } = self;

        let _scanning = ScanningGuard::new(&signals, config.install_watcher);

        // Reset the UI-visible counter; the walker bumps it per file
        // and `get_scan_progress` reads it without locks.
        scanned_files_counter.store(0, Ordering::Relaxed);

        // 1. Walk filesystem off-lock.
        let walk = match file_picker::walk_filesystem(
            &base_path,
            &scanned_files_counter,
            &shared_frecency,
            mode,
        ) {
            Ok(w) => w,
            Err(e) => {
                error!(?e, "scan walk failed");
                return;
            }
        };

        if signals.cancelled.load(Ordering::Acquire) {
            info!("walk completed but picker was replaced, discarding results");
            return;
        }

        let git_workdir = walk.sync.git_workdir.clone();

        // 2. Brief write to install the freshly-walked file list.
        if let Ok(mut guard) = shared_picker.write()
            && let Some(picker) = guard.as_mut()
        {
            picker.commit_new_sync(walk.sync);
        } else {
            error!("failed to install scan results into picker");
            return;
        }

        // Files are now searchable — flip the scan signal *early* so
        // UI progress polls see the picker as "ready" while we run the
        // optional post-scan steps in the background.
        signals.scanning.store(false, Ordering::Relaxed);

        // 3. Apply git status + frecency off-lock.
        if !signals.cancelled.load(Ordering::Acquire) {
            file_picker::apply_git_status_and_frecency(
                &shared_picker,
                &shared_frecency,
                walk.git_handle,
                mode,
            );
        }

        // 4. Install filesystem watcher (initial scan only).
        if config.install_watcher && config.watch && !signals.cancelled.load(Ordering::Acquire) {
            let shared_picker: &SharedPicker = &shared_picker;
            let shared_frecency: &SharedFrecency = &shared_frecency;
            let base_path: &std::path::Path = &base_path;
            match BackgroundWatcher::new(
                base_path.to_path_buf(),
                git_workdir,
                shared_picker.clone(),
                shared_frecency.clone(),
                mode,
            ) {
                Ok(watcher) => {
                    if let Ok(mut guard) = shared_picker.write()
                        && let Some(picker) = guard.as_mut()
                    {
                        picker.install_background_watcher(watcher);
                    }
                }
                Err(e) => error!(?e, "failed to initialize background watcher"),
            };
        }

        // 5. Post-scan warmup + bigram build.
        if (config.warmup || config.content_indexing) && !signals.cancelled.load(Ordering::Acquire)
        {
            run_post_scan(&shared_picker, &base_path, &signals, &config);
        }
    }
}

/// RAII helper that flips the `scanning` signal on construction and
/// resets it on drop (so early-returns can't leave it stuck on `true`).
/// Also drives the `watcher_ready` signal on the initial-scan path.
struct ScanningGuard<'a> {
    signals: &'a ScanSignals,
    release_watcher_ready_on_drop: bool,
}

impl<'a> ScanningGuard<'a> {
    fn new(signals: &'a ScanSignals, release_watcher_ready_on_drop: bool) -> Self {
        signals.scanning.store(true, Ordering::Relaxed);
        Self {
            signals,
            release_watcher_ready_on_drop,
        }
    }
}

impl Drop for ScanningGuard<'_> {
    fn drop(&mut self) {
        self.signals.scanning.store(false, Ordering::Relaxed);
        if self.release_watcher_ready_on_drop {
            self.signals.watcher_ready.store(true, Ordering::Release);
        }
    }
}

fn run_post_scan(
    shared_picker: &SharedPicker,
    base_path: &std::path::Path,
    signals: &ScanSignals,
    config: &ScanConfig,
) {
    signals.post_scan_busy.store(true, Ordering::Release);
    let _busy = PostScanBusyGuard(&signals.post_scan_busy);
    let phase_start = std::time::Instant::now();

    // Auto-scale the cache budget before we take the files snapshot —
    // warmup needs the final budget.
    if config.auto_cache_budget
        && !signals.cancelled.load(Ordering::Acquire)
        && let Ok(mut guard) = shared_picker.write()
        && let Some(picker) = guard.as_mut()
        && !picker.has_explicit_cache_budget()
    {
        let (files, _, _) = picker.sync_data_snapshot();
        picker.set_cache_budget(ContentCacheBudget::new_for_repo(files.len()));
    }

    // SAFETY: `post_scan_busy` blocks concurrent rescans from replacing
    // `sync_data` for the lifetime of the raw slice below.
    let Some((files, indexable_count, budget, arena)) = shared_picker
        .read()
        .ok()
        .and_then(|guard| guard.as_ref().map(snapshot_sync_data))
    else {
        return;
    };

    if config.warmup && !signals.cancelled.load(Ordering::Acquire) {
        let t = std::time::Instant::now();
        warmup_mmaps(files, &budget, base_path, arena);
        info!(
            "Warmup completed in {:.2}s (cached {} files, {} bytes)",
            t.elapsed().as_secs_f64(),
            budget.cached_count.load(Ordering::Relaxed),
            budget.cached_bytes.load(Ordering::Relaxed),
        );
    }

    if config.content_indexing && !signals.cancelled.load(Ordering::Acquire) {
        let indexable_files = &files[..indexable_count.min(files.len())];
        let (index, content_binary) =
            build_bigram_index(indexable_files, &budget, base_path, arena);

        if let Ok(mut guard) = shared_picker.write()
            && let Some(picker) = guard.as_mut()
        {
            for &idx in &content_binary {
                if let Some(file) = picker.get_file_mut(idx) {
                    file.set_binary(true);
                }
            }
            picker.set_bigram_index(index, BigramOverlay::new(indexable_count));
        }
    }

    info!(
        "Post-scan phase total: {:.2}s (warmup={}, content_indexing={})",
        phase_start.elapsed().as_secs_f64(),
        config.warmup,
        config.content_indexing,
    );
}

struct PostScanBusyGuard<'a>(&'a AtomicBool);
impl Drop for PostScanBusyGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

/// SAFETY: caller must hold `post_scan_busy` so `sync_data` isn't
/// replaced while the returned slice is alive.
fn snapshot_sync_data(
    picker: &crate::file_picker::FilePicker,
) -> (
    &'static [crate::types::FileItem],
    usize,
    Arc<ContentCacheBudget>,
    crate::simd_path::ArenaPtr,
) {
    let (files, indexable_count, arena) = picker.sync_data_snapshot();
    let ptr = files.as_ptr();
    let len = files.len();
    let static_files: &'static [crate::types::FileItem] =
        unsafe { std::slice::from_raw_parts(ptr, len) };
    (
        static_files,
        indexable_count,
        picker.cache_budget_arc(),
        arena,
    )
}
