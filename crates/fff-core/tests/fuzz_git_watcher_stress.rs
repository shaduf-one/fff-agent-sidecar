//! Vibe coded stress test: randomized file + git operations driven against the *real*
//! `BackgroundWatcher`, asserting the git-status invariant after every mutation.
//!
//! ## What this test is trying to catch
//!
//! The background watcher is supposed to keep every indexed file's
//! `git_status` in sync with the actual repository state at all times.
//! There are two code paths that can drift:
//!
//!   1. **Per-file updates** — when regular files are created / modified the
//!      watcher queries `git_status_for_paths(&[changed_file])`. If the git
//!      state changes for files that weren't in the batch (rename detection,
//!      submodule paths, index modifications applied by a sibling process),
//!      those other files keep a stale `git_status` until the next full
//!      rescan is triggered.
//!
//!   2. **Full rescans** — triggered by changes under `.git/` (index, HEAD,
//!      MERGE_HEAD, etc.) and by `.gitignore` edits. A missed event here is
//!      the most common failure mode: the watcher coalesced or dropped the
//!      `.git/index` notification and the picker never learns that e.g. a
//!      `git commit` cleared every `WT_MODIFIED`.
//!
//! ## How the test works
//!
//!  * Spin up a real `FilePicker` with `watch: true` on a fresh temp repo.
//!  * Use `proptest` to generate a sequence of 20–40 randomized ops (heavy on
//!    git mutations: add / commit / reset / stash / gitignore edits).
//!  * After **every** op, poll until the picker's per-file `git_status` agrees
//!    with `git2::Repository::statuses()` verbatim, or bail with a rich diff.
//!  * If there is ever a divergence that doesn't resolve within
//!    `CONVERGE_TIMEOUT`, the test fails with the list of `(path, truth,
//!    picker)` mismatches.
//!
//! The shape of the scenario is shrinkable: on failure proptest will shrink
//! the `Vec<AbstractOp>` so the reported diff corresponds to the smallest
//! surviving sequence.
//!
//! ## Runtime
//!
//! Each scenario does a real filesystem scan + watcher setup (~0.5–1 s) and
//! then runs 20–40 ops each of which waits for event propagation
//! (~100–500 ms on macOS FSEvents). With `cases = 2` the test takes
//! ~30–45 s on a typical dev machine — too slow to run as part of the
//! default `cargo test`, so it's gated behind the `stress` cfg.
//!
//! Run it explicitly with:
//! ```sh
//! RUSTFLAGS="--cfg stress" cargo test -p fff-search --test fuzz_git_watcher_stress -- --nocapture
//! ```
//!
//! Or increase coverage via env:
//! ```sh
//! FFF_STRESS_CASES=8 FFF_STRESS_MAX_OPS=60 \
//!     RUSTFLAGS="--cfg stress" cargo test -p fff-search --test fuzz_git_watcher_stress -- --nocapture
//! ```

#![cfg(stress)]
use fff_search::file_picker::{FFFMode, FilePicker};
use fff_search::{
    FilePickerOptions, FuzzySearchOptions, PaginationArgs, QueryParser, SharedFrecency,
    SharedPicker,
};
use git2::{Repository, Status, StatusOptions};
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::{
    Config as ProptestConfig, FileFailurePersistence, RngAlgorithm, TestRng, TestRunner,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};
use tempfile::TempDir;

/// Upper bound on how long we wait for the picker to converge after any op.
/// The watcher debounce is 50 ms; worst-case FSEvents propagation + queued
/// full-rescan can easily reach several seconds. 15 s is very generous.
const CONVERGE_TIMEOUT: Duration = Duration::from_secs(15);

/// Poll interval while waiting for convergence.
const CONVERGE_POLL: Duration = Duration::from_millis(50);

/// Small pause between back-to-back ops so the debouncer has a chance to
/// batch its deliveries naturally. Keeping this small stresses the watcher
/// harder (more overlapping batches).
const PER_OP_SETTLE: Duration = Duration::from_millis(10);

fn stress_cases() -> u32 {
    std::env::var("FFF_STRESS_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2)
}

fn stress_max_ops() -> usize {
    std::env::var("FFF_STRESS_MAX_OPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(40)
}

fn stress_min_ops() -> usize {
    std::env::var("FFF_STRESS_MIN_OPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20)
}

// ═══════════════════════════════════════════════════════════════════════════
// Abstract ops (the fuzz alphabet)
// ═══════════════════════════════════════════════════════════════════════════

/// A single randomized action. The runtime interprets abstract handles
/// (`idx`) by modulo against the *current* list of live files, so the same
/// abstract op sequence is always runnable regardless of which files exist.
#[derive(Debug, Clone)]
enum AbstractOp {
    CreateFile {
        seed: u32,
        content_seed: u32,
    },
    EditFile {
        idx: usize,
        content_seed: u32,
    },
    DeleteFile {
        idx: usize,
    },
    RenameFile {
        idx: usize,
        new_seed: u32,
    },
    CreateSubdirFile {
        dir_seed: u16,
        file_seed: u16,
        content_seed: u32,
    },
    GitignoreAppend {
        pattern_seed: u16,
    },
    GitAddAll,
    GitCommit {
        msg_seed: u16,
    },
    GitResetHard,
    GitStashThenPop,
    Touch {
        idx: usize,
    }, // rewrite with same content; should still fire Modify
    Noop,
}

fn op_strategy() -> impl Strategy<Value = AbstractOp> {
    // Weights tuned to lean on git status churn without starving the other ops.
    prop_oneof![
        10 => (any::<u32>(), any::<u32>()).prop_map(|(a, b)| AbstractOp::CreateFile {
            seed: a, content_seed: b,
        }),
        12 => (any::<usize>(), any::<u32>()).prop_map(|(i, c)| AbstractOp::EditFile {
            idx: i, content_seed: c,
        }),
        4  => any::<usize>().prop_map(|i| AbstractOp::DeleteFile { idx: i }),
        4  => (any::<usize>(), any::<u32>()).prop_map(|(i, s)| AbstractOp::RenameFile {
            idx: i, new_seed: s,
        }),
        3  => (any::<u16>(), any::<u16>(), any::<u32>()).prop_map(
            |(d, f, c)| AbstractOp::CreateSubdirFile {
                dir_seed: d, file_seed: f, content_seed: c,
            }
        ),
        2  => any::<u16>().prop_map(|p| AbstractOp::GitignoreAppend { pattern_seed: p }),
        3  => any::<usize>().prop_map(|i| AbstractOp::Touch { idx: i }),
        8  => Just(AbstractOp::GitAddAll),
        6  => any::<u16>().prop_map(|m| AbstractOp::GitCommit { msg_seed: m }),
        2  => Just(AbstractOp::GitResetHard),
        3  => Just(AbstractOp::GitStashThenPop),
        1  => Just(AbstractOp::Noop),
    ]
}

fn ops_strategy() -> impl Strategy<Value = Vec<AbstractOp>> {
    let min = stress_min_ops();
    let max = stress_max_ops();
    prop::collection::vec(op_strategy(), min..=max)
}

const DEFAULT_STRESS_SEED: u64 = 0xDEAD_BEEF_CAFE_BABE;

fn proptest_config() -> ProptestConfig {
    ProptestConfig {
        cases: stress_cases(),
        // Cap shrinking because each shrink iteration replays the whole
        // scenario (seconds of real IO). Proptest's default 4096 is wild.
        max_shrink_iters: 16,
        // `fork: true` would isolate runs but swallows the panic payload
        // and printed diff through `rusty-fork`'s wire format — we want
        // the convergence report visible on stderr, so we keep in-process.
        // The watcher is fully torn down between cases via `Drop`.
        fork: false,
        // Integration tests don't live alongside a `lib.rs`/`main.rs`, so
        // proptest's default `SourceParallel` persistence strategy can't
        // locate the source tree and logs a noisy warning on every run.
        // Pin the regression file to an explicit path next to this test
        // so failing seeds get checked in and reproduced in CI.
        failure_persistence: Some(Box::new(FileFailurePersistence::Direct(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fuzz_git_watcher_stress.proptest-regressions",
        )))),
        ..ProptestConfig::default()
    }
}

proptest! {
    #![proptest_config(proptest_config())]

    /// Random-seeded scenario. Proptest picks a fresh seed every run from
    /// system entropy, so CI sees different trajectories on every build
    /// while known-bad seeds remain pinned in the regressions file.
    #[test]
    fn stress_random(ops in ops_strategy()) {
        run_stress_scenario(&ops);
    }
}

/// Deterministic scenario keyed off `FFF_STRESS_SEED` (or
/// [`DEFAULT_STRESS_SEED`] if the env var is not set). Runs the same
/// proptest `ops_strategy()` but uses a ChaCha RNG seeded by the expanded
/// u64, so the exact case sequence is reproducible across machines and CI
/// runs.
///
/// When this test panics, the panic message includes the seed so you can
/// re-run the failing case locally via:
///
/// ```sh
/// RUSTFLAGS="--cfg stress" FFF_STRESS_SEED=0xDEADBEEFCAFEBABE \
///     cargo test -p fff-search --test fuzz_git_watcher_stress seeded -- --nocapture
/// ```
#[test]
fn stress_seeded() {
    let seed = parse_stress_seed();
    let seed_bytes = expand_u64_seed(seed);

    eprintln!(
        "stress_seeded: using deterministic seed {seed:#018x} \
         (override via FFF_STRESS_SEED=<u64|0xHEX>)"
    );

    let mut config = proptest_config();
    // The seeded run should never write to the shared regressions file —
    // its failures are reproducible from the env var alone, and polluting
    // the shared file with the deterministic seed would mask regressions
    // for the random run that actually needs persistence.
    config.failure_persistence = Some(Box::new(FileFailurePersistence::Off));

    let rng = TestRng::from_seed(RngAlgorithm::ChaCha, &seed_bytes);
    let mut runner = TestRunner::new_with_rng(config, rng);
    let strategy = ops_strategy();

    // Mimic `proptest!`'s case loop: run `Config::cases` independent draws
    // from the strategy, failing the test as soon as any one scenario
    // diverges. We drive this ourselves because `runner.run()` can't
    // accept a non-Fn closure, and our scenario runner is side-effectful.
    for case_idx in 0..runner.config().cases {
        let tree = strategy
            .new_tree(&mut runner)
            .expect("ops_strategy::new_tree");
        let ops = tree.current();
        eprintln!(
            "  case {}/{}: {} ops",
            case_idx + 1,
            runner.config().cases,
            ops.len()
        );
        run_stress_scenario(&ops);
    }
}

/// Parse `FFF_STRESS_SEED` as either decimal or `0x`-prefixed hex.
fn parse_stress_seed() -> u64 {
    match std::env::var("FFF_STRESS_SEED") {
        Ok(raw) => {
            let trimmed = raw.trim();
            if let Some(hex) = trimmed
                .strip_prefix("0x")
                .or_else(|| trimmed.strip_prefix("0X"))
            {
                u64::from_str_radix(hex, 16)
                    .unwrap_or_else(|e| panic!("FFF_STRESS_SEED={raw:?} is not valid hex: {e}"))
            } else {
                trimmed
                    .parse::<u64>()
                    .unwrap_or_else(|e| panic!("FFF_STRESS_SEED={raw:?} is not a valid u64: {e}"))
            }
        }
        Err(_) => DEFAULT_STRESS_SEED,
    }
}

/// Expand a u64 into the 32-byte seed ChaCha20 wants, by repeating the
/// little-endian bytes four times. The cycle is deliberate: two different
/// u64 seeds produce completely different byte sequences, so collisions
/// across the expansion are irrelevant in practice.
fn expand_u64_seed(seed: u64) -> [u8; 32] {
    let le = seed.to_le_bytes();
    let mut out = [0u8; 32];
    for i in 0..4 {
        out[i * 8..(i + 1) * 8].copy_from_slice(&le);
    }
    out
}

#[derive(Debug)]
struct Live {
    /// Path relative to the repo root (forward slashes on all platforms).
    relative: String,
    abs: PathBuf,
}

fn run_stress_scenario(ops: &[AbstractOp]) {
    // Opt-in tracing so `RUST_LOG=fff_search=debug` shows the watcher /
    // refresh / update trail when debugging a failing run. Uses
    // `try_init` so proptest can run many scenarios in the same process
    // without double-initialising the subscriber.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("off")),
        )
        .with_test_writer()
        .try_init();

    let tmp = TempDir::new().expect("mktemp");
    let base = tmp.path().canonicalize().expect("canonicalize tmp");

    seed_repo(&base);

    let (shared_picker, _frecency) = start_watched_picker(&base);
    wait_ready(&shared_picker);

    // Reconcile `live` from disk: after `git reset --hard` or similar the
    // worktree can change under us, so we resync at every step instead of
    // trusting our in-memory list.
    let mut live: Vec<Live> = scan_live_files(&base);

    // Sleep past the coarse (1 s) mtime tick so subsequent edits definitely
    // advance mtime. Not strictly required for git status correctness but
    // matches what real users experience.
    std::thread::sleep(Duration::from_millis(1100));

    for (step, op) in ops.iter().enumerate() {
        apply_op(op, &base, &mut live);

        // Re-sync live from disk — ops like GitResetHard / GitStashThenPop
        // may have created or removed files behind our back.
        live = scan_live_files(&base);

        std::thread::sleep(PER_OP_SETTLE);

        if let Err(err) = converge_git_status(&shared_picker, &base) {
            panic!(
                "\n──────────────────────────────────────────────────────────\n\
                 ❌ Picker git_status diverged from repository truth.\n\
                 ──────────────────────────────────────────────────────────\n\
                 step:    {step}\n\
                 op:      {op:?}\n\
                 scenario ops:\n{trace}\n\
                 ──────────────────────────────────────────────────────────\n\
                 {err}\n",
                trace = format_ops_trace(ops, step),
            );
        }
    }
}

fn format_ops_trace(ops: &[AbstractOp], up_to_and_including: usize) -> String {
    let mut s = String::new();
    for (i, op) in ops.iter().enumerate().take(up_to_and_including + 1) {
        s.push_str(&format!("   [{i:>3}] {op:?}\n"));
    }
    s
}

fn seed_repo(base: &Path) {
    fs::create_dir_all(base.join("src")).unwrap();
    fs::write(base.join("README.md"), "# seed\n").unwrap();
    fs::write(base.join("src/main.rs"), "fn main() {}\n").unwrap();
    fs::write(base.join("src/lib.rs"), "// lib\n").unwrap();
    fs::write(base.join(".gitignore"), "*.log\ntmp/\n").unwrap();

    git(base, &["init", "-b", "main"]);
    git(base, &["config", "user.email", "fuzz@fff.test"]);
    git(base, &["config", "user.name", "fuzz"]);
    // Disable rename detection noise — libgit2 still computes ranks, but
    // keeping the porcelain behaviour deterministic helps with diff reading.
    git(base, &["config", "status.renames", "false"]);
    git(base, &["add", "-A"]);
    git(base, &["commit", "-m", "seed", "--no-gpg-sign"]);
}

fn apply_op(op: &AbstractOp, base: &Path, live: &mut [Live]) {
    use AbstractOp::*;

    match op {
        CreateFile { seed, content_seed } => {
            let rel = format!("f_{seed:08x}.rs");
            let abs = base.join(&rel);
            if abs.exists() {
                return;
            }
            fs::write(&abs, content_for(*content_seed, &rel)).unwrap();
        }
        EditFile { idx, content_seed } => {
            if live.is_empty() {
                return;
            }
            let i = idx % live.len();
            let abs = &live[i].abs;
            if !abs.is_file() {
                return;
            }
            // Body must differ so git sees WT_MODIFIED (not just atime touch).
            let body = format!(
                "// edited seed={content_seed:08x}\n{}",
                content_for(*content_seed, &live[i].relative)
            );
            fs::write(abs, body).unwrap();
        }
        Touch { idx } => {
            // Rewrite with the same bytes we currently hold on disk. Still
            // generates a Modify event; git status stays the same if the
            // content matches the index, or WT_MODIFIED if it differs.
            if live.is_empty() {
                return;
            }
            let i = idx % live.len();
            let abs = &live[i].abs;
            if let Ok(contents) = fs::read(abs) {
                let _ = fs::write(abs, contents);
            }
        }
        DeleteFile { idx } => {
            if live.is_empty() {
                return;
            }
            let i = idx % live.len();
            let _ = fs::remove_file(&live[i].abs);
        }
        RenameFile { idx, new_seed } => {
            if live.is_empty() {
                return;
            }
            let i = idx % live.len();
            let old = &live[i].abs;
            if !old.is_file() {
                return;
            }
            let new_rel = format!("r_{new_seed:08x}.rs");
            let new_abs = base.join(&new_rel);
            if new_abs.exists() {
                return;
            }
            let _ = fs::rename(old, &new_abs);
        }
        CreateSubdirFile {
            dir_seed,
            file_seed,
            content_seed,
        } => {
            let rel = format!("d_{dir_seed:04x}/inside_{file_seed:04x}.rs");
            let abs = base.join(&rel);
            if abs.exists() {
                return;
            }
            fs::create_dir_all(abs.parent().unwrap()).unwrap();
            fs::write(&abs, content_for(*content_seed, &rel)).unwrap();
        }
        GitignoreAppend { pattern_seed } => {
            // Introduce NEW ignore patterns for a namespace not used by any
            // other op so we never accidentally re-ignore a file under test.
            let pattern = format!("__ignored_{pattern_seed:x}/\n");
            let gi = base.join(".gitignore");
            let mut cur = fs::read_to_string(&gi).unwrap_or_default();
            cur.push_str(&pattern);
            fs::write(&gi, cur).unwrap();
        }
        GitAddAll => {
            git_allow_fail(base, &["add", "-A"]);
        }
        GitCommit { msg_seed } => {
            // `--allow-empty` so we don't depend on there actually being
            // staged changes; this still bumps HEAD and rewrites .git/index.
            let _ = git_output(
                base,
                &[
                    "commit",
                    "-m",
                    &format!("fuzz-{msg_seed:x}"),
                    "--allow-empty",
                    "--allow-empty-message",
                    "--no-gpg-sign",
                ],
            );
        }
        GitResetHard => {
            git_allow_fail(base, &["reset", "--hard"]);
        }
        GitStashThenPop => {
            git_allow_fail(base, &["add", "-A"]);
            let stash = git_output(base, &["stash", "push", "-u", "-m", "fuzz"]);
            let had_stash = stash
                .as_ref()
                .map(|o| {
                    o.status.success()
                        && !String::from_utf8_lossy(&o.stdout).contains("No local changes")
                })
                .unwrap_or(false);
            if had_stash {
                git_allow_fail(base, &["stash", "pop"]);
            }
        }
        Noop => {}
    }
}

/// Block until the picker's git status agrees with `git2::Repository::statuses`.
///
/// On timeout, returns an `Err` describing every disagreement.
fn converge_git_status(shared_picker: &SharedPicker, base: &Path) -> Result<(), String> {
    let deadline = Instant::now() + CONVERGE_TIMEOUT;
    let mut last_mismatches: Vec<Mismatch>;

    loop {
        let truth = read_truth_status(base);
        let picker_view = read_picker_status(shared_picker);
        last_mismatches = diff_statuses(&truth, &picker_view);

        if last_mismatches.is_empty() {
            return Ok(());
        }
        if Instant::now() >= deadline {
            // Second-opinion probe via a direct fuzzy_search per-mismatched-path.
            // Proves the bug reproduces via the exact user-facing lookup path
            // (not just via the enumeration page).
            let report = format_mismatches(&last_mismatches, shared_picker);
            return Err(report);
        }
        std::thread::sleep(CONVERGE_POLL);
    }
}

#[derive(Debug)]
enum Mismatch {
    /// Git knows about this path with `truth`, picker has `picker` (or None = missing).
    Disagree {
        path: String,
        truth: Status,
        picker: Option<Option<Status>>,
    },
    /// Picker has a non-clean entry for a path git doesn't report at all.
    ExtraInPicker { path: String, picker: Status },
}

fn read_truth_status(base: &Path) -> BTreeMap<String, Status> {
    let repo = Repository::open(base).expect("open repo for truth");
    let mut opts = StatusOptions::new();
    opts.include_untracked(true)
        .recurse_untracked_dirs(true)
        .include_unmodified(true)
        .exclude_submodules(true);
    let statuses = repo.statuses(Some(&mut opts)).expect("read statuses");

    let mut out = BTreeMap::new();
    for entry in statuses.iter() {
        if let Some(p) = entry.path() {
            // git2 returns forward-slash paths; accept as-is.
            out.insert(p.to_string(), entry.status());
        }
    }
    out
}

fn read_picker_status(shared: &SharedPicker) -> BTreeMap<String, Option<Status>> {
    let guard = shared.read().expect("picker read lock");
    let picker = guard.as_ref().expect("picker initialized");

    // Use the user-facing `fuzzy_search` API with an empty query to enumerate
    // every file a user could ever see in the picker. An empty fuzzy query
    // falls through to the frecency-only scoring path which returns ALL
    // non-deleted files (base + overflow). A very large `limit` makes sure
    // we don't silently paginate anything away for scenarios with many files.
    //
    // This intentionally mirrors what a real Neovim user observes — soft-
    // deleted tombstones, files filtered out by search-side predicates, etc.
    // are all excluded from the picker's user-facing view.
    let parser = QueryParser::default();
    let parsed = parser.parse("");
    let result = picker.fuzzy_search(
        &parsed,
        None,
        FuzzySearchOptions {
            max_threads: 1,
            pagination: PaginationArgs {
                offset: 0,
                limit: 100_000,
            },
            ..Default::default()
        },
    );

    let mut out = BTreeMap::new();
    for f in &result.items {
        out.insert(f.relative_path(picker), f.git_status);
    }
    out
}

/// Probe a single file by name via the public `fuzzy_search` API and return
/// its status, or `None` if the file is not findable. Used for diagnostics
/// when the top-level enumeration reports a disagreement — confirms that
/// the mismatch reproduces via the exact user-facing lookup path.
fn probe_single_file_status(shared: &SharedPicker, relative: &str) -> Option<Option<Status>> {
    let guard = shared.read().ok()?;
    let picker = guard.as_ref()?;
    let parser = QueryParser::default();
    let parsed = parser.parse(relative);
    let result = picker.fuzzy_search(
        &parsed,
        None,
        FuzzySearchOptions {
            max_threads: 1,
            pagination: PaginationArgs {
                offset: 0,
                limit: 200,
            },
            ..Default::default()
        },
    );
    result
        .items
        .iter()
        .find(|f| f.relative_path(picker) == relative)
        .map(|f| f.git_status)
}

/// Returns the full list of disagreements. Empty means "in sync".
///
/// We deliberately exclude a few categories from being considered bugs:
///   * Paths that git marks as `WT_DELETED` / `INDEX_DELETED` but the picker
///     has already dropped — this is by design (deleted files leave the
///     index immediately).
///   * Paths that git marks as `IGNORED` but the picker has already dropped.
///   * Paths under `.git/` — never tracked by the picker.
fn diff_statuses(
    truth: &BTreeMap<String, Status>,
    picker: &BTreeMap<String, Option<Status>>,
) -> Vec<Mismatch> {
    let mut out = Vec::new();

    for (path, &truth_status) in truth {
        if path.starts_with(".git/") || path == ".git" {
            continue;
        }
        match picker.get(path) {
            Some(&p) => {
                if !status_equivalent(truth_status, p) {
                    out.push(Mismatch::Disagree {
                        path: path.clone(),
                        truth: truth_status,
                        picker: Some(p),
                    });
                }
            }
            None => {
                // Tolerate picker not having deleted-from-disk files.
                let only_absence_reasons =
                    Status::WT_DELETED | Status::INDEX_DELETED | Status::IGNORED;
                if !truth_status.intersects(only_absence_reasons) {
                    out.push(Mismatch::Disagree {
                        path: path.clone(),
                        truth: truth_status,
                        picker: None,
                    });
                }
            }
        }
    }

    for (path, &p) in picker {
        if path.starts_with(".git/") || path == ".git" {
            continue;
        }
        if !truth.contains_key(path) {
            // Picker thinks a file exists that git has never heard of.
            // Only a bug if the picker also thinks it has a non-clean
            // status for it — otherwise it's a transient during which the
            // picker has indexed a file before the next truth snapshot.
            let non_clean = match p {
                None => false,
                Some(s) => !(s.is_empty() || s == Status::CURRENT),
            };
            if non_clean {
                out.push(Mismatch::ExtraInPicker {
                    path: path.clone(),
                    picker: p.unwrap_or(Status::CURRENT),
                });
            }
        }
    }

    out
}

/// `None` and `Some(CURRENT|empty)` are both "clean"; otherwise the bitsets
/// must match exactly.
fn status_equivalent(truth: Status, picker: Option<Status>) -> bool {
    let picker_bits = picker.unwrap_or(Status::CURRENT);
    let truth_clean = truth.is_empty() || truth == Status::CURRENT;
    let picker_clean = picker_bits.is_empty() || picker_bits == Status::CURRENT;
    if truth_clean && picker_clean {
        return true;
    }
    truth == picker_bits
}

fn format_mismatches(mismatches: &[Mismatch], shared: &SharedPicker) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "{} mismatch(es) after {} of wait:\n",
        mismatches.len(),
        humantime(CONVERGE_TIMEOUT),
    ));
    for m in mismatches {
        match m {
            Mismatch::Disagree {
                path,
                truth,
                picker,
            } => {
                let probe = probe_single_file_status(shared, path);
                s.push_str(&format!(
                    "  • {path}\n      truth       : {}\n      picker(enum): {}\n      picker(probe): {}\n",
                    format_status(Some(*truth)),
                    match picker {
                        Some(p) => format_status(*p),
                        None => "<missing from enumeration>".into(),
                    },
                    match probe {
                        Some(p) => format_status(p),
                        None => "<not returned by fuzzy_search>".into(),
                    }
                ));
            }
            Mismatch::ExtraInPicker { path, picker } => {
                let probe = probe_single_file_status(shared, path);
                s.push_str(&format!(
                    "  • {path}\n      truth        : <missing from repo>\n      picker(enum) : {}\n      picker(probe): {}\n",
                    format_status(Some(*picker)),
                    match probe {
                        Some(p) => format_status(p),
                        None => "<not returned by fuzzy_search>".into(),
                    }
                ));
            }
        }
    }
    s
}

fn format_status(s: Option<Status>) -> String {
    match s {
        None => "None (= clean)".into(),
        Some(st) if st.is_empty() || st == Status::CURRENT => "CURRENT (= clean)".into(),
        Some(st) => format!("{st:?}"),
    }
}

fn humantime(d: Duration) -> String {
    format!("{:.1}s", d.as_secs_f64())
}

fn scan_live_files(base: &Path) -> Vec<Live> {
    let mut out = Vec::new();
    let repo = match Repository::open(base) {
        Ok(r) => r,
        Err(_) => return out,
    };
    let mut opts = StatusOptions::new();
    opts.include_untracked(true)
        .recurse_untracked_dirs(true)
        .include_unmodified(true)
        .exclude_submodules(true);
    let statuses = match repo.statuses(Some(&mut opts)) {
        Ok(s) => s,
        Err(_) => return out,
    };
    for entry in statuses.iter() {
        if let Some(p) = entry.path() {
            let abs = base.join(p);
            // Must be a real file *right now* — ignore stale WT_DELETED rows.
            if abs.is_file() {
                out.push(Live {
                    relative: p.to_string(),
                    abs,
                });
            }
        }
    }
    out
}

fn content_for(seed: u32, name: &str) -> String {
    // Keep it small but distinct so adjacent edits produce different SHAs.
    format!(
        "// file: {name}\n\
         // seed: {seed:08x}\n\
         pub fn anchor_{seed:x}() {{ let _ = \"{seed:08x}\"; }}\n"
    )
}

fn start_watched_picker(base: &Path) -> (SharedPicker, SharedFrecency) {
    let shared_picker = SharedPicker::default();
    let shared_frecency = SharedFrecency::noop();

    FilePicker::new_with_shared_state(
        shared_picker.clone(),
        shared_frecency.clone(),
        FilePickerOptions {
            base_path: base.to_string_lossy().to_string(),
            enable_mmap_cache: false,
            enable_content_indexing: false,
            mode: FFFMode::Neovim,
            watch: true,
            ..Default::default()
        },
    )
    .expect("FilePicker::new_with_shared_state");

    (shared_picker, shared_frecency)
}

fn wait_ready(p: &SharedPicker) {
    assert!(
        p.wait_for_scan(Duration::from_secs(15)),
        "timed out waiting for initial scan"
    );
    assert!(
        p.wait_for_watcher(Duration::from_secs(15)),
        "timed out waiting for watcher"
    );
    // macOS FSEvents sometimes delivers a burst of "warmup" events right
    // after the stream opens — let them drain before we start fuzzing.
    std::thread::sleep(Duration::from_millis(200));
}

fn git_env() -> [(&'static str, &'static str); 4] {
    [
        ("GIT_AUTHOR_NAME", "fuzz"),
        ("GIT_AUTHOR_EMAIL", "fuzz@fff.test"),
        ("GIT_COMMITTER_NAME", "fuzz"),
        ("GIT_COMMITTER_EMAIL", "fuzz@fff.test"),
    ]
}

fn git(base: &Path, args: &[&str]) {
    let out = Command::new("git")
        .args(args)
        .current_dir(base)
        .envs(git_env())
        .output()
        .unwrap_or_else(|e| panic!("git {args:?}: {e}"));
    assert!(
        out.status.success(),
        "git {args:?} failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

fn git_allow_fail(base: &Path, args: &[&str]) {
    let _ = Command::new("git")
        .args(args)
        .current_dir(base)
        .envs(git_env())
        .output();
}

fn git_output(base: &Path, args: &[&str]) -> Option<std::process::Output> {
    Command::new("git")
        .args(args)
        .current_dir(base)
        .envs(git_env())
        .output()
        .ok()
}
