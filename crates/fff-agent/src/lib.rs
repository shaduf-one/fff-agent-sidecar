use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use fff::{
    ContentCacheBudget, FFFMode, FilePicker, FilePickerOptions, FuzzySearchOptions,
    GrepMode as FffGrepMode, GrepSearchOptions, PaginationArgs, QueryParser, SharedFrecency,
    SharedPicker,
};
use git2::Repository;
use serde::{Deserialize, Serialize};

const DEFAULT_LIMIT: usize = 20;
const STARTUP_TIMEOUT: Duration = Duration::from_secs(60);
const SCAN_TIMEOUT: Duration = Duration::from_secs(30);
const STDERR_TAIL_BYTES: usize = 8 * 1024;

pub type Result<T> = std::result::Result<T, AgentError>;

#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("failed to resolve current directory: {0}")]
    CurrentDir(#[source] std::io::Error),
    #[error("failed to canonicalize path {path}: {source}")]
    Canonicalize {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("refusing to index unsafe repository root: {path}")]
    UnsafeRepoRoot { path: PathBuf },
    #[error("no git repository found from {path}")]
    GitRootNotFound { path: PathBuf },
    #[error("git repository has no worktree at {path}")]
    BareGitRepository { path: PathBuf },
    #[error("failed to create runtime directory {path}: {source}")]
    CreateRuntimeDir {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to remove stale socket {path}: {source}")]
    RemoveStaleSocket {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to connect to daemon socket {path}: {source}")]
    Connect {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to spawn daemon: {0}")]
    SpawnDaemon(#[source] std::io::Error),
    #[error(
        "daemon exited before becoming ready at {path}; stderr log: {log_path}; status: {status}; stderr: {stderr}"
    )]
    DaemonExitedBeforeReady {
        path: PathBuf,
        log_path: PathBuf,
        status: ExitStatus,
        stderr: String,
    },
    #[error("daemon did not become ready at {path}; stderr log: {log_path}; stderr: {stderr}")]
    DaemonStartupTimeout {
        path: PathBuf,
        log_path: PathBuf,
        stderr: String,
    },
    #[error("failed to poll daemon process: {0}")]
    PollDaemon(#[source] std::io::Error),
    #[error("failed to bind daemon socket {path}: {source}")]
    Bind {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to create daemon stderr log {path}: {source}")]
    CreateLogFile {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to read daemon request: {0}")]
    ReadRequest(#[source] std::io::Error),
    #[error("failed to write daemon response: {0}")]
    WriteResponse(#[source] std::io::Error),
    #[error("failed to encode request: {0}")]
    EncodeRequest(#[source] serde_json::Error),
    #[error("failed to decode request: {0}")]
    DecodeRequest(#[source] serde_json::Error),
    #[error("failed to decode daemon response: {0}")]
    DecodeResponse(#[source] serde_json::Error),
    #[error("daemon returned an error: {0}")]
    Daemon(String),
    #[error("failed to initialize FFF picker: {0}")]
    PickerInit(String),
    #[error("FFF initial scan timed out after {0:?}")]
    ScanTimeout(Duration),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GrepMode {
    #[default]
    Plain,
    Regex,
    Fuzzy,
}

impl From<GrepMode> for FffGrepMode {
    fn from(value: GrepMode) -> Self {
        match value {
            GrepMode::Plain => Self::PlainText,
            GrepMode::Regex => Self::Regex,
            GrepMode::Fuzzy => Self::Fuzzy,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentRequest {
    Find {
        query: String,
        limit: usize,
        json: bool,
    },
    Grep {
        query: String,
        limit: usize,
        mode: GrepMode,
        json: bool,
    },
    MultiGrep {
        patterns: Vec<String>,
        constraints: Option<String>,
        limit: usize,
        json: bool,
    },
    Status,
    Rescan,
    Shutdown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentResponse {
    Text {
        text: String,
    },
    Status {
        repo: String,
        indexed_files: usize,
        scanning: bool,
        watch: bool,
    },
    Error {
        message: String,
    },
    Shutdown,
}

pub fn resolve_repo_root(start: Option<&Path>) -> Result<PathBuf> {
    let start = match start {
        Some(path) => path.to_path_buf(),
        None => std::env::current_dir().map_err(AgentError::CurrentDir)?,
    };
    let canonical = start
        .canonicalize()
        .map_err(|source| AgentError::Canonicalize {
            path: start.clone(),
            source,
        })?;

    reject_unsafe_root(&canonical)?;

    let repo = Repository::discover(&canonical).map_err(|_| AgentError::GitRootNotFound {
        path: canonical.clone(),
    })?;
    let workdir = repo
        .workdir()
        .ok_or_else(|| AgentError::BareGitRepository {
            path: canonical.clone(),
        })?
        .canonicalize()
        .map_err(|source| AgentError::Canonicalize {
            path: canonical.clone(),
            source,
        })?;

    reject_unsafe_root(&workdir)?;
    Ok(workdir)
}

pub fn socket_path_for_repo(repo: &Path) -> Result<PathBuf> {
    Ok(runtime_dir().join(format!("{}.sock", repo_hash(repo)?)))
}

fn daemon_log_path_for_repo(repo: &Path) -> Result<PathBuf> {
    Ok(runtime_dir().join(format!("{}.log", repo_hash(repo)?)))
}

fn repo_hash(repo: &Path) -> Result<String> {
    let canonical = repo
        .canonicalize()
        .map_err(|source| AgentError::Canonicalize {
            path: repo.to_path_buf(),
            source,
        })?;
    let hash = blake3::hash(canonical.to_string_lossy().as_bytes());
    Ok(hash.to_hex()[..16].to_string())
}

fn runtime_dir() -> PathBuf {
    match std::env::var_os("FFF_AGENT_RUNTIME_DIR") {
        Some(path) if !path.is_empty() => PathBuf::from(path),
        _ => std::env::temp_dir().join("fff-agent"),
    }
}

pub fn send_request(repo: &Path, request: AgentRequest) -> Result<AgentResponse> {
    let repo = resolve_repo_root(Some(repo))?;
    let socket = socket_path_for_repo(&repo)?;
    if matches!(request, AgentRequest::Shutdown) {
        return match call_daemon(&socket, &request) {
            Ok(response) => Ok(response),
            Err(AgentError::Connect { .. }) => Ok(AgentResponse::Shutdown),
            Err(error) => Err(error),
        };
    }
    ensure_daemon(&repo, &socket)?;
    call_daemon(&socket, &request)
}

pub fn run_daemon(repo: &Path, socket: &Path) -> Result<()> {
    let repo = resolve_repo_root(Some(repo))?;
    if let Some(parent) = socket.parent() {
        fs::create_dir_all(parent).map_err(|source| AgentError::CreateRuntimeDir {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    if socket.exists() {
        fs::remove_file(socket).map_err(|source| AgentError::RemoveStaleSocket {
            path: socket.to_path_buf(),
            source,
        })?;
    }

    let engine = AgentEngine::new(repo)?;
    let listener = UnixListener::bind(socket).map_err(|source| AgentError::Bind {
        path: socket.to_path_buf(),
        source,
    })?;

    for stream in listener.incoming() {
        let stream = match stream {
            Ok(stream) => stream,
            Err(_) => continue,
        };
        let should_stop = handle_client(stream, &engine)?;
        if should_stop {
            break;
        }
    }

    let _ = fs::remove_file(socket);
    engine.stop();
    Ok(())
}

fn reject_unsafe_root(path: &Path) -> Result<()> {
    if path.parent().is_none() || home_dir().as_deref() == Some(path) {
        return Err(AgentError::UnsafeRepoRoot {
            path: path.to_path_buf(),
        });
    }
    Ok(())
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

fn ensure_daemon(repo: &Path, socket: &Path) -> Result<()> {
    if call_daemon(socket, &AgentRequest::Status).is_ok() {
        return Ok(());
    }

    let log_path = daemon_log_path_for_repo(repo)?;
    if socket.exists() {
        fs::remove_file(socket).map_err(|source| AgentError::RemoveStaleSocket {
            path: socket.to_path_buf(),
            source,
        })?;
    }
    if let Some(parent) = socket.parent() {
        fs::create_dir_all(parent).map_err(|source| AgentError::CreateRuntimeDir {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    let stderr_log = File::create(&log_path).map_err(|source| AgentError::CreateLogFile {
        path: log_path.clone(),
        source,
    })?;

    let mut child = Command::new(std::env::current_exe().map_err(AgentError::SpawnDaemon)?)
        .arg("daemon")
        .arg("--repo")
        .arg(repo)
        .arg("--socket")
        .arg(socket)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::from(stderr_log))
        .spawn()
        .map_err(AgentError::SpawnDaemon)?;

    let start = Instant::now();
    while start.elapsed() < STARTUP_TIMEOUT {
        if call_daemon(socket, &AgentRequest::Status).is_ok() {
            return Ok(());
        }
        if let Some(status) = child.try_wait().map_err(AgentError::PollDaemon)? {
            return Err(AgentError::DaemonExitedBeforeReady {
                path: socket.to_path_buf(),
                log_path: log_path.clone(),
                status,
                stderr: stderr_tail(&log_path),
            });
        }
        std::thread::sleep(Duration::from_millis(50));
    }

    Err(AgentError::DaemonStartupTimeout {
        path: socket.to_path_buf(),
        log_path: log_path.clone(),
        stderr: stderr_tail(&log_path),
    })
}

fn stderr_tail(path: &Path) -> String {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(error) => return format!("<unavailable: {error}>"),
    };
    let mut bytes = Vec::new();
    if let Err(error) = file.read_to_end(&mut bytes) {
        return format!("<unavailable: {error}>");
    }
    if bytes.is_empty() {
        return "<empty>".to_string();
    }
    let start = bytes.len().saturating_sub(STDERR_TAIL_BYTES);
    String::from_utf8_lossy(&bytes[start..]).trim().to_string()
}

fn call_daemon(socket: &Path, request: &AgentRequest) -> Result<AgentResponse> {
    let mut stream = UnixStream::connect(socket).map_err(|source| AgentError::Connect {
        path: socket.to_path_buf(),
        source,
    })?;
    let payload = serde_json::to_string(request).map_err(AgentError::EncodeRequest)?;
    writeln!(stream, "{payload}").map_err(AgentError::WriteResponse)?;

    let mut response = String::new();
    let mut reader = BufReader::new(stream);
    reader
        .read_line(&mut response)
        .map_err(AgentError::ReadRequest)?;
    let response: AgentResponse =
        serde_json::from_str(&response).map_err(AgentError::DecodeResponse)?;
    if let AgentResponse::Error { message } = &response {
        return Err(AgentError::Daemon(message.clone()));
    }
    Ok(response)
}

fn handle_client(mut stream: UnixStream, engine: &AgentEngine) -> Result<bool> {
    let mut line = String::new();
    {
        let mut reader = BufReader::new(&mut stream);
        reader
            .read_line(&mut line)
            .map_err(AgentError::ReadRequest)?;
    }

    let request: AgentRequest = serde_json::from_str(&line).map_err(AgentError::DecodeRequest)?;
    let should_stop = matches!(request, AgentRequest::Shutdown);
    let response = match engine.handle(request) {
        Ok(response) => response,
        Err(error) => AgentResponse::Error {
            message: error.to_string(),
        },
    };
    let payload = serde_json::to_string(&response).map_err(AgentError::EncodeRequest)?;
    writeln!(stream, "{payload}").map_err(AgentError::WriteResponse)?;
    Ok(should_stop)
}

struct AgentEngine {
    repo: PathBuf,
    picker: SharedPicker,
    frecency: SharedFrecency,
}

impl AgentEngine {
    fn new(repo: PathBuf) -> Result<Self> {
        let picker = SharedPicker::default();
        let frecency = SharedFrecency::noop();
        FilePicker::new_with_shared_state(
            picker.clone(),
            frecency.clone(),
            FilePickerOptions {
                base_path: repo.to_string_lossy().to_string(),
                enable_mmap_cache: true,
                enable_content_indexing: true,
                mode: FFFMode::Ai,
                cache_budget: ContentCacheBudget::from_overrides(
                    10_000,
                    128 * 1024 * 1024,
                    10 * 1024 * 1024,
                ),
                watch: false,
            },
        )
        .map_err(|error| AgentError::PickerInit(error.to_string()))?;

        if !picker.wait_for_scan(SCAN_TIMEOUT) {
            return Err(AgentError::ScanTimeout(SCAN_TIMEOUT));
        }

        Ok(Self {
            repo,
            picker,
            frecency,
        })
    }

    fn handle(&self, request: AgentRequest) -> Result<AgentResponse> {
        match request {
            AgentRequest::Find { query, limit, json } => {
                self.find(&query, normalized_limit(limit), json)
            }
            AgentRequest::Grep {
                query,
                limit,
                mode,
                json,
            } => self.grep(&query, normalized_limit(limit), mode, json),
            AgentRequest::MultiGrep {
                patterns,
                constraints,
                limit,
                json,
            } => self.multi_grep(
                &patterns,
                constraints.as_deref(),
                normalized_limit(limit),
                json,
            ),
            AgentRequest::Status => self.status(),
            AgentRequest::Rescan => self.rescan(),
            AgentRequest::Shutdown => Ok(AgentResponse::Shutdown),
        }
    }

    fn find(&self, query: &str, limit: usize, json: bool) -> Result<AgentResponse> {
        let guard = self
            .picker
            .read()
            .map_err(|error| AgentError::PickerInit(error.to_string()))?;
        let picker = guard
            .as_ref()
            .ok_or_else(|| AgentError::PickerInit("file picker not initialized".to_string()))?;
        let parser = QueryParser::default();
        let parsed = parser.parse(query);
        let result = picker.fuzzy_search(
            &parsed,
            None,
            FuzzySearchOptions {
                max_threads: 0,
                current_file: None,
                project_path: Some(picker.base_path()),
                combo_boost_score_multiplier: 0,
                min_combo_count: 0,
                pagination: PaginationArgs { offset: 0, limit },
            },
        );

        if json {
            let rows: Vec<_> = result
                .items
                .iter()
                .zip(result.scores.iter())
                .map(|(item, score)| {
                    serde_json::json!({
                        "path": item.relative_path(picker),
                        "score": score.total,
                        "git_status": format!("{:?}", item.git_status),
                    })
                })
                .collect();
            return Ok(AgentResponse::Text {
                text: serde_json::to_string_pretty(&rows).unwrap_or_else(|_| "[]".to_string()),
            });
        }

        let mut lines = Vec::new();
        for item in result.items {
            lines.push(item.relative_path(picker));
        }
        Ok(AgentResponse::Text {
            text: empty_or_join(lines),
        })
    }

    fn grep(&self, query: &str, limit: usize, mode: GrepMode, json: bool) -> Result<AgentResponse> {
        let guard = self
            .picker
            .read()
            .map_err(|error| AgentError::PickerInit(error.to_string()))?;
        let picker = guard
            .as_ref()
            .ok_or_else(|| AgentError::PickerInit("file picker not initialized".to_string()))?;
        let parser = QueryParser::default();
        let parsed = parser.parse(query);
        let options = GrepSearchOptions {
            mode: mode.into(),
            page_limit: limit,
            max_matches_per_file: limit,
            smart_case: true,
            classify_definitions: true,
            trim_whitespace: true,
            ..Default::default()
        };
        let result = picker.grep(&parsed, &options);
        Ok(format_grep_response(picker, &result, json))
    }

    fn multi_grep(
        &self,
        patterns: &[String],
        constraints: Option<&str>,
        limit: usize,
        json: bool,
    ) -> Result<AgentResponse> {
        if patterns.is_empty() {
            return Ok(AgentResponse::Error {
                message: "at least one --pattern is required".to_string(),
            });
        }
        let guard = self
            .picker
            .read()
            .map_err(|error| AgentError::PickerInit(error.to_string()))?;
        let picker = guard
            .as_ref()
            .ok_or_else(|| AgentError::PickerInit("file picker not initialized".to_string()))?;
        let parser = QueryParser::default();
        let parsed_constraints = parser.parse(constraints.unwrap_or(""));
        let pattern_refs: Vec<&str> = patterns.iter().map(String::as_str).collect();
        let options = GrepSearchOptions {
            mode: FffGrepMode::PlainText,
            page_limit: limit,
            max_matches_per_file: limit,
            smart_case: true,
            classify_definitions: true,
            trim_whitespace: true,
            ..Default::default()
        };
        let result = picker.multi_grep(&pattern_refs, &parsed_constraints.constraints, &options);
        Ok(format_grep_response(picker, &result, json))
    }

    fn status(&self) -> Result<AgentResponse> {
        let guard = self
            .picker
            .read()
            .map_err(|error| AgentError::PickerInit(error.to_string()))?;
        let picker = guard
            .as_ref()
            .ok_or_else(|| AgentError::PickerInit("file picker not initialized".to_string()))?;
        Ok(AgentResponse::Status {
            repo: self.repo.to_string_lossy().to_string(),
            indexed_files: picker.get_files().len(),
            scanning: picker.is_scan_active(),
            watch: picker.need_watch(),
        })
    }

    fn rescan(&self) -> Result<AgentResponse> {
        {
            let mut guard = self
                .picker
                .write()
                .map_err(|error| AgentError::PickerInit(error.to_string()))?;
            let picker = guard
                .as_mut()
                .ok_or_else(|| AgentError::PickerInit("file picker not initialized".to_string()))?;
            picker
                .trigger_rescan(&self.frecency)
                .map_err(|error| AgentError::PickerInit(error.to_string()))?;
        }
        Ok(AgentResponse::Text {
            text: "rescan complete".to_string(),
        })
    }

    fn stop(&self) {
        if let Ok(mut guard) = self.picker.write()
            && let Some(picker) = guard.as_mut()
        {
            picker.stop_background_monitor();
        }
    }
}

fn format_grep_response(
    picker: &FilePicker,
    result: &fff::GrepResult<'_>,
    json: bool,
) -> AgentResponse {
    if json {
        let rows: Vec<_> = result
            .matches
            .iter()
            .map(|m| {
                let file = result.files[m.file_index];
                serde_json::json!({
                    "path": file.relative_path(picker),
                    "line": m.line_number,
                    "column": m.col,
                    "text": m.line_content,
                    "is_definition": m.is_definition,
                })
            })
            .collect();
        return AgentResponse::Text {
            text: serde_json::to_string_pretty(&rows).unwrap_or_else(|_| "[]".to_string()),
        };
    }

    let mut lines = Vec::new();
    for m in &result.matches {
        let file = result.files[m.file_index];
        lines.push(format!(
            "{}:{}:{}: {}",
            file.relative_path(picker),
            m.line_number,
            m.col + 1,
            m.line_content
        ));
    }
    AgentResponse::Text {
        text: empty_or_join(lines),
    }
}

fn normalized_limit(limit: usize) -> usize {
    if limit == 0 { DEFAULT_LIMIT } else { limit }
}

fn empty_or_join(lines: Vec<String>) -> String {
    if lines.is_empty() {
        "0 results".to_string()
    } else {
        lines.join("\n")
    }
}
