use std::{fs, process::Command};

use fff_agent::{
    AgentError, AgentRequest, AgentResponse, resolve_repo_root, send_request, socket_path_for_repo,
};
use tempfile::TempDir;

fn init_repo() -> TempDir {
    let dir = TempDir::new().unwrap();
    let status = Command::new("git")
        .arg("init")
        .arg("--quiet")
        .arg(dir.path())
        .status()
        .unwrap();
    assert!(status.success());
    dir
}

#[test]
fn resolves_git_repo_from_nested_directory() {
    let repo = init_repo();
    let nested = repo.path().join("src/bin");
    fs::create_dir_all(&nested).unwrap();

    let resolved = resolve_repo_root(Some(&nested)).unwrap();

    assert_eq!(resolved, repo.path().canonicalize().unwrap());
}

#[test]
fn rejects_filesystem_root() {
    let err = resolve_repo_root(Some(std::path::Path::new("/"))).unwrap_err();

    assert!(matches!(err, AgentError::UnsafeRepoRoot { .. }));
}

#[test]
fn rejects_non_git_directory() {
    let dir = TempDir::new().unwrap();
    let err = resolve_repo_root(Some(dir.path())).unwrap_err();

    assert!(matches!(err, AgentError::GitRootNotFound { .. }));
}

#[test]
fn socket_path_is_stable_and_repo_specific() {
    let repo_a = init_repo();
    let repo_b = init_repo();

    let first = socket_path_for_repo(repo_a.path()).unwrap();
    let second = socket_path_for_repo(repo_a.path()).unwrap();
    let other = socket_path_for_repo(repo_b.path()).unwrap();

    assert_eq!(first, second);
    assert_ne!(first, other);
    assert_eq!(first.extension().and_then(|ext| ext.to_str()), Some("sock"));
}

#[test]
fn stop_is_idempotent_when_daemon_is_not_running() {
    let repo = init_repo();

    let response = send_request(repo.path(), AgentRequest::Shutdown).unwrap();

    assert_eq!(response, AgentResponse::Shutdown);
}
