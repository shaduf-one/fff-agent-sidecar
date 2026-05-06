use std::{fs, process::Command};

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
fn cli_auto_starts_daemon_and_greps_literal_punctuation() {
    let repo = init_repo();
    let source = repo.path().join("src/main.rs");
    fs::create_dir_all(source.parent().unwrap()).unwrap();
    fs::write(&source, "fn main() {\n    attachApplicationScoring();\n}\n").unwrap();

    let bin = env!("CARGO_BIN_EXE_fff-agent");
    let output = Command::new(bin)
        .arg("--repo")
        .arg(repo.path())
        .arg("grep")
        .arg("attachApplicationScoring(")
        .arg("--limit")
        .arg("5")
        .output()
        .unwrap();

    let _ = Command::new(bin)
        .arg("--repo")
        .arg(repo.path())
        .arg("stop")
        .output();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("src/main.rs:2:1: attachApplicationScoring();"),
        "stdout: {stdout}"
    );
}

#[test]
fn cli_uses_runtime_dir_override_for_socket_and_log() {
    let repo = init_repo();
    let runtime = TempDir::new().unwrap();
    let source = repo.path().join("src/main.rs");
    fs::create_dir_all(source.parent().unwrap()).unwrap();
    fs::write(&source, "fn main() {}\n").unwrap();

    let bin = env!("CARGO_BIN_EXE_fff-agent");
    let output = Command::new(bin)
        .env("FFF_AGENT_RUNTIME_DIR", runtime.path())
        .arg("--repo")
        .arg(repo.path())
        .arg("status")
        .output()
        .unwrap();

    let entries: Vec<_> = fs::read_dir(runtime.path())
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();

    let _ = Command::new(bin)
        .env("FFF_AGENT_RUNTIME_DIR", runtime.path())
        .arg("--repo")
        .arg(repo.path())
        .arg("stop")
        .output();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("watch: false"), "stdout: {stdout}");
    assert!(
        entries
            .iter()
            .any(|path| path.extension().is_some_and(|ext| ext == "sock")),
        "runtime entries: {entries:?}"
    );
    assert!(
        entries
            .iter()
            .any(|path| path.extension().is_some_and(|ext| ext == "log")),
        "runtime entries: {entries:?}"
    );
}
