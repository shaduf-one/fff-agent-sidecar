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
