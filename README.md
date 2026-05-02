# fff-agent-sidecar

Agent-friendly FFF sidecar for fast repository search without keeping an MCP
server permanently attached to an agent context.

This repository is a fork of
[dmtrKovalenko/fff](https://github.com/dmtrKovalenko/fff). The upstream project
already provides a Rust search core, MCP server, Node bindings, C FFI, Neovim
plugin, and Pi extension. This fork adds a small experimental layer for
Codex-like coding agents that want most of the speed benefits of FFF without
constant MCP tool/schema/instruction overhead.

## What Was Added

- `crates/fff-agent`: a tiny Rust CLI and sidecar daemon.
- `skills/fff-agent/SKILL.md`: a short local agent skill describing when to use
  `fff-agent`, `rg`, and `sg`.
- Workspace wiring for the new crate in `Cargo.toml` and `Cargo.lock`.
- Tests for repository safety, protocol serialization, and daemon-backed CLI
  grep.

## Why

Persistent MCP servers are useful, but they have a cost: every connected MCP
surface can add instructions, tool descriptions, schemas, and tool-call options
to the active agent context. For file search, that can be wasteful when the
agent only needs fast discovery occasionally.

`fff-agent` keeps the agent interface small:

```bash
fff-agent find "query" --repo /path/to/repo
fff-agent grep "Identifier" --repo /path/to/repo
fff-agent multi-grep --repo /path/to/repo --pattern Foo --pattern foo_bar
fff-agent status --repo /path/to/repo
fff-agent rescan --repo /path/to/repo
fff-agent stop --repo /path/to/repo
```

The first command starts one daemon per repository. The daemon keeps FFF's index
warm in memory. Later CLI calls reuse that daemon through a local Unix socket.

## Safe Defaults

The prototype is intentionally conservative for coding-agent workflows:

- no MCP registration;
- no global configuration changes;
- no telemetry;
- no update checks;
- no filesystem watcher by default;
- no persistent frecency database;
- git repository root validation before indexing;
- refusal to index unsafe roots such as `/` or `$HOME`;
- literal content grep by default, with `--regex` and `--fuzzy` as explicit
  opt-ins.

## When To Use It

Use `fff-agent` for repeated, ranked discovery inside a git repository:

- fuzzy file/path discovery;
- repeated identifier search;
- agent loops where a warm in-memory index pays off;
- compact result sets where ranking matters more than exhaustive parity.

Do not treat it as a universal `rg` replacement. Prefer raw `rg` for exact
audits, counts, generated or ignored files, one-off shell searches, and cases
where exhaustive behavior matters. Prefer `sg` / ast-grep for syntax-aware
matching, references, refactors, and codemods.

## Development

```bash
cargo test -p fff-agent
cargo clippy -p fff-agent -- -D warnings
cargo fmt --package fff-agent -- --check
```

On macOS, daemon integration tests need permission to bind a local Unix socket.
If a sandbox blocks socket binding, run the tests in a normal local shell.

## Status

This is an early prototype, not an upstream replacement. The goal is to validate
the agent workflow shape first: a short skill plus a local sidecar CLI, with FFF
core doing the actual search work.
