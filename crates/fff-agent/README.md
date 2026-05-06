# fff-agent

`fff-agent` is a tiny sidecar CLI for using FFF in coding-agent workflows without
keeping an MCP server permanently attached to the prompt.

The first CLI call starts one daemon per repository. The daemon keeps the FFF
index warm in memory, while agents keep using normal shell commands:

```bash
fff-agent find "query" --repo /path/to/repo --limit 20
fff-agent grep "query" --repo /path/to/repo --limit 20
fff-agent multi-grep --repo /path/to/repo --pattern Foo --pattern foo_bar
fff-agent status --repo /path/to/repo
fff-agent rescan --repo /path/to/repo
fff-agent stop --repo /path/to/repo
```

Set `FFF_AGENT_RUNTIME_DIR=/path/to/runtime-dir` to override where the per-repo
Unix socket and daemon stderr log are written. By default, both live under
`$TMPDIR/fff-agent`.

Defaults are conservative:

- no MCP registration;
- no update checks;
- no telemetry;
- no filesystem watcher;
- explicit git repository root validation;
- literal grep by default. Use `--regex` or `--fuzzy` to opt in.

Use `rg` for exact exhaustive search, counts, audits, generated or ignored
files, and one-off shell searches. Use `ast-grep` / `sg` for syntax-aware
matching and rewrites.

If daemon startup fails, `fff-agent` reports the socket path, stderr log path,
and the last daemon stderr output. This is especially useful in sandboxes that
block Unix socket binding.
