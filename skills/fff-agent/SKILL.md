---
name: fff-agent
description: Use FFF through the local fff-agent sidecar CLI without persistent MCP context overhead.
---

# fff-agent

Use `fff-agent` for fast ranked discovery in a known git repository.

Prefer:

- `fff-agent find "query"` for fuzzy file/path discovery.
- `fff-agent grep "Identifier"` for repeated literal content search.
- `fff-agent multi-grep --pattern Foo --pattern foo_bar` for identifier variants.
- `fff-agent status`, `rescan`, and `stop` for daemon lifecycle checks.

Do not use it as a universal `rg` replacement.

Use `rg -F` or raw `rg` for exact exhaustive search, counts, audits,
punctuation-heavy parity checks, ignored/generated files, and one-off searches.

Use `sg` / ast-grep for syntax-aware matching, references, refactors, and
codemods.

Keep commands repo-scoped with `--repo /path/to/repo` when the current working
directory is ambiguous.
