# Development Workflow

Robot-task context: read this file, then use
[docs/codex/HANDOFF.md](docs/codex/HANDOFF.md) as the sole cross-session handoff
entry. Follow its references only as needed. `AGENTS.md` is a symlink to this
file; preserve the link and keep one set of rules.

**Always use `uv run`, not python**.

```sh

# 1. Make changes.

# 2. Type check.
uv run ty check  # Fast
uv run pyright  # More thorough, but slower

# 3. Run tests.
uv run pytest tests/  # Single suite
uv run pytest tests/<test_file>.py  # Specific file

# 4. Format and lint before committing.
uv run ruff format
uv run ruff check --fix
```

We've bundled common commands into a Makefile for convenience.

```sh
make format     # Format and lint
make type       # Type-check
make check      # make format && make type
make test-fast  # Run tests excluding slow ones
make test       # Run the full test suite
make docs       # Build documentation
```

Always run `make check` before committing. This runs formatting, linting,
and type checking. Do not commit code that fails type checking.

Before creating a PR, ensure all checks pass with `make test`.

When making user-facing changes, add an entry to `docs/source/changelog.rst`
under the "Upcoming version (not yet released)" section using
Added/Changed/Fixed categories. Reference issues with `:issue:\`123\``
(renders as a link to the GitHub issue).

## Context Management Rules

本项目采用固定的上下文文件体系。AGENTS.md保存长期工作规范，以下文件按职责维护：

| File | Responsibility and update trigger |
| --- | --- |
| `AGENTS.md` | Long-term work and context rules; update only when those rules change. No development logs. |
| `docs/codex/PROJECT_CONTEXT.md` | Goals, platform, framework, architecture, source paths and workflow; update only for lasting facts or structural changes. |
| `docs/codex/CURRENT_STATUS.md` | Latest stage, completed/current work, issues and near-term plan; refresh at session end and after important stages, replacing stale status rather than appending history. |
| `docs/codex/DECISIONS.md` | Important technical choices, reasons and impacts; update only when a choice is added, changed or retired. Label obsolete decisions. |
| `docs/codex/EXPERIMENT_LOG.md` | Actual experiments, parameters, runs, checkpoints, results and conclusions; update only for new or corrected experimental evidence. Never mark planned training as completed. |
| `docs/codex/HANDOFF.md` | Sole cross-session entry; refresh at session end and every formal handoff. Keep roughly 100–150 lines or fewer. |

### Session Start

1. 先阅读 `docs/codex/HANDOFF.md`。
2. 按需阅读 `docs/codex/CURRENT_STATUS.md` 和其他文档。
3. 检查Git状态及本次任务涉及的源码；无Git元数据时如实记录，不自行初始化仓库。
4. 不进行无关的大范围源码扫描。

### Session End

1. 更新 `docs/codex/CURRENT_STATUS.md`。
2. 更新 `docs/codex/HANDOFF.md`。
3. 只有发生相关变化时才更新其他文档。
4. 禁止自动创建新的上下文Markdown文件。
5. 不得自动执行Git commit或push。

### Documentation Principles

- HANDOFF.md是唯一交接入口；CURRENT_STATUS和HANDOFF只保留最新有效状态，
  避免重复和无限追加。DECISIONS与EXPERIMENT_LOG保留有依据的历史记录。
- HANDOFF uses these seven headings: 当前目标、当前进展、最近一次开发、当前问题、
  下一步任务、Git 与实验状态、需要进一步查阅的文档. Give at most 3–5 next tasks.
- Do not create Markdown for task completion, compaction or handoff, including
  dated handoffs, SUMMARY, PROGRESS, SESSION_NOTES or NEXT_STEPS. Do not add
  handoff, stage-summary or temporary-note files under `docs/codex/`, or
  automatically create Markdown backup copies.
- Existing extra indexes, plans, audits and reports are supporting or historical
  material, not additional context authorities. If they duplicate the six files,
  identify and report the overlap before proposing consolidation; do not delete
  or overwrite them. Do not keep updating a second handoff entry.
- Put details in the file that owns them; elsewhere use a brief summary and
  a path reference. Preserve important information before removing duplication.
  Apart from the two session-end updates above, update only files whose
  maintenance trigger applies, not all six by default.
- Separate verified facts, user reports, inferences and plans; mark unknown
  information `待确认`. Check actual code, Git when available, and saved evidence.
  Report conflicts; do not invent Git state, remote state, checkpoint paths,
  training completion or success, or rewrite historical results without evidence.
- Documentation work alone never authorizes business-code changes, training,
  commits or deletion of documents, logs, checkpoints, datasets or other results.
  Read historical authorizations in context and follow the current user request.
- 未经用户授权，不得删除任何已有文件，包括文档、训练日志、checkpoint和数据集。
- For development, explain the changes with their core code after each batch.
  Codex runs the local uv checks; provide the user only the required remote
  synchronization/check commands, with short explanations. Check device resources
  before simulation and use small local runs. Long remote training uses the
  screen name recorded in PROJECT_CONTEXT.
- Before a handoff, verify links, current/remote/experiment distinctions, obsolete
  decisions, and changed-file scope. Check whether existing sync archives contain
  older context files; do not let an old bundle overwrite the current handoff.

# Commits and PRs

- Put `Fixes #<number>` at the end of the commit message body, not in
  the title.
- PR body should be plain, concise prose. No section headers, checklists,
  or structured templates. Describe the problem, what the change does, and
  any non-obvious tradeoffs. A good PR description reads like a short
  paragraph to a colleague, not a form.
- PR and commit messages are rendered on GitHub, so don't hard-wrap them
  at 88 columns. Let each sentence flow on one line.

Some style guidelines to follow:
- Line length limit is 88 columns. This applies to code, comments, and docstrings.
- 在 `src/` 下新增/生成Python代码时，必须为关键部分补充简明中文注释，帮助用户理解。
  重点说明逻辑目的、关键步骤，以及必要的张量形状/排列、单位、坐标系和非直观约束，
  避免逐行翻译或重复代码；后续修改相关实现时同步维护注释，不把计划写成已验证事实。
- Avoid local imports unless they are strictly necessary (e.g. circular imports).
- Tests should follow these principles:
  - Use functions and fixtures; do not use test classes.
  - Favor targeted, efficient tests over exhaustive edge-case coverage.
  - Prefer running individual tests rather than the full test suite to improve iteration speed.
