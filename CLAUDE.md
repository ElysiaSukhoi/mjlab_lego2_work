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
- 远程服务器默认恢复SSH直接传输：先完成本地修改/验证，再提供明确文件清单、
  备份与SCP/SSH同步及远程核验指令；以实际传输文件的内容校验值确认版本，保护旧训练结果。
  GitHub仅用于本地代码的可选版本管理，是否commit/push及何时进行均由用户决定，
  服务器同步和验证不以commit、push、GitHub连通或提交SHA为前置条件。
  新/更新的远程测试.sh不包含Git下载/更新操作，不硬编码期望commit SHA；同步与验证分开。
- 远程执行的.sh及配套校验清单/上传包，默认传至
  `~/zrq/mjlab_lego2/docs/codex/artifacts/`，不再散放到`~/zrq/`顶层。
  传输前备份目标同名文件，启动时明确源码目录与脚本路径；脚本应在切换工作目录前
  解析自身绝对路径，不能依赖用户启动时的当前目录来读取自身或相邻资源。
  整理旧上传物时先备份并记录原路径/新路径/内容校验值，同名不同内容禁止覆盖；
  历史副本可移至artifacts下独立历史目录，不删除，也不把未适配路径的旧脚本当当前入口执行。
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
- Python关键代码的中文说明遵守下方“中文代码注释规范”。
- Avoid local imports unless they are strictly necessary (e.g. circular imports).
- Tests should follow these principles:
  - Use functions and fixtures; do not use test classes.
  - Favor targeted, efficient tests over exhaustive edge-case coverage.
  - Prefer running individual tests rather than the full test suite to improve iteration speed.

## 中文代码注释规范

注释应帮助用户理解训练代码的用途、参数和文件关系，使用简明、通俗的中文。
在`src/`下新增或修改关键Python实现时同步维护；相关训练、评估脚本也按此要求说明。

- **优先覆盖的内容**：针对高台/中央坑训练修改的地形、出生、奖励、观测、
  课程、结束条件和训练指标，以及任务注册、神经网络、PPO、模型迁移和回放入口。
  重点解释影响训练行为或结果理解的部分，不要求逐行翻译代码。
- **模块开头说明文件关系**：在重点文件开头，用分行注释或模块说明写清文件职责，
  由哪个入口调用、从哪些文件取得配置/函数、把结果交给谁使用。
  例如区分任务注册、环境配置、奖励计算和网络/PPO配置各自负责什么。
  只写已核对的实际关系，不把文件名相似当成存在调用关系。
- **重点函数/类开头说明用途**：先说“做什么”或“大致配置了什么”，再补充理解所需的
  输入、输出或适用范围。优先使用动作和目的描述；必要术语首次出现时简要解释，
  不堆叠术语，也不重复整段模块介绍或函数内部实现。
- **关键参数就近解释**：奖励权重、阈值、地形尺寸、网络层宽、学习率、采样长度、
  环境数和迭代预算等，优先在对应代码行后写明作用；需要时注明正负号、单位、
  坐标参考、维数/排列及非直观约束。说明较长时放在该参数上方分行书写，
  不为加行旁注释而把一行挤得过长。
- **解释训练指标的统计口径**：在指标定义或登记位置说明它观察什么、如何汇总。
  分清原始代价、带权得分、乘时间步后的奖励、帧占比和回合成功标记；
  必要时说明分母、有效数据范围和高度参考点，避免仅凭指标名称误读结果。
- **分行和排版**：遵守88字符行长限制。多项内容分多行，按用途/关系/参数等含义分段，
  不把一段说明全部写在一行；保持注释靠近对应实现，避免无关说明打断代码阅读。
- **维护准确性**：明确区分当前任务与旧方案、通用默认值与实际覆盖值、候选参数与
  已验证结论。实现或参数变更时同步核对注释，清理过时说法；单独优化注释时保持
  执行逻辑和参数不变，不把计划、推测或旧实验结果写成当前已证实的事实。
