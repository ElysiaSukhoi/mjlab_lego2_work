#!/usr/bin/env bash
# HTTP/2中断后的续接版：保留失败现场，HTTP/1.1浅克隆到另一新目录。
set -euo pipefail
umask 077

LEGO2_GIT_OLD="${LEGO2_GIT_OLD:-$HOME/zrq/mjlab_lego2}"
LEGO2_GIT_NEW="${LEGO2_GIT_NEW:-$HOME/zrq/mjlab_lego2_git_http1}"
LEGO2_GIT_REPO="${LEGO2_GIT_REPO:-https://github.com/ElysiaSukhoi/mjlab_lego2_work.git}"
LEGO2_GIT_UV="${LEGO2_GIT_UV:-$HOME/.local/bin/uv}"
LEGO2_GIT_BASE="${LEGO2_GIT_BASE:-$HOME/zrq/backups}"
LEGO2_GIT_EXPECTED="be4f42aaf97bd6e3b42ec9342d7d702750dfa27c"

test -d "$LEGO2_GIT_OLD/src"
test -f "$LEGO2_GIT_OLD/pyproject.toml"
test -d "$LEGO2_GIT_OLD/.venv"
test -x "$LEGO2_GIT_UV"
test "$LEGO2_GIT_NEW" != "$LEGO2_GIT_OLD"
mkdir -p "$LEGO2_GIT_BASE"
LEGO2_GIT_AUDIT="$(mktemp -d "$LEGO2_GIT_BASE/git_bootstrap_http1_20261115_XXXXXX")"
printf '接入核查目录：%s\n' "$LEGO2_GIT_AUDIT"

# 仅归档旧源码，不搬动或打包已有logs/checkpoint/.venv。
tar -czf "$LEGO2_GIT_AUDIT/old_source_before.tar.gz" \
  -C "$LEGO2_GIT_OLD" src scripts tests pyproject.toml uv.lock
sha256sum "$LEGO2_GIT_AUDIT/old_source_before.tar.gz" \
  > "$LEGO2_GIT_AUDIT/old_source_before.sha256"

if [[ ! -e "$LEGO2_GIT_NEW" && ! -L "$LEGO2_GIT_NEW" ]]; then
  # -c只影响本次命令；浅克隆减少历史传输，TLS证书验证保持原设置。
  if ! git -c http.version=HTTP/1.1 clone --depth 1 --branch main --single-branch \
    "$LEGO2_GIT_REPO" "$LEGO2_GIT_NEW" \
    2>&1 | tee "$LEGO2_GIT_AUDIT/clone.txt"; then
    printf '下载未完成；原目录和失败现场保留。核查目录：%s\n' \
      "$LEGO2_GIT_AUDIT" >&2
    exit 1
  fi
else
  # 重跑只接受同一个干净克隆；不fetch、切分支或覆盖已有目录。
  if [[ ! -d "$LEGO2_GIT_NEW/.git" || -L "$LEGO2_GIT_NEW" ]]; then
    printf '目标目录已存在且不是可复核的克隆，保留原状：%s\n' "$LEGO2_GIT_NEW" >&2
    exit 1
  fi
fi

test "$(git -C "$LEGO2_GIT_NEW" rev-parse --show-toplevel)" = "$LEGO2_GIT_NEW"
test "$(git -C "$LEGO2_GIT_NEW" remote get-url origin)" = "$LEGO2_GIT_REPO"
test "$(git -C "$LEGO2_GIT_NEW" branch --show-current)" = main
test -z "$(git -C "$LEGO2_GIT_NEW" status --porcelain)"
LEGO2_GIT_ACTUAL="$(git -C "$LEGO2_GIT_NEW" rev-parse HEAD)"
printf '%s\n' "$LEGO2_GIT_ACTUAL" > "$LEGO2_GIT_AUDIT/new_head.txt"
if [[ "$LEGO2_GIT_ACTUAL" != "$LEGO2_GIT_EXPECTED" ]]; then
  printf '提交不匹配：期望%s，实际%s。保留两目录，请回传本核查目录。\n' \
    "$LEGO2_GIT_EXPECTED" "$LEGO2_GIT_ACTUAL" >&2
  exit 1
fi

# 用旧环境运行标准库比对工具；--no-sync确保不更改其依赖。
"$LEGO2_GIT_UV" run --no-sync --project "$LEGO2_GIT_OLD" python - \
  "$LEGO2_GIT_OLD" "$LEGO2_GIT_NEW" "$LEGO2_GIT_AUDIT" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

old, new, audit = map(Path, sys.argv[1:])
assert old.resolve() != new.resolve()
tracked = subprocess.check_output(
    ['git', '-C', str(new), 'ls-files', '-z'], text=True
).split('\0')[:-1]

def describe(path):
    if path.is_symlink():
        return {'kind': 'symlink', 'target': os.readlink(path)}
    if path.is_file():
        with path.open('rb') as stream:
            return {'kind': 'file', 'sha256': hashlib.file_digest(
                stream, 'sha256'
            ).hexdigest()}
    return {'kind': 'directory' if path.is_dir() else 'absent'}

same, different, missing = [], [], []
for name in tracked:
    left, right = describe(old / name), describe(new / name)
    if left['kind'] == 'absent':
        missing.append(name)
    elif left == right:
        same.append(name)
    else:
        different.append({'path': name, 'old': left, 'new': right})

extra = []
for folder in ('src', 'scripts', 'tests'):
    for path in (old / folder).rglob('*'):
        if '__pycache__' not in path.parts and (path.is_file() or path.is_symlink()):
            name = str(path.relative_to(old))
            if name not in tracked:
                extra.append(name)

protected = {}
for name in ('.venv', 'logs', 'wandb', 'videos', 'onnx', 'motions'):
    path = old / name
    protected[name] = {
        'exists': path.exists(), 'is_symlink': path.is_symlink(),
        'location': str(path.resolve()),
    }
result = {
    'old_directory': str(old), 'new_directory': str(new),
    'head': subprocess.check_output(
        ['git', '-C', str(new), 'rev-parse', 'HEAD'], text=True
    ).strip(),
    'old_python_used_for_audit_only': sys.executable,
    'same_count': len(same), 'different_count': len(different),
    'missing_in_old_count': len(missing), 'old_extra_source_count': len(extra),
    'different': different, 'missing_in_old': missing,
    'old_extra_source': sorted(extra), 'protected_old_paths': protected,
    'new_environment_created': False, 'training_started': False,
    'clone_http_version_requested': 'HTTP/1.1', 'clone_depth_requested': 1,
    'next': 'Review differences before environment installation and directory switch',
}
(audit / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
print('代码比对：same=%d different=%d missing_in_old=%d old_extra=%d' % (
    len(same), len(different), len(missing), len(extra)
))
print('旧目录/环境/训练结果保留；新目录仅完成Git克隆，尚未安装环境或切换训练入口。')
PY
printf '首次克隆及比对完成。请回传：%s/comparison.json\n' "$LEGO2_GIT_AUDIT"
