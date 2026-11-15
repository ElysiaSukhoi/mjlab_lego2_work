#!/usr/bin/env bash
# 第一步只下载/快进代码并记录实际版本；源码比对、环境和运行验收留到第二步。
set -euo pipefail
umask 077

LEGO2_GIT_OLD="${LEGO2_GIT_OLD:-$HOME/zrq/mjlab_lego2}"
LEGO2_GIT_NEW="${LEGO2_GIT_NEW:-$HOME/zrq/mjlab_lego2_git_http1}"
LEGO2_GIT_REPO="${LEGO2_GIT_REPO:-https://github.com/ElysiaSukhoi/mjlab_lego2_work.git}"
LEGO2_GIT_BASE="${LEGO2_GIT_BASE:-$HOME/zrq/backups}"

test -d "$LEGO2_GIT_OLD/src"
# 不允许目标指向旧训练目录、位于其中，或通过软链接间接覆盖旧目录。
test ! -L "$LEGO2_GIT_NEW"
LEGO2_GIT_OLD="$(realpath -e "$LEGO2_GIT_OLD")"
LEGO2_GIT_NEW="$(realpath -m "$LEGO2_GIT_NEW")"
case "$LEGO2_GIT_NEW/" in
  "$LEGO2_GIT_OLD/"*)
    printf '目标不能是旧训练目录或其子目录：%s\n' "$LEGO2_GIT_NEW" >&2
    exit 1 ;;
esac
mkdir -p "$LEGO2_GIT_BASE"
LEGO2_GIT_AUDIT="$(mktemp -d "$LEGO2_GIT_BASE/git_stage1_20261115_XXXXXX")"
printf '下载记录目录：%s\n' "$LEGO2_GIT_AUDIT"

if [[ ! -e "$LEGO2_GIT_NEW" ]]; then
  # 临时指定HTTP/1.1，不改变全局设置；失败时不删除任何已有目录。
  if ! git -c http.version=HTTP/1.1 clone --depth 1 --branch main --single-branch \
    "$LEGO2_GIT_REPO" "$LEGO2_GIT_NEW" \
    2>&1 | tee "$LEGO2_GIT_AUDIT/download.txt"; then
    printf '下载中断。请回传报错和记录目录，旧目录与失败现场保留。\n' >&2
    exit 1
  fi
else
  # 已有候选目录只接受正确来源的干净main；不自动合并或覆盖本地修改。
  if [[ ! -d "$LEGO2_GIT_NEW/.git" ]]; then
    printf '目标已存在且不是Git克隆，保留原状：%s\n' "$LEGO2_GIT_NEW" >&2
    exit 1
  fi
  test "$(git -C "$LEGO2_GIT_NEW" rev-parse --show-toplevel)" = "$LEGO2_GIT_NEW"
  test "$(git -C "$LEGO2_GIT_NEW" remote get-url origin)" = "$LEGO2_GIT_REPO"
  test "$(git -C "$LEGO2_GIT_NEW" branch --show-current)" = main
  if [[ -n "$(git -C "$LEGO2_GIT_NEW" status --porcelain --untracked-files=all)" ]]; then
    printf '候选目录有未审阅改动，停止更新：%s\n' "$LEGO2_GIT_NEW" >&2
    exit 1
  fi
  git -C "$LEGO2_GIT_NEW" -c http.version=HTTP/1.1 pull --ff-only origin main \
    2>&1 | tee "$LEGO2_GIT_AUDIT/download.txt"
fi

# 不预设期望SHA，只记录下载后的真实版本；后续回传后再给比对/验收指令。
{
  printf 'directory=%s\n' "$LEGO2_GIT_NEW"
  printf 'branch=%s\n' "$(git -C "$LEGO2_GIT_NEW" branch --show-current)"
  printf 'commit=%s\n' "$(git -C "$LEGO2_GIT_NEW" rev-parse HEAD)"
  printf 'origin_main=%s\n' "$(git -C "$LEGO2_GIT_NEW" rev-parse origin/main)"
  printf 'status:\n'
  git -C "$LEGO2_GIT_NEW" status --short
} | tee "$LEGO2_GIT_AUDIT/receipt.txt"
printf '第一步完成：仅下载并记录版本，尚未比对源码、安装环境或切换训练入口。\n'
printf '请回传以上版本信息与下载记录目录：%s\n' "$LEGO2_GIT_AUDIT"
