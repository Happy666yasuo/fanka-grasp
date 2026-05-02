#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

mode="${1:-summary}"

files="00_index.md 01_current_status.md 02_next_actions.md 04_feishu_alignment.md 03_github_references.md"

if [ "$mode" = "full" ]; then
  files="$files 05_full_analysis_20260419.md"
fi

for file in $files; do
  printf '\n===== %s =====\n\n' "$file"
  cat "$file"
done