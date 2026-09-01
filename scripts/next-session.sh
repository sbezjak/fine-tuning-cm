#!/usr/bin/env bash
# Launch a fresh Claude Code session seeded with the "Build session" prompt from
# session-prompts.md, so you don't hand-copy it after /clear.
#
#   scripts/next-session.sh
#
# It extracts the text between the "## Build session" heading's `---` fences and
# passes it as the initial prompt to a new `claude` session in this repo.
set -euo pipefail
cd "$(dirname "$0")/.."

prompt=$(awk '
  /^## Build session:/ {found=1}
  found && /^---$/ {dash++; next}
  found && dash==1 {print}
  found && dash==2 {exit}
' session-prompts.md)

if [ -z "$prompt" ]; then
  echo "next-session.sh: could not find the Build session prompt in session-prompts.md" >&2
  exit 1
fi

exec claude "$prompt"
