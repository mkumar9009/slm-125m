#!/usr/bin/env bash
# Stop hook: commit + push whatever changed this turn.
#
# Guard rail: the repo is PUBLIC and the push is irreversible, so staged content is
# scanned for credentials FIRST. A committed secret is already a problem even before
# it is pushed, so a hit aborts the commit -- it does not merely defer the push.
#
# Exits 0 always. A hook that hard-fails would wedge every turn.

set -uo pipefail

REPO="/home/floweraura/code_repos/slm"
cd "$REPO" 2>/dev/null || exit 0

# Emit a JSON systemMessage for the Claude Code UI.
say() { printf '{"systemMessage": %s}\n' "$(printf '%s' "$1" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')"; }

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || exit 0

# Never auto-commit on top of a half-finished merge/rebase/cherry-pick.
git_dir=$(git rev-parse --git-dir)
for state in MERGE_HEAD REBASE_HEAD CHERRY_PICK_HEAD BISECT_LOG; do
  if [ -e "$git_dir/$state" ]; then
    say "auto-commit skipped: repo is mid-$state. Finish it, then commit manually."
    exit 0
  fi
done

# Nothing to do? Stay silent -- this fires on every turn.
if [ -z "$(git status --porcelain)" ]; then exit 0; fi

git add -A

# ---------------------------------------------------------------------------
# Secret scan. Runs against STAGED CONTENT, which is exactly what would ship.
# Lengths are tuned to match live credentials but NOT the `as-XXXXXXXX` style
# placeholders in the setup docs (8 chars, below every threshold here).
# ---------------------------------------------------------------------------
SECRETS='(hf_[A-Za-z0-9]{30,}|ak-[A-Za-z0-9_-]{20,}|as-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----)'

hits=$(git diff --cached -U0 | grep -Eo "$SECRETS" | sort -u | head -5)
env_staged=$(git diff --cached --name-only | grep -E '^\.env(\.|$)|\.pem$|\.key$' | head -3)

if [ -n "$hits" ] || [ -n "$env_staged" ]; then
  git reset -q                      # unstage; leave the working tree untouched
  say "BLOCKED: auto-commit aborted -- possible credential in the changes.
${env_staged:+  secret-ish file staged: $env_staged}
${hits:+  pattern matched: $(echo "$hits" | tr '\n' ' ')}
Nothing was committed or pushed. Remove the secret (and .gitignore the file), then commit by hand."
  exit 0
fi

# ---------------------------------------------------------------------------
# Commit. Name the files so the history stays skimmable.
# ---------------------------------------------------------------------------
files=$(git diff --cached --name-only)
n=$(echo "$files" | wc -l | tr -d ' ')
summary=$(echo "$files" | head -3 | tr '\n' ' ' | sed 's/ $//')
[ "$n" -gt 3 ] && summary="$summary +$((n - 3)) more"

git commit -q -m "auto: $summary" \
  -m "Committed automatically by a Claude Code Stop hook at $(date '+%Y-%m-%d %H:%M:%S')." \
  -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" || { say "auto-commit: git commit failed"; exit 0; }

sha=$(git rev-parse --short HEAD)

if ! git remote get-url origin >/dev/null 2>&1; then
  say "auto-commit: $sha ($n file(s)) -- committed locally, no 'origin' remote to push to."
  exit 0
fi

if push_out=$(git push origin HEAD 2>&1); then
  say "auto-commit: pushed $sha -- $n file(s): $summary"
else
  say "auto-commit: committed $sha locally, but PUSH FAILED:
$(echo "$push_out" | tail -2)
Run 'git push' by hand once resolved."
fi
exit 0
