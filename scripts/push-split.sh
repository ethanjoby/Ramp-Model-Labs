#!/usr/bin/env bash
# Publish the frontend/ and backend/ folders to their own GitHub repositories.
#
#   scripts/push-split.sh              # both
#   scripts/push-split.sh frontend     # one of them
#
# Each run adds one commit on top of that repository's main whose tree is exactly the folder's tree at HEAD
# (so commit your work here first). Nothing is force-pushed and the other repository's history is kept.
# Set PUSH_SPLIT_TRAILER to append a line (e.g. a Co-Authored-By) to the generated commit message.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

repo_url() {
  case "$1" in
    frontend) echo "https://github.com/ethanjoby/ascentlabsfrontend.git" ;;
    backend) echo "https://github.com/ethanjoby/ascentlabsbackend.git" ;;
    *) echo "unknown target: $1 (frontend|backend)" >&2; exit 2 ;;
  esac
}

# Tracked files that make no sense outside the main project.
drop_list() {
  case "$1" in
    frontend) echo "firebase-debug.log" ;;
    *) echo "" ;;
  esac
}

targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=(frontend backend)

for name in "${targets[@]}"; do
  url="$(repo_url "$name")"
  if git remote get-url "$name" >/dev/null 2>&1; then git remote set-url "$name" "$url"; else git remote add "$name" "$url"; fi

  parent=""
  if git fetch -q "$name" main 2>/dev/null; then parent="$(git rev-parse FETCH_HEAD)"; fi

  idx="$(mktemp -u)"
  export GIT_INDEX_FILE="$idx"
  git read-tree "HEAD:$name"
  for f in $(drop_list "$name"); do git update-index --force-remove "$f" >/dev/null 2>&1 || true; done
  tree="$(git write-tree)"
  unset GIT_INDEX_FILE
  rm -f "$idx"

  if [ -n "$parent" ] && [ "$(git rev-parse "${parent}^{tree}")" = "$tree" ]; then
    echo "$name: already up to date with $url"
    continue
  fi

  msg="$name snapshot from $(git rev-parse --short HEAD): $(git log -1 --format=%s)"
  if [ -n "${PUSH_SPLIT_TRAILER:-}" ]; then msg="$msg

$PUSH_SPLIT_TRAILER"; fi
  if [ -n "$parent" ]; then
    commit="$(git commit-tree "$tree" -p "$parent" -m "$msg")"
  else
    commit="$(git commit-tree "$tree" -m "$msg")"
  fi
  git push "$name" "${commit}:refs/heads/main"
  echo "$name: pushed $(git rev-parse --short "$commit") -> $url"
done
