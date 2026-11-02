#!/usr/bin/env bash
# One command, every check: backend unit + integration, frontend lint + types, browser end-to-end.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "▸ backend: pytest"
(cd backend && uv run pytest)
echo "▸ frontend: eslint + tsc"
(cd frontend && pnpm lint && npx tsc --noEmit)
echo "▸ frontend: playwright (emulator + backend + production build start automatically)"
(cd frontend && npx playwright test "$@")
