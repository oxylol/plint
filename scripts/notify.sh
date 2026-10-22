#!/usr/bin/env bash
# Plint · send a push notification.   notify.sh "Title" "Body" [priority] [tags]
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"
load_env
title="${1:-Plint}"; body="${2:-}"; prio="${3:-default}"; tags="${4:-package}"
curl -fsS -o /dev/null --max-time 10 \
  -H "Authorization: Bearer ${NTFY_TOKEN:-}" \
  -H "Title: ${title}" -H "Priority: ${prio}" -H "Tags: ${tags}" \
  -d "${body}" "http://127.0.0.1:8070/plint" || true
