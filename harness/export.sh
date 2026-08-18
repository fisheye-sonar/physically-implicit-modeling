#!/usr/bin/env bash
# harness/export.sh — copy the portable harness into another project.
#
#   bash harness/export.sh /path/to/new-project
#
# What it does:
#   * copies harness/*.md, check.sh, export.sh, theme.py, templates/ into <target>/harness/
#   * STRIPS every "## Local instantiations" section (they point at this project)
#   * copies .claude/settings.local.json only if the target has none
#   * creates the empty research record scaffolding
#   * does NOT write the target's CLAUDE.md — see harness/templates/PROJECT_INIT.md step 3
#
# After running, in the new project: retarget the DENY list in check.sh, write CLAUDE.md, and
# fill in UPSTREAM.md §2 with every edit you had to make to a harness file to fit the project.
# That record is the only real test of the quarantine — do not skip it.

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: bash harness/export.sh /path/to/new-project" >&2
  exit 64
fi

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$1"
DEST="$TARGET/harness"

[ -d "$TARGET" ] || { echo "target does not exist: $TARGET" >&2; exit 66; }

if [ -e "$DEST" ]; then
  echo "refusing to overwrite existing $DEST" >&2
  echo "move it aside first, then diff the two — that diff is your port record." >&2
  exit 73
fi

mkdir -p "$DEST/templates"

# Markdown: copy with the local-pointer sections stripped.
strip_local() {
  awk '/^##[[:space:]]+Local instantiations/ { exit } { print }' "$1" \
    | awk 'BEGIN{blank=0} /^$/{blank++; next} {while(blank-->0) print ""; blank=0; print} END{print ""}'
}

for f in "$SRC"/*.md; do
  strip_local "$f" > "$DEST/$(basename "$f")"
done
for f in "$SRC"/templates/*.md; do
  strip_local "$f" > "$DEST/templates/$(basename "$f")"
done

cp "$SRC/check.sh" "$SRC/export.sh" "$SRC/theme.py" "$DEST/"
chmod +x "$DEST/check.sh" "$DEST/export.sh"

# Empty research record.
mkdir -p "$TARGET/research/findings" "$TARGET/research/directions" "$TARGET/research/scratch"
[ -f "$TARGET/research/GOTCHAS.md" ] || cp "$SRC/templates/gotchas.md" "$TARGET/research/GOTCHAS.md"
[ -f "$TARGET/research/PROGRESS.md" ] || cat > "$TARGET/research/PROGRESS.md" <<EOF
# PROGRESS.md — Session Handoff

> Agent-owned, rewritten freely. Answers **"where is the work right now?"** — not "what's true"
> (that is \`findings/\`). Update it as state changes, never as an end-of-session chore.

_Last updated: $(date +%F) — project initialized from a harness export._
EOF

# Settings: only if the target has none, so we never clobber an existing setup.
if [ -f "$SRC/../.claude/settings.local.json" ] && [ ! -f "$TARGET/.claude/settings.local.json" ]; then
  mkdir -p "$TARGET/.claude"
  cp "$SRC/../.claude/settings.local.json" "$TARGET/.claude/settings.local.json"
  echo "note: copied .claude/settings.local.json — retarget its permission allowlist and hook paths."
fi

cat <<EOF

✓ harness exported to $DEST

Next, per harness/templates/PROJECT_INIT.md:
  1. retarget the DENY list in $DEST/check.sh to this project's vocabulary
  2. write $TARGET/CLAUDE.md (role fork first, then triggers, then mechanics)
  3. run: bash $DEST/check.sh
  4. record every harness edit you make to fit this project in $DEST/UPSTREAM.md §2
EOF
