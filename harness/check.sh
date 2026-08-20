#!/usr/bin/env bash
# harness/check.sh — quarantine lint for the portable harness.
#
# Rule: a harness file may not name a project-specific noun outside its
# "## Local instantiations" section. Everything above that heading must be
# portable prose; the section itself holds pointers and is deleted on port.
#
# Usage:
#   bash harness/check.sh            # scan every harness markdown file
#   bash harness/check.sh <file>...  # scan specific files
#
# Exit status: 0 clean, 1 violations found.
#
# Maintenance: when a new project noun enters the vocabulary, add it below.
# Known limit: this catches project NOUNS only. A rule phrased generically but
# true only in this domain will pass silently — the port itself is the only real
# test for those (see UPSTREAM.md).

set -uo pipefail

HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Project vocabulary that must not appear in portable prose ────────────────
DENY=(
  # world / domain objects
  'waterfall' 'frustum' 'reflectivit' 'teleport' 'intensity scan' 'ray.?cast'
  # architectures and model names
  '\bGRU\b' '\bRSSM\b' '\bDiT\b' '\bJEPA\b' '\bVAE\b'
  # project concepts
  'world model' 'hidden state' 'latent state' 'editabilit' 'canonical state'
  # note: bare "steering" is NOT in the list — it over-matched "research steering",
  # which is ordinary English. Only the project's latent-steering sense is denied.
  '\bprobe' 'steering vector' 'grad.{0,3}steering' 'edit index' '\bghost\b'
  'collateral' 'fidelity ratio'
  'endogenous' 'omniscient'
  # code / data identifiers
  '\bpim\b' 'obs_intensity' 'clean_obs' 'edits\.obs' '\bh_edit\b'
  # anticipated vocabulary of sibling projects (keep the harness clean of these too)
  'vesicle' 'micrograph' 'tomogram' 'synap' 'neuro'
)

PATTERN="$(IFS='|'; echo "${DENY[*]}")"

if [ "$#" -gt 0 ]; then
  FILES=("$@")
else
  mapfile -t FILES < <(find "$HARNESS_DIR" -name '*.md' | sort)
fi

violations=0
scanned=0

for f in "${FILES[@]}"; do
  [ -f "$f" ] || continue
  scanned=$((scanned + 1))

  # Body = everything before the "## Local instantiations" heading.
  # Match against the LINE ONLY, never a filename-prefixed line: the absolute path
  # is not portable prose, and a deny term appearing in the checkout path made every
  # line of every file a hit.
  hits="$(awk '
      /^##[[:space:]]+Local instantiations/ { exit }
      { printf "%d:%s\n", FNR, $0 }
    ' "$f" | grep -i -E "$PATTERN" || true)"

  if [ -n "$hits" ]; then
    violations=$((violations + 1))
    echo "✗ $f — project vocabulary in portable prose:"
    echo "$hits" | sed "s|^|    $f:|"
    echo
  fi
done

if [ "$violations" -eq 0 ]; then
  echo "✓ harness quarantine clean ($scanned files scanned)"
  exit 0
fi

cat <<'EOF'
Fix by one of:
  - rewrite the rule so it is true in any project (preferred), or
  - move the concrete detail to where it is used (experiments tree, CLAUDE.md,
    or the project gotchas file) and leave a one-line pointer in the file's
    "## Local instantiations" section, or
  - if the term is genuinely generic and the deny-list is over-matching,
    narrow the pattern in harness/check.sh and say why in the commit.
EOF
exit 1
