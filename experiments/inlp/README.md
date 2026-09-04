# inlp

**Question:** how large is the linearly readable code (nullspace cascade), and does writing all of it edit? **Status:** done 2026-09-01 — 15–20 orthogonal probes, rank 120–160, and writing all of them lands where one probe lands. Finding: `research/findings/inlp-redundancy.md`; results `scores/inlp_L-dw-20m_frustum.json`. ⚠ The 2026-09-01 run did not persist the fitted cascades (only their statistics); `scripts/inlp_dw.py` now saves each cascade through `ProbeCache` (unverified path, see the finding). A refit-with-save was cancelled by Sevan on 2026-09-02.
