# full_vs_pos

**Question:** can one full-state probe set replace the position-only set (PI restricted to the position dims)? **Status:** done 2026-09-01. For the linear probe the position rows are bit-identical (multi-output lstsq decomposes per output), so the pos-only probes were retired and `master_eval` sweeps both dim sets. Results: `scores/`; the decision is recorded in `research/REGISTRY.md` § Probes.
