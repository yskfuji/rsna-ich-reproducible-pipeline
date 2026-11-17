# Public Portfolio Hardening Roadmap

## Phase 1 — Findability and first-run experience
- [x] Rewrite the landing README for value-first communication
- [x] Add static overview figures for architecture, repository map, and metrics
- [x] Add a no-data smoke test with a bundled dummy manifest
- [ ] Set GitHub About description/topics/homepage in the web UI
- [ ] Publish GitHub Release notes for the portfolio snapshot tag

## Phase 2 — Trust and reuse
- [x] Add `CITATION.cff`
- [x] Add release-note source files under `docs/releases/`
- [x] Add a public FastAPI demo surface and documentation
- [x] Publish a benchmark summary table in the detailed READMEs
- [x] Add a reproducibility checklist page for external reviewers

## Phase 3 — Evaluation rigor (from the 2026-11 integrity audit)
- [x] Publish per-fold and per-seed values (`core/pipeline/reports/rsna_cv_summary.json`) and check README numbers in CI
- [x] Report every spread as the sample standard deviation (ddof=1)
- [ ] Measure the overlap between the initial checkpoint's training studies and each validation fold
- [ ] Re-run GroupKFold(5) from ImageNet weights, or from an initial checkpoint whose training excludes every validation study
- [ ] Add patient-level grouping (`split_by=patient`); this needs a pseudonymous patient key in the preprocessed metadata
- [ ] Fit temperature scaling on a calibration split that is separate from the evaluation holdout
- [ ] Report study-level bootstrap intervals in addition to the fold-to-fold spread

## Phase 4 — Distribution
- [ ] Post an English launch thread on LinkedIn or X
- [ ] Publish a Qiita/Zenn article summarizing design decisions
- [ ] Track GitHub traffic after external posts
