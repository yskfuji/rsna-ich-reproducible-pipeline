#!/usr/bin/env python3
"""Check that headline metrics in the docs match reports/rsna_cv_summary.json.

For every claim, the expected text "mean ± sd" is recomputed from the per-fold /
per-seed values (sample standard deviation, ddof=1) and must appear verbatim in
each listed document. Superseded strings must not appear anywhere, and every
document that shows the numbers must carry the evaluation caveats.
"""

from __future__ import annotations

import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "core/pipeline/reports/rsna_cv_summary.json"

HEADLINE_DOCS = {
    "README.md": "en",
    "README_ja.md": "ja",
    "rsna_ich/README.md": "ja",
    "rsna_ich/README_en.md": "en",
    "docs/assets/results_snapshot.svg": "svg",
}
DETAIL_DOCS = {
    "core/pipeline/README.md": "ja",
    "core/pipeline/README_en.md": "en",
}

CAVEAT_MARKERS = {
    "en": (
        "1 epoch",
        "initial checkpoint",
        "same holdout",
        "not a medical device",
        "rsna_cv_summary.json",
    ),
    "ja": (
        "1エポック",
        "初期重み",
        "同じ検証データ",
        "医療機器ではありません",
        "rsna_cv_summary.json",
    ),
    "svg": ("rsna_cv_summary.json",),
}

# Values that were published with population SD (ddof=0) before 2026-11.
SUPERSEDED = (
    "0.9424 ± 0.0190",
    "0.0231 ± 0.0032",
    "0.0209 ± 0.0054",
    "0.00837 ± 0.00214",
    "2.53 ± 0.89",
)


@dataclass(frozen=True)
class Claim:
    name: str
    values: Callable[[dict[str, Any]], list[float]]
    decimals: int
    headline: bool
    prefix: str = ""
    suffix: str = ""

    def expected(self, summary: dict[str, Any]) -> str:
        xs = self.values(summary)
        mean = statistics.mean(xs)
        sd = statistics.stdev(xs)
        d = self.decimals
        return f"{self.prefix}{mean:.{d}f} ± {sd:.{d}f}{self.suffix}"


def _cv(key: str) -> Callable[[dict[str, Any]], list[float]]:
    return lambda s: [float(f[key]) for f in s["cross_validation"]["folds"]]


def _unc(key: str) -> Callable[[dict[str, Any]], list[float]]:
    return lambda s: [float(r["retrained_with_dropout"][key]) for r in s["uncertainty"]["seeds"]]


def _section_bounds(text: str, pos: int) -> tuple[int, int]:
    """Return the [start, end) span of the Markdown section that contains pos.

    A section runs from its heading to the next heading of the same or a higher
    level. Lines inside fenced code blocks are never treated as headings.
    """
    headings: list[tuple[int, int]] = []
    offset = 0
    in_fence = False
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
        elif not in_fence and stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            if level <= 6 and stripped[level : level + 1] in (" ", "\t"):
                headings.append((offset, level))
        offset += len(line)

    start, level = 0, 0
    for h_off, h_level in headings:
        if h_off > pos:
            break
        start, level = h_off, h_level
    end = len(text)
    for h_off, h_level in headings:
        if h_off > pos and (level == 0 or h_level <= level):
            end = h_off
            break
    return start, end


CLAIMS = (
    Claim("weighted logloss (CV)", _cv("val_logloss_weighted"), 5, True),
    Claim("mean AUC (CV)", _cv("val_auc_mean"), 5, True),
    Claim("error-detection AUROC (any)", _unc("auroc_uncertainty_detect_error_any"), 4, True),
    Claim("ECE (any)", _unc("ece_any"), 4, True),
    Claim("Brier (any)", _unc("brier_any"), 4, False),
    Claim("AURC (weighted logloss)", _unc("aurc_weighted_logloss"), 5, False),
    Claim("accuracy gain at coverage 0.8", _unc("accuracy_any_improve_pp"), 2, False, "+", " pp"),
)


def run_checks(root: Path = ROOT) -> list[str]:
    summary = json.loads((root / SUMMARY.relative_to(ROOT)).read_text(encoding="utf-8"))
    errors: list[str] = []

    n_folds = len(summary["cross_validation"]["folds"])
    n_seeds = len(summary["uncertainty"]["seeds"])
    if n_folds != 5:
        errors.append(f"expected 5 CV folds, found {n_folds}")
    if n_seeds != 10:
        errors.append(f"expected 10 uncertainty seeds, found {n_seeds}")

    docs = {**HEADLINE_DOCS, **DETAIL_DOCS}
    texts: dict[str, str] = {}
    for rel in docs:
        path = root / rel
        if not path.exists():
            errors.append(f"missing document: {rel}")
            continue
        texts[rel] = path.read_text(encoding="utf-8")

    for claim in CLAIMS:
        want = claim.expected(summary)
        targets = list(DETAIL_DOCS) + (list(HEADLINE_DOCS) if claim.headline else [])
        for rel in targets:
            if rel in texts and want not in texts[rel]:
                errors.append(f"{rel}: '{claim.name}' should read '{want}'")

    for rel, lang in docs.items():
        text = texts.get(rel, "")
        for marker in CAVEAT_MARKERS[lang]:
            if marker not in text:
                errors.append(f"{rel}: missing caveat marker '{marker}'")

    # In headline Markdown docs the caveats must sit in the same section as the numbers.
    for rel, lang in HEADLINE_DOCS.items():
        text = texts.get(rel)
        if text is None or lang == "svg":
            continue
        spans = set()
        for claim in CLAIMS:
            if not claim.headline:
                continue
            pos = text.find(claim.expected(summary))
            if pos >= 0:
                spans.add(_section_bounds(text, pos))
        for start, end in sorted(spans):
            section = text[start:end]
            title = section.splitlines()[0] if start else "(top of file)"
            for marker in CAVEAT_MARKERS[lang]:
                if marker not in section:
                    errors.append(f"{rel}: section '{title}' lacks caveat marker '{marker}'")

    for path in sorted([*root.glob("**/*.md"), *root.glob("**/*.svg")]):
        if ".git" in path.parts or "audit" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for old in SUPERSEDED:
            if old in text:
                errors.append(f"{path.relative_to(root)}: superseded value '{old}'")
    return errors


def main() -> int:
    errors = run_checks()
    if errors:
        print("metric claim check FAILED:")
        for e in errors:
            print(f"- {e}")
        return 1
    print(f"metric claim check passed ({len(CLAIMS)} claims)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
