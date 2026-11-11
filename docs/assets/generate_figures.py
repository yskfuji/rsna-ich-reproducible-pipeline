#!/usr/bin/env python3
"""Regenerate public figures from rsna_cv_summary.json.

Writes responsive SVG (width=100%, viewBox only) and PNG for GitHub.
Run from the repository root:
  python docs/assets/generate_figures.py

No medical images and no network calls.
"""

from __future__ import annotations

import json
import shutil
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "core/pipeline/reports/rsna_cv_summary.json"
OUT = Path(__file__).resolve().parent

FONT = "Helvetica, Arial, sans-serif"
BG = "#0B1220"
CARD = "#152033"
LINE = "#1F2A3D"
TEXT = "#F1F5F9"
MUTED = "#94A3B8"
TEAL = "#14B8A6"
SKY = "#38BDF8"
AMBER = "#F59E0B"
ROSE = "#F43F5E"
LIME = "#84CC16"
INK = "#0F172A"


def _esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _svg(width: int, height: int, title: str, desc: str, body: str) -> str:
    # width=100% + viewBox only: scales inside GitHub without horizontal overflow.
    # Fixed pixel width/height caused clipping in the README column.
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" preserveAspectRatio="xMidYMid meet" role="img" aria-labelledby="title desc">
  <title id="title">{_esc(title)}</title>
  <desc id="desc">{_esc(desc)}</desc>
  <rect width="{width}" height="{height}" fill="{BG}"/>
  <g font-family="{FONT}">
{body}
  </g>
</svg>
"""


def _mean_sd(xs: list[float], d: int) -> str:
    return f"{statistics.mean(xs):.{d}f} ± {statistics.stdev(xs):.{d}f}"


def write(name: str, content: str) -> None:
    path = OUT / name
    path.write_text(content, encoding="utf-8")
    png = path.with_suffix(".png")
    rsvg = shutil.which("rsvg-convert")
    if rsvg:
        subprocess.run(
            [rsvg, "-w", "1600", str(path), "-o", str(png)],
            check=True,
        )
        print(f"wrote {path.relative_to(ROOT)} + {png.name}")
    else:
        print(f"wrote {path.relative_to(ROOT)} (PNG skipped: no rsvg-convert)")


def architecture() -> None:
    # Two-row layout avoids five-across overflow on narrow GitHub columns.
    stages = [
        (40, 110, TEAL, "1. CT input", "DICOM / SQLite", "HU window + stacks"),
        (340, 110, SKY, "2. Split", "study GroupKFold", "leakage audit"),
        (640, 110, AMBER, "3. Model", "ConvNeXt 2.5D", "weighted logloss"),
        (190, 300, LIME, "4. Calibrate", "temperature + ECE", "MC Dropout"),
        (490, 300, ROSE, "5. Publish", "reports + checks", "no patient images"),
    ]
    boxes = []
    for x, y, color, title, a, b in stages:
        boxes.append(
            f"""    <rect x="{x}" y="{y}" rx="16" ry="16" width="260" height="140" fill="{CARD}" stroke="{color}" stroke-width="3"/>
    <circle cx="{x + 28}" cy="{y + 32}" r="12" fill="{color}"/>
    <text x="{x + 50}" y="{y + 38}" fill="{TEXT}" font-size="20" font-weight="700">{title}</text>
    <text x="{x + 20}" y="{y + 78}" fill="{MUTED}" font-size="15">{a}</text>
    <text x="{x + 20}" y="{y + 104}" fill="{MUTED}" font-size="15">{b}</text>"""
        )
    body = f"""    <text x="40" y="42" fill="{TEXT}" font-size="28" font-weight="700">What happens in this pipeline</text>
    <text x="40" y="74" fill="{MUTED}" font-size="16">CT slices to audited metrics. Medical images stay off the public repository.</text>
{chr(10).join(boxes)}
    <path d="M300 180 H330" stroke="{MUTED}" stroke-width="4" stroke-linecap="round"/>
    <path d="M600 180 H630" stroke="{MUTED}" stroke-width="4" stroke-linecap="round"/>
    <path d="M450 250 V290" stroke="{MUTED}" stroke-width="4" stroke-linecap="round"/>
    <text x="40" y="480" fill="{MUTED}" font-size="14">Public bundle = code + configs + summary JSON. Datasets/, results/, *.pt excluded.</text>"""
    write("architecture.svg", _svg(960, 510, "RSNA pipeline architecture", "Five-stage pipeline overview.", body))


def labels_taxonomy() -> None:
    labels = [
        (TEAL, "epidural", "outside dura"),
        (AMBER, "subdural", "under dura"),
        (SKY, "subarachnoid", "CSF spaces"),
        (LIME, "intraparenchymal", "inside brain"),
        (ROSE, "intraventricular", "ventricles"),
        ("#C084FC", "any", "union of subtypes"),
    ]
    rows = []
    for i, (color, name, hint) in enumerate(labels):
        y = 120 + i * 52
        rows.append(
            f"""    <rect x="520" y="{y}" rx="12" ry="12" width="400" height="44" fill="{CARD}" stroke="{color}" stroke-width="2"/>
    <circle cx="548" cy="{y + 22}" r="8" fill="{color}"/>
    <text x="570" y="{y + 18}" fill="{color}" font-size="16" font-weight="700">{name}</text>
    <text x="570" y="{y + 36}" fill="{MUTED}" font-size="13">{hint}</text>"""
        )
    body = f"""    <text x="40" y="42" fill="{TEXT}" font-size="28" font-weight="700">Six labels the model predicts</text>
    <text x="40" y="74" fill="{MUTED}" font-size="16">Schematic only — not a patient image. Classes follow Flanders et al., 2020.</text>
    <ellipse cx="250" cy="300" rx="150" ry="140" fill="#111827" stroke="{SKY}" stroke-width="6"/>
    <ellipse cx="250" cy="300" rx="105" ry="95" fill="#1E293B" stroke="{TEAL}" stroke-width="3"/>
    <ellipse cx="250" cy="300" rx="36" ry="48" fill="#0EA5E9" opacity="0.35"/>
    <path d="M140 220 Q250 160 360 220" fill="none" stroke="{ROSE}" stroke-width="5" stroke-linecap="round"/>
    <path d="M150 380 Q250 430 350 380" fill="none" stroke="{AMBER}" stroke-width="5" stroke-linecap="round"/>
    <circle cx="250" cy="280" r="16" fill="{TEAL}" opacity="0.75"/>
    <circle cx="250" cy="320" r="12" fill="{LIME}" opacity="0.8"/>
    <text x="190" y="470" fill="{MUTED}" font-size="14">axial schematic</text>
{chr(10).join(rows)}
    <text x="40" y="500" fill="{MUTED}" font-size="13">Loss weights subtypes at 1 and any at 2 (community convention).</text>"""
    write(
        "labels_taxonomy.svg",
        _svg(960, 530, "RSNA ICH label taxonomy", "Schematic head with six hemorrhage labels.", body),
    )


def eval_protocol() -> None:
    cards = [
        (40, 110, ROSE, "1 epoch only", "Fine-tune once per fold/seed", "6,852-image subsample"),
        (260, 110, AMBER, "Shared init", "Same checkpoint every run", "trained on ~98% studies"),
        (480, 110, SKY, "Same holdout", "Temperature fit = ECE set", "calibration optimistic"),
        (700, 110, TEAL, "Not a device", "Research / portfolio only", "no clinical validation"),
    ]
    parts = []
    for x, y, color, title, a, b in cards:
        parts.append(
            f"""    <rect x="{x}" y="{y}" rx="14" ry="14" width="200" height="170" fill="{CARD}" stroke="{color}" stroke-width="3"/>
    <rect x="{x}" y="{y}" width="200" height="8" fill="{color}"/>
    <text x="{x + 14}" y="{y + 48}" fill="{TEXT}" font-size="18" font-weight="700">{title}</text>
    <text x="{x + 14}" y="{y + 88}" fill="{MUTED}" font-size="13">{a}</text>
    <text x="{x + 14}" y="{y + 112}" fill="{MUTED}" font-size="13">{b}</text>"""
        )
    body = f"""    <text x="40" y="42" fill="{TEXT}" font-size="28" font-weight="700">Read numbers with these constraints</text>
    <text x="40" y="74" fill="{MUTED}" font-size="16">Headline metrics measure fine-tuning stability, not unseen-study performance.</text>
{chr(10).join(parts)}
    <text x="40" y="320" fill="{MUTED}" font-size="13">Details: docs/evaluation_notes.md · Source: core/pipeline/reports/rsna_cv_summary.json</text>"""
    write(
        "eval_protocol.svg",
        _svg(960, 360, "Evaluation protocol caveats", "Four constraint cards for metrics.", body),
    )


def split_study() -> None:
    body = f"""    <text x="40" y="42" fill="{TEXT}" font-size="28" font-weight="700">How the split avoids image-level leakage</text>
    <text x="40" y="74" fill="{MUTED}" font-size="16">Grouping key = StudyInstanceUID. Patient-level split is not implemented yet.</text>

    <rect x="40" y="110" rx="16" ry="16" width="420" height="220" fill="{CARD}" stroke="{TEAL}" stroke-width="3"/>
    <text x="64" y="150" fill="{TEAL}" font-size="20" font-weight="700">This repo (study split)</text>
    <rect x="64" y="175" rx="8" width="100" height="40" fill="{TEAL}"/>
    <text x="82" y="201" fill="{INK}" font-size="14" font-weight="700">Study A</text>
    <rect x="180" y="175" rx="8" width="100" height="40" fill="{TEAL}"/>
    <text x="198" y="201" fill="{INK}" font-size="14" font-weight="700">Study B</text>
    <rect x="296" y="175" rx="8" width="100" height="40" fill="{SKY}"/>
    <text x="314" y="201" fill="{INK}" font-size="14" font-weight="700">Study C</text>
    <text x="64" y="250" fill="{MUTED}" font-size="14">train = A,B · val = C</text>
    <text x="64" y="280" fill="{MUTED}" font-size="14">All slices of a study stay on one side.</text>

    <rect x="500" y="110" rx="16" ry="16" width="420" height="220" fill="{CARD}" stroke="{ROSE}" stroke-width="3"/>
    <text x="524" y="150" fill="{ROSE}" font-size="20" font-weight="700">Risky (image split)</text>
    <rect x="524" y="175" rx="6" width="55" height="34" fill="{ROSE}"/>
    <rect x="592" y="175" rx="6" width="55" height="34" fill="{AMBER}"/>
    <rect x="660" y="175" rx="6" width="55" height="34" fill="{ROSE}"/>
    <rect x="728" y="175" rx="6" width="55" height="34" fill="{AMBER}"/>
    <rect x="796" y="175" rx="6" width="55" height="34" fill="{ROSE}"/>
    <text x="524" y="250" fill="{MUTED}" font-size="14">Slices from one study can land in</text>
    <text x="524" y="280" fill="{MUTED}" font-size="14">both train and val → optimistic scores.</text>

    <text x="40" y="370" fill="{MUTED}" font-size="13">Audit: core/pipeline/tools/audit_rsna_split.py</text>"""
    write(
        "split_study.svg",
        _svg(960, 400, "Study-level split vs image-level split", "Comparison of study grouping versus image splits.", body),
    )


def results_snapshot(summary: dict) -> None:
    logloss = _mean_sd([float(f["val_logloss_weighted"]) for f in summary["cross_validation"]["folds"]], 5)
    auc = _mean_sd([float(f["val_auc_mean"]) for f in summary["cross_validation"]["folds"]], 5)
    auroc = _mean_sd(
        [float(s["retrained_with_dropout"]["auroc_uncertainty_detect_error_any"]) for s in summary["uncertainty"]["seeds"]],
        4,
    )
    ece = _mean_sd(
        [float(s["retrained_with_dropout"]["ece_any"]) for s in summary["uncertainty"]["seeds"]],
        4,
    )
    cards = [
        (40, 100, SKY, "Weighted logloss ↓", logloss, "5 folds", 0.12),
        (260, 100, LIME, "Mean AUC ↑", auc, "5 folds", 0.99),
        (480, 100, AMBER, "Error AUROC ↑", auroc, "10 seeds", 0.94),
        (700, 100, ROSE, "ECE (any) ↓", ece, "10 seeds", 0.08),
    ]
    parts = []
    for x, y, color, title, value, n, fill in cards:
        bar = int(160 * fill)
        parts.append(
            f"""    <rect x="{x}" y="{y}" rx="14" ry="14" width="200" height="200" fill="{CARD}"/>
    <text x="{x + 14}" y="{y + 32}" fill="{MUTED}" font-size="13">{title}</text>
    <text x="{x + 14}" y="{y + 78}" fill="{TEXT}" font-size="18" font-weight="700">{value}</text>
    <text x="{x + 14}" y="{y + 108}" fill="{MUTED}" font-size="13">{n}</text>
    <rect x="{x + 14}" y="{y + 140}" width="170" height="12" rx="6" fill="{LINE}"/>
    <rect x="{x + 14}" y="{y + 140}" width="{bar}" height="12" rx="6" fill="{color}"/>"""
        )
    body = f"""    <text x="40" y="42" fill="{TEXT}" font-size="28" font-weight="700">3-minute metrics snapshot</text>
    <text x="40" y="74" fill="{MUTED}" font-size="15">Mean ± sample SD (ddof=1). Source: core/pipeline/reports/rsna_cv_summary.json</text>
{chr(10).join(parts)}
    <text x="40" y="340" fill="{MUTED}" font-size="13">1-epoch fine-tuning from a shared checkpoint; calibration fitted on the evaluation holdout.</text>"""
    write(
        "results_snapshot.svg",
        _svg(960, 370, "RSNA metrics snapshot", "Four headline metrics as cards.", body),
    )


def cv_folds(summary: dict) -> None:
    folds = summary["cross_validation"]["folds"]
    losses = [float(f["val_logloss_weighted"]) for f in folds]
    aucs = [float(f["val_auc_mean"]) for f in folds]
    bars_l, bars_r = [], []
    max_loss = max(losses) * 1.2
    min_auc, max_auc = 0.98, 1.0
    for i, (loss, auc) in enumerate(zip(losses, aucs)):
        h = int(160 * (loss / max_loss))
        x = 80 + i * 70
        bars_l.append(
            f"""    <rect x="{x}" y="{280 - h}" width="45" height="{h}" rx="6" fill="{SKY}"/>
    <text x="{x + 8}" y="305" fill="{MUTED}" font-size="12">f{i}</text>
    <text x="{x - 2}" y="{268 - h}" fill="{TEXT}" font-size="11">{loss:.3f}</text>"""
        )
        h2 = max(8, int(160 * ((auc - min_auc) / (max_auc - min_auc))))
        x2 = 520 + i * 70
        bars_r.append(
            f"""    <rect x="{x2}" y="{280 - h2}" width="45" height="{h2}" rx="6" fill="{LIME}"/>
    <text x="{x2 + 8}" y="305" fill="{MUTED}" font-size="12">f{i}</text>
    <text x="{x2 - 4}" y="{268 - h2}" fill="{TEXT}" font-size="11">{auc:.4f}</text>"""
        )
    body = f"""    <text x="40" y="42" fill="{TEXT}" font-size="28" font-weight="700">Per-fold view (5-fold GroupKFold)</text>
    <text x="40" y="74" fill="{MUTED}" font-size="15">Lower logloss and higher AUC are better. From rsna_cv_summary.json.</text>
    <text x="80" y="110" fill="{SKY}" font-size="16" font-weight="700">Weighted logloss ↓</text>
    <text x="520" y="110" fill="{LIME}" font-size="16" font-weight="700">Mean AUC ↑</text>
    <line x1="60" y1="280" x2="430" y2="280" stroke="{LINE}" stroke-width="2"/>
    <line x1="500" y1="280" x2="870" y2="280" stroke="{LINE}" stroke-width="2"/>
{chr(10).join(bars_l)}
{chr(10).join(bars_r)}
    <text x="40" y="350" fill="{MUTED}" font-size="13">Fold SD understates variance (Bengio &amp; Grandvalet, 2004).</text>"""
    write(
        "cv_folds.svg",
        _svg(960, 380, "Cross-validation fold metrics", "Bar charts of logloss and AUC.", body),
    )


def uncertainty_seeds(summary: dict) -> None:
    seeds = summary["uncertainty"]["seeds"]
    aurocs = [float(s["retrained_with_dropout"]["auroc_uncertainty_detect_error_any"]) for s in seeds]
    eces = [float(s["retrained_with_dropout"]["ece_any"]) for s in seeds]
    dots = []
    for i, (a, e) in enumerate(zip(aurocs, eces)):
        x = 70 + i * 85
        ya = 260 - int((a - 0.90) / 0.10 * 140)
        ye = 260 - int((0.04 - e) / 0.04 * 140)
        dots.append(
            f"""    <circle cx="{x}" cy="{ya}" r="8" fill="{AMBER}"/>
    <circle cx="{x}" cy="{ye}" r="8" fill="{ROSE}" opacity="0.85"/>
    <text x="{x - 6}" y="290" fill="{MUTED}" font-size="12">s{i}</text>"""
        )
    body = f"""    <text x="40" y="42" fill="{TEXT}" font-size="28" font-weight="700">Uncertainty sweep across 10 seeds</text>
    <text x="40" y="74" fill="{MUTED}" font-size="15">Amber = error-detection AUROC ↑. Rose = ECE ↓ (plotted inverted).</text>
    <line x1="50" y1="260" x2="910" y2="260" stroke="{LINE}" stroke-width="2"/>
{chr(10).join(dots)}
    <rect x="70" y="320" width="12" height="12" fill="{AMBER}"/><text x="90" y="331" fill="{MUTED}" font-size="13">AUROC</text>
    <rect x="180" y="320" width="12" height="12" fill="{ROSE}"/><text x="200" y="331" fill="{MUTED}" font-size="13">ECE (inverted)</text>
    <text x="40" y="365" fill="{MUTED}" font-size="13">MC Dropout n=30 · temperature on same holdout · rsna_cv_summary.json</text>"""
    write(
        "uncertainty_seeds.svg",
        _svg(960, 390, "Uncertainty metrics by seed", "Per-seed AUROC and ECE markers.", body),
    )


def repo_map() -> None:
    nodes = [
        (40, 110, 200, SKY, "README", "30-second picture", "then quickstart"),
        (270, 110, 200, TEAL, "docs/", "figures · notes", "glossary · audit"),
        (500, 110, 210, AMBER, "core/pipeline/", "train · eval · API", "tests · reports"),
        (740, 110, 180, ROSE, "scripts/", "CI guards", "smoke · claims"),
    ]
    parts = []
    for x, y, w, color, title, a, b in nodes:
        parts.append(
            f"""    <rect x="{x}" y="{y}" rx="14" ry="14" width="{w}" height="150" fill="{CARD}" stroke="{color}" stroke-width="3"/>
    <text x="{x + 16}" y="{y + 42}" fill="{TEXT}" font-size="20" font-weight="700">{title}</text>
    <text x="{x + 16}" y="{y + 80}" fill="{MUTED}" font-size="14">{a}</text>
    <text x="{x + 16}" y="{y + 106}" fill="{MUTED}" font-size="14">{b}</text>"""
        )
    body = f"""    <text x="40" y="42" fill="{TEXT}" font-size="28" font-weight="700">Where to look</text>
    <text x="40" y="74" fill="{MUTED}" font-size="16">Four doors. Open them in order if you have five minutes.</text>
{chr(10).join(parts)}
    <text x="40" y="300" fill="{MUTED}" font-size="13">rsna_ich/ mirrors this landing page for portfolio reviewers.</text>"""
    write("repo_map.svg", _svg(960, 340, "Repository map", "Four primary directories.", body))


def reader_path() -> None:
    rows = [
        (100, SKY, "Hiring manager", "Pictures → metrics → caveats", "15 min"),
        (180, TEAL, "ML engineer", "Smoke test → split audit → code", "45 min"),
        (260, AMBER, "Researcher", "evaluation_notes → JSON → protocol", "2 h"),
    ]
    parts = []
    for y, color, who, path, time in rows:
        parts.append(
            f"""    <rect x="40" y="{y}" rx="12" ry="12" width="880" height="60" fill="{CARD}"/>
    <rect x="40" y="{y}" width="10" height="60" fill="{color}"/>
    <text x="70" y="{y + 36}" fill="{TEXT}" font-size="18" font-weight="700">{who}</text>
    <text x="300" y="{y + 36}" fill="{MUTED}" font-size="15">{path}</text>
    <text x="820" y="{y + 36}" fill="{color}" font-size="15" font-weight="700">{time}</text>"""
        )
    body = f"""    <text x="40" y="42" fill="{TEXT}" font-size="28" font-weight="700">Pick your path</text>
    <text x="40" y="74" fill="{MUTED}" font-size="16">Same repository, three depths. Every path ends at the evaluation caveats.</text>
{chr(10).join(parts)}"""
    write("reader_path.svg", _svg(960, 360, "Reader paths", "Three reviewer personas.", body))


def included_excluded() -> None:
    body = f"""    <text x="40" y="42" fill="{TEXT}" font-size="28" font-weight="700">What ships publicly</text>
    <text x="40" y="74" fill="{MUTED}" font-size="16">If it can identify a patient or recreate a scan, it stays private.</text>
    <rect x="40" y="110" rx="16" ry="16" width="420" height="200" fill="{CARD}" stroke="{LIME}" stroke-width="3"/>
    <text x="70" y="155" fill="{LIME}" font-size="22" font-weight="700">Included</text>
    <text x="70" y="195" fill="{TEXT}" font-size="16">source · configs · tests · CI</text>
    <text x="70" y="225" fill="{TEXT}" font-size="16">summary JSON · static figures</text>
    <text x="70" y="255" fill="{TEXT}" font-size="16">evaluation notes · checklists</text>
    <rect x="500" y="110" rx="16" ry="16" width="420" height="200" fill="{CARD}" stroke="{ROSE}" stroke-width="3"/>
    <text x="530" y="155" fill="{ROSE}" font-size="22" font-weight="700">Excluded</text>
    <text x="530" y="195" fill="{TEXT}" font-size="16">RSNA DICOM / labels</text>
    <text x="530" y="225" fill="{TEXT}" font-size="16">Datasets/ · runs/ · results/ · logs/</text>
    <text x="530" y="255" fill="{TEXT}" font-size="16">trained weights (*.pt)</text>"""
    write(
        "included_excluded.svg",
        _svg(960, 360, "Included versus excluded artifacts", "Public versus private content.", body),
    )


def evidence_chain() -> None:
    # 2x2 grid avoids four-across overflow.
    boxes = [
        (40, 110, SKY, "1. README claim"),
        (280, 110, TEAL, "2. mean ± sd"),
        (520, 110, AMBER, "3. summary JSON"),
        (760, 110, ROSE, "4. fold / seed"),
    ]
    parts = []
    for x, y, color, title in boxes:
        parts.append(
            f"""    <rect x="{x}" y="{y}" rx="12" width="180" height="90" fill="{CARD}" stroke="{color}" stroke-width="3"/>
    <text x="{x + 16}" y="{y + 52}" fill="{TEXT}" font-size="15" font-weight="700">{title}</text>"""
        )
    body = f"""    <text x="40" y="42" fill="{TEXT}" font-size="28" font-weight="700">How a README number is checked</text>
    <text x="40" y="74" fill="{MUTED}" font-size="16">CI recomputes every headline value from the per-fold / per-seed JSON.</text>
{chr(10).join(parts)}
    <path d="M220 155 H270" stroke="{MUTED}" stroke-width="3"/>
    <path d="M460 155 H510" stroke="{MUTED}" stroke-width="3"/>
    <path d="M700 155 H750" stroke="{MUTED}" stroke-width="3"/>
    <text x="40" y="250" fill="{MUTED}" font-size="13">Guard: scripts/check_metric_claims.py · Hygiene: scripts/check_public_hygiene.py</text>
    <text x="40" y="278" fill="{MUTED}" font-size="13">Caveats must sit in the same Markdown section as the numbers they qualify.</text>"""
    write(
        "evidence_chain.svg",
        _svg(960, 310, "Evidence chain for metrics", "Flow from README claim to scalars.", body),
    )


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    architecture()
    labels_taxonomy()
    eval_protocol()
    split_study()
    results_snapshot(summary)
    cv_folds(summary)
    uncertainty_seeds(summary)
    repo_map()
    reader_path()
    included_excluded()
    evidence_chain()


if __name__ == "__main__":
    main()
