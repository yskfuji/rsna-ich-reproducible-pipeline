#!/usr/bin/env python3
"""Regenerate public SVG figures from rsna_cv_summary.json.

Run from the repository root:
  python docs/assets/generate_figures.py

Only package names and numeric scalars from the local JSON are used.
No medical images and no network calls.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "core/pipeline/reports/rsna_cv_summary.json"
OUT = Path(__file__).resolve().parent

FONT = "ui-sans-serif, system-ui, -apple-system, Segoe UI, Helvetica, Arial, sans-serif"
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
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
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
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")


def architecture() -> None:
    stages = [
        (40, TEAL, "1. CT input", "DICOM / SQLite", "HU window + stacks"),
        (270, SKY, "2. Split", "study GroupKFold", "leakage audit"),
        (500, AMBER, "3. Model", "ConvNeXt 2.5D", "weighted logloss"),
        (730, LIME, "4. Calibrate", "temperature + ECE", "MC Dropout"),
        (960, ROSE, "5. Publish", "reports + checks", "no patient images"),
    ]
    boxes = []
    for x, color, title, a, b in stages:
        boxes.append(
            f"""    <rect x="{x}" y="120" rx="20" ry="20" width="200" height="170" fill="{CARD}" stroke="{color}" stroke-width="3"/>
    <circle cx="{x + 36}" cy="158" r="14" fill="{color}"/>
    <text x="{x + 60}" y="166" fill="{TEXT}" font-size="22" font-weight="700">{title}</text>
    <text x="{x + 24}" y="210" fill="{MUTED}" font-size="16">{a}</text>
    <text x="{x + 24}" y="238" fill="{MUTED}" font-size="16">{b}</text>"""
        )
    arrows = []
    for x in (240, 470, 700, 930):
        arrows.append(
            f'    <path d="M{x} 205 H{x + 28}" stroke="{MUTED}" stroke-width="5" stroke-linecap="round"/>'
            f'<path d="M{x + 20} 193 L{x + 34} 205 L{x + 20} 217" fill="{MUTED}"/>'
        )
    body = f"""    <text x="40" y="48" fill="{TEXT}" font-size="32" font-weight="700">What happens in this pipeline</text>
    <text x="40" y="82" fill="{MUTED}" font-size="18">From CT slices to audited metrics — medical images stay off the public repository.</text>
{chr(10).join(boxes)}
{chr(10).join(arrows)}
    <text x="40" y="340" fill="{MUTED}" font-size="15">Public bundle = code + configs + summary JSON. Datasets/, results/, and *.pt weights are excluded.</text>"""
    write("architecture.svg", _svg(1200, 370, "RSNA pipeline architecture", "Five-stage flow from CT input to published audits.", body))


def labels_taxonomy() -> None:
    # Stylized axial head outline — schematic only, not a real scan.
    labels = [
        (780, 110, ROSE, "epidural", "outside dura"),
        (980, 170, AMBER, "subdural", "under dura"),
        (980, 250, SKY, "subarachnoid", "CSF spaces"),
        (780, 320, TEAL, "intraparenchymal", "inside brain"),
        (560, 320, LIME, "intraventricular", "inside ventricles"),
        (560, 110, "#E879F9", "any", "union of subtypes"),
    ]
    callouts = []
    for x, y, color, name, hint in labels:
        callouts.append(
            f"""    <rect x="{x}" y="{y}" rx="14" ry="14" width="190" height="58" fill="{CARD}" stroke="{color}" stroke-width="2"/>
    <text x="{x + 16}" y="{y + 26}" fill="{color}" font-size="18" font-weight="700">{name}</text>
    <text x="{x + 16}" y="{y + 46}" fill="{MUTED}" font-size="14">{hint}</text>"""
        )
    body = f"""    <text x="40" y="48" fill="{TEXT}" font-size="32" font-weight="700">Six labels the model predicts</text>
    <text x="40" y="82" fill="{MUTED}" font-size="18">Schematic head — not a patient image. Classes follow Flanders et al., 2020.</text>
    <ellipse cx="300" cy="230" rx="170" ry="150" fill="#111827" stroke="{SKY}" stroke-width="8"/>
    <ellipse cx="300" cy="230" rx="120" ry="105" fill="#1E293B" stroke="{TEAL}" stroke-width="4"/>
    <ellipse cx="300" cy="230" rx="42" ry="55" fill="#0EA5E9" opacity="0.35"/>
    <path d="M180 160 Q300 90 420 160" fill="none" stroke="{ROSE}" stroke-width="6" stroke-linecap="round"/>
    <path d="M190 280 Q300 340 410 280" fill="none" stroke="{AMBER}" stroke-width="6" stroke-linecap="round"/>
    <path d="M220 210 Q300 250 380 210" fill="none" stroke="{SKY}" stroke-width="5" stroke-dasharray="8 6"/>
    <circle cx="300" cy="210" r="18" fill="{TEAL}" opacity="0.7"/>
    <circle cx="300" cy="245" r="14" fill="{LIME}" opacity="0.8"/>
    <text x="250" y="390" fill="{MUTED}" font-size="15">axial schematic</text>
{chr(10).join(callouts)}
    <text x="40" y="430" fill="{MUTED}" font-size="15">Loss weights subtypes at 1 and any at 2 (community convention; Kaggle only says any is heavier).</text>"""
    write(
        "labels_taxonomy.svg",
        _svg(1200, 460, "RSNA ICH label taxonomy", "Schematic brain with six hemorrhage labels.", body),
    )


def eval_protocol() -> None:
    cards = [
        (40, ROSE, "1 epoch only", "Each fold/seed fine-tunes once", "on a 6,852-image subsample"),
        (320, AMBER, "Shared init", "Same checkpoint for every run", "trained on ~98% of studies"),
        (600, SKY, "Same holdout", "Temperature fit = ECE set", "calibration looks optimistic"),
        (880, TEAL, "Not a device", "Research / portfolio only", "no clinical validation"),
    ]
    parts = []
    for x, color, title, a, b in cards:
        parts.append(
            f"""    <rect x="{x}" y="120" rx="18" ry="18" width="250" height="180" fill="{CARD}" stroke="{color}" stroke-width="3"/>
    <rect x="{x}" y="120" width="250" height="10" fill="{color}"/>
    <text x="{x + 20}" y="175" fill="{TEXT}" font-size="22" font-weight="700">{title}</text>
    <text x="{x + 20}" y="215" fill="{MUTED}" font-size="15">{a}</text>
    <text x="{x + 20}" y="242" fill="{MUTED}" font-size="15">{b}</text>"""
        )
    body = f"""    <text x="40" y="48" fill="{TEXT}" font-size="32" font-weight="700">Read the numbers with these constraints</text>
    <text x="40" y="82" fill="{MUTED}" font-size="18">The headline metrics measure fine-tuning stability, not unseen-study performance.</text>
{chr(10).join(parts)}
    <text x="40" y="340" fill="{MUTED}" font-size="15">Details: docs/evaluation_notes.md · Source scalars: core/pipeline/reports/rsna_cv_summary.json</text>"""
    write(
        "eval_protocol.svg",
        _svg(1200, 370, "Evaluation protocol caveats", "Four constraint cards for interpreting metrics.", body),
    )


def split_study() -> None:
    body = f"""    <text x="40" y="48" fill="{TEXT}" font-size="32" font-weight="700">How the split avoids image-level leakage</text>
    <text x="40" y="82" fill="{MUTED}" font-size="18">Grouping key = StudyInstanceUID. Patient-level split is not implemented yet.</text>

    <rect x="40" y="120" rx="18" ry="18" width="540" height="230" fill="{CARD}" stroke="{TEAL}" stroke-width="3"/>
    <text x="70" y="165" fill="{TEAL}" font-size="22" font-weight="700">This repo (study split)</text>
    <rect x="70" y="190" rx="10" width="140" height="50" fill="{TEAL}"/>
    <text x="95" y="222" fill="{INK}" font-size="16" font-weight="700">Study A</text>
    <rect x="230" y="190" rx="10" width="140" height="50" fill="{TEAL}"/>
    <text x="255" y="222" fill="{INK}" font-size="16" font-weight="700">Study B</text>
    <rect x="390" y="190" rx="10" width="140" height="50" fill="{SKY}"/>
    <text x="415" y="222" fill="{INK}" font-size="16" font-weight="700">Study C</text>
    <text x="70" y="280" fill="{MUTED}" font-size="15">train ← A,B · val ← C</text>
    <text x="70" y="308" fill="{MUTED}" font-size="15">All slices of a study stay on one side.</text>

    <rect x="620" y="120" rx="18" ry="18" width="540" height="230" fill="{CARD}" stroke="{ROSE}" stroke-width="3"/>
    <text x="650" y="165" fill="{ROSE}" font-size="22" font-weight="700">Risky (image split)</text>
    <rect x="650" y="190" rx="8" width="70" height="40" fill="{ROSE}"/>
    <rect x="735" y="190" rx="8" width="70" height="40" fill="{AMBER}"/>
    <rect x="820" y="190" rx="8" width="70" height="40" fill="{ROSE}"/>
    <rect x="905" y="190" rx="8" width="70" height="40" fill="{AMBER}"/>
    <rect x="990" y="190" rx="8" width="70" height="40" fill="{ROSE}"/>
    <text x="650" y="270" fill="{MUTED}" font-size="15">Slices from the same study can land in</text>
    <text x="650" y="298" fill="{MUTED}" font-size="15">both train and val → optimistic scores.</text>

    <text x="40" y="390" fill="{MUTED}" font-size="15">Audit hook: core/pipeline/tools/audit_rsna_split.py reports group and image-id intersections.</text>"""
    write(
        "split_study.svg",
        _svg(1200, 420, "Study-level split vs image-level split", "Comparison of study grouping versus risky image splits.", body),
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
        (40, SKY, "Weighted logloss ↓", logloss, "5 folds", 0.12),
        (320, LIME, "Mean AUC ↑", auc, "5 folds", 0.99),
        (600, AMBER, "Error AUROC ↑", auroc, "10 seeds", 0.94),
        (880, ROSE, "ECE (any) ↓", ece, "10 seeds", 0.08),
    ]
    parts = []
    for x, color, title, value, n, fill in cards:
        bar = int(220 * fill)
        parts.append(
            f"""    <rect x="{x}" y="110" rx="20" ry="20" width="250" height="230" fill="{CARD}"/>
    <text x="{x + 22}" y="150" fill="{MUTED}" font-size="16">{title}</text>
    <text x="{x + 22}" y="205" fill="{TEXT}" font-size="26" font-weight="700">{value}</text>
    <text x="{x + 22}" y="240" fill="{MUTED}" font-size="15">{n}</text>
    <rect x="{x + 22}" y="270" width="206" height="16" rx="8" fill="{LINE}"/>
    <rect x="{x + 22}" y="270" width="{bar}" height="16" rx="8" fill="{color}"/>"""
        )
    body = f"""    <text x="40" y="48" fill="{TEXT}" font-size="32" font-weight="700">3-minute metrics snapshot</text>
    <text x="40" y="82" fill="{MUTED}" font-size="18">Mean ± sample SD (ddof=1). Source: core/pipeline/reports/rsna_cv_summary.json</text>
{chr(10).join(parts)}
    <text x="40" y="380" fill="{MUTED}" font-size="15">1-epoch fine-tuning from a shared initial checkpoint; calibration fitted on the evaluation holdout. See docs/evaluation_notes.md.</text>"""
    write(
        "results_snapshot.svg",
        _svg(1200, 410, "RSNA metrics snapshot", "Four headline metrics as cards with bars.", body),
    )


def cv_folds(summary: dict) -> None:
    folds = summary["cross_validation"]["folds"]
    losses = [float(f["val_logloss_weighted"]) for f in folds]
    aucs = [float(f["val_auc_mean"]) for f in folds]
    # Panels: left logloss, right AUC
    bars_l = []
    bars_r = []
    max_loss = max(losses) * 1.15
    min_auc, max_auc = 0.98, 1.0
    for i, (loss, auc) in enumerate(zip(losses, aucs)):
        h = int(180 * (loss / max_loss))
        x = 90 + i * 90
        bars_l.append(
            f"""    <rect x="{x}" y="{300 - h}" width="55" height="{h}" rx="8" fill="{SKY}"/>
    <text x="{x + 8}" y="325" fill="{MUTED}" font-size="14">f{i}</text>
    <text x="{x}" y="{290 - h}" fill="{TEXT}" font-size="13">{loss:.3f}</text>"""
        )
        h2 = int(180 * ((auc - min_auc) / (max_auc - min_auc)))
        x2 = 620 + i * 90
        bars_r.append(
            f"""    <rect x="{x2}" y="{300 - h2}" width="55" height="{h2}" rx="8" fill="{LIME}"/>
    <text x="{x2 + 8}" y="325" fill="{MUTED}" font-size="14">f{i}</text>
    <text x="{x2}" y="{290 - h2}" fill="{TEXT}" font-size="13">{auc:.4f}</text>"""
        )
    body = f"""    <text x="40" y="48" fill="{TEXT}" font-size="32" font-weight="700">Per-fold view (5-fold GroupKFold)</text>
    <text x="40" y="82" fill="{MUTED}" font-size="18">Lower logloss and higher AUC are better. Values recomputed from rsna_cv_summary.json.</text>
    <text x="90" y="120" fill="{SKY}" font-size="20" font-weight="700">Weighted logloss ↓</text>
    <text x="620" y="120" fill="{LIME}" font-size="20" font-weight="700">Mean AUC ↑</text>
    <line x1="70" y1="300" x2="520" y2="300" stroke="{LINE}" stroke-width="2"/>
    <line x1="600" y1="300" x2="1050" y2="300" stroke="{LINE}" stroke-width="2"/>
{chr(10).join(bars_l)}
{chr(10).join(bars_r)}
    <text x="40" y="370" fill="{MUTED}" font-size="15">Fold SD understates variance because folds share training data (Bengio &amp; Grandvalet, 2004).</text>"""
    write(
        "cv_folds.svg",
        _svg(1200, 400, "Cross-validation fold metrics", "Bar charts of logloss and AUC for five folds.", body),
    )


def uncertainty_seeds(summary: dict) -> None:
    seeds = summary["uncertainty"]["seeds"]
    aurocs = [float(s["retrained_with_dropout"]["auroc_uncertainty_detect_error_any"]) for s in seeds]
    eces = [float(s["retrained_with_dropout"]["ece_any"]) for s in seeds]
    dots = []
    for i, (a, e) in enumerate(zip(aurocs, eces)):
        x = 80 + i * 100
        ya = 280 - int((a - 0.90) / 0.10 * 160)
        ye = 280 - int((0.04 - e) / 0.04 * 160)  # lower ECE → taller
        dots.append(
            f"""    <circle cx="{x}" cy="{ya}" r="10" fill="{AMBER}"/>
    <circle cx="{x}" cy="{ye}" r="10" fill="{ROSE}" opacity="0.85"/>
    <text x="{x - 8}" y="310" fill="{MUTED}" font-size="13">s{i}</text>"""
        )
    body = f"""    <text x="40" y="48" fill="{TEXT}" font-size="32" font-weight="700">Uncertainty sweep across 10 seeds</text>
    <text x="40" y="82" fill="{MUTED}" font-size="18">Amber = error-detection AUROC (higher better). Rose = ECE (lower better → plotted inverted).</text>
    <line x1="60" y1="280" x2="1120" y2="280" stroke="{LINE}" stroke-width="2"/>
{chr(10).join(dots)}
    <rect x="80" y="340" width="14" height="14" fill="{AMBER}"/><text x="104" y="352" fill="{MUTED}" font-size="14">AUROC</text>
    <rect x="200" y="340" width="14" height="14" fill="{ROSE}"/><text x="224" y="352" fill="{MUTED}" font-size="14">ECE (inverted)</text>
    <text x="40" y="385" fill="{MUTED}" font-size="15">MC Dropout n=30 · temperature fitted on the same holdout · source: rsna_cv_summary.json</text>"""
    write(
        "uncertainty_seeds.svg",
        _svg(1200, 410, "Uncertainty metrics by seed", "Per-seed AUROC and ECE markers.", body),
    )


def repo_map() -> None:
    nodes = [
        (40, 130, 260, SKY, "README", "30-second picture", "then quickstart"),
        (340, 130, 240, TEAL, "docs/", "figures · notes", "glossary · audit"),
        (620, 130, 260, AMBER, "core/pipeline/", "train · eval · API", "tests · reports"),
        (920, 130, 240, ROSE, "scripts/", "CI guards", "smoke · claims"),
    ]
    parts = []
    for x, y, w, color, title, a, b in nodes:
        parts.append(
            f"""    <rect x="{x}" y="{y}" rx="18" ry="18" width="{w}" height="160" fill="{CARD}" stroke="{color}" stroke-width="3"/>
    <text x="{x + 22}" y="{y + 48}" fill="{TEXT}" font-size="24" font-weight="700">{title}</text>
    <text x="{x + 22}" y="{y + 88}" fill="{MUTED}" font-size="16">{a}</text>
    <text x="{x + 22}" y="{y + 116}" fill="{MUTED}" font-size="16">{b}</text>"""
        )
    body = f"""    <text x="40" y="48" fill="{TEXT}" font-size="32" font-weight="700">Where to look</text>
    <text x="40" y="82" fill="{MUTED}" font-size="18">Four doors. Open them in order if you have five minutes.</text>
{chr(10).join(parts)}
    <text x="40" y="340" fill="{MUTED}" font-size="15">rsna_ich/ mirrors this landing page for portfolio reviewers who start from that folder.</text>"""
    write("repo_map.svg", _svg(1200, 370, "Repository map", "Four primary directories for reviewers.", body))


def reader_path() -> None:
    rows = [
        (120, SKY, "Hiring manager", "Look at pictures → metrics → caveats", "15 min"),
        (210, TEAL, "ML engineer", "Smoke test → split audit → training code", "45 min"),
        (300, AMBER, "Researcher", "evaluation_notes → JSON → re-run protocol", "2 h"),
    ]
    parts = []
    for y, color, who, path, time in rows:
        parts.append(
            f"""    <rect x="40" y="{y}" rx="16" ry="16" width="1120" height="70" fill="{CARD}"/>
    <rect x="40" y="{y}" width="12" height="70" fill="{color}"/>
    <text x="80" y="{y + 42}" fill="{TEXT}" font-size="22" font-weight="700">{who}</text>
    <text x="360" y="{y + 42}" fill="{MUTED}" font-size="18">{path}</text>
    <text x="1040" y="{y + 42}" fill="{color}" font-size="18" font-weight="700">{time}</text>"""
        )
    body = f"""    <text x="40" y="48" fill="{TEXT}" font-size="32" font-weight="700">Pick your path</text>
    <text x="40" y="82" fill="{MUTED}" font-size="18">Same repository, three depths. Every path ends at the evaluation caveats.</text>
{chr(10).join(parts)}"""
    write("reader_path.svg", _svg(1200, 400, "Reader paths", "Three reviewer personas and suggested paths.", body))


def included_excluded() -> None:
    body = f"""    <text x="40" y="48" fill="{TEXT}" font-size="32" font-weight="700">What ships publicly</text>
    <text x="40" y="82" fill="{MUTED}" font-size="18">If it can identify a patient or recreate a scan, it stays private.</text>
    <rect x="40" y="120" rx="20" ry="20" width="540" height="240" fill="{CARD}" stroke="{LIME}" stroke-width="3"/>
    <text x="70" y="170" fill="{LIME}" font-size="24" font-weight="700">Included</text>
    <text x="70" y="215" fill="{TEXT}" font-size="18">source · configs · tests · CI</text>
    <text x="70" y="250" fill="{TEXT}" font-size="18">summary JSON · static figures</text>
    <text x="70" y="285" fill="{TEXT}" font-size="18">evaluation notes · checklists</text>
    <rect x="620" y="120" rx="20" ry="20" width="540" height="240" fill="{CARD}" stroke="{ROSE}" stroke-width="3"/>
    <text x="650" y="170" fill="{ROSE}" font-size="24" font-weight="700">Excluded</text>
    <text x="650" y="215" fill="{TEXT}" font-size="18">RSNA DICOM / labels</text>
    <text x="650" y="250" fill="{TEXT}" font-size="18">Datasets/ · runs/ · results/ · logs/</text>
    <text x="650" y="285" fill="{TEXT}" font-size="18">trained weights (*.pt)</text>"""
    write(
        "included_excluded.svg",
        _svg(1200, 400, "Included versus excluded artifacts", "Two-column view of public versus private content.", body),
    )


def evidence_chain() -> None:
    body = f"""    <text x="40" y="48" fill="{TEXT}" font-size="32" font-weight="700">How a README number is checked</text>
    <text x="40" y="82" fill="{MUTED}" font-size="18">CI recomputes every headline value from the per-fold / per-seed JSON.</text>
    <rect x="40" y="130" rx="16" width="200" height="100" fill="{CARD}" stroke="{SKY}" stroke-width="3"/>
    <text x="70" y="190" fill="{TEXT}" font-size="18" font-weight="700">README claim</text>
    <path d="M250 180 H300" stroke="{MUTED}" stroke-width="4"/><polygon points="300,172 318,180 300,188" fill="{MUTED}"/>
    <rect x="320" y="130" rx="16" width="220" height="100" fill="{CARD}" stroke="{TEAL}" stroke-width="3"/>
    <text x="350" y="180" fill="{TEXT}" font-size="18" font-weight="700">mean ± sd</text>
    <text x="350" y="205" fill="{MUTED}" font-size="14">recomputed</text>
    <path d="M550 180 H600" stroke="{MUTED}" stroke-width="4"/><polygon points="600,172 618,180 600,188" fill="{MUTED}"/>
    <rect x="620" y="130" rx="16" width="240" height="100" fill="{CARD}" stroke="{AMBER}" stroke-width="3"/>
    <text x="650" y="180" fill="{TEXT}" font-size="18" font-weight="700">rsna_cv_summary</text>
    <text x="650" y="205" fill="{MUTED}" font-size="14">.json</text>
    <path d="M870 180 H920" stroke="{MUTED}" stroke-width="4"/><polygon points="920,172 938,180 920,188" fill="{MUTED}"/>
    <rect x="940" y="130" rx="16" width="220" height="100" fill="{CARD}" stroke="{ROSE}" stroke-width="3"/>
    <text x="970" y="180" fill="{TEXT}" font-size="18" font-weight="700">fold / seed</text>
    <text x="970" y="205" fill="{MUTED}" font-size="14">raw scalars</text>
    <text x="40" y="290" fill="{MUTED}" font-size="15">Guard: scripts/check_metric_claims.py · Hygiene: scripts/check_public_hygiene.py</text>
    <text x="40" y="320" fill="{MUTED}" font-size="15">Caveats must sit in the same Markdown section as the numbers they qualify.</text>"""
    write(
        "evidence_chain.svg",
        _svg(1200, 360, "Evidence chain for metrics", "Flow from README claim to per-fold scalars.", body),
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
