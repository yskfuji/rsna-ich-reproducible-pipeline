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

# Single-quoted family names so the SVG attribute can use double quotes.
FONT = "'Hiragino Sans', 'Hiragino Kaku Gothic ProN', 'Noto Sans CJK JP', 'Yu Gothic', 'Helvetica Neue', Helvetica, Arial, sans-serif"
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

def T(en: str, ja: str) -> str:
    return ja if LANG == "ja" else en


def _esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _text_width(s: str, size: float) -> float:
    """Approximate rendered width for mixed JA/EN (used only for layout)."""
    w = 0.0
    for ch in s:
        w += size if ord(ch) > 0x2E80 else size * 0.55
    return w


def _wrap(s: str, size: float, max_width: float) -> list[str]:
    """Greedy wrap that prefers breaks at spaces / middle dots / ideographic commas."""
    if _text_width(s, size) <= max_width:
        return [s]
    lines: list[str] = []
    buf = ""
    for ch in s:
        trial = buf + ch
        if _text_width(trial, size) <= max_width:
            buf = trial
            continue
        if buf:
            lines.append(buf)
        buf = ch
    if buf:
        lines.append(buf)
    return lines or [s]


def _multiline(x: float, y: float, lines: list[str], *, fill: str, size: float, weight: str = "400", leading: float | None = None) -> str:
    lead = leading if leading is not None else size * 1.35
    out = []
    for i, line in enumerate(lines):
        yy = y + i * lead
        out.append(
            f'    <text x="{x}" y="{yy}" fill="{fill}" font-size="{size}" font-weight="{weight}">{_esc(line)}</text>'
        )
    return "\n".join(out)


def _arrow(x1: float, y1: float, x2: float, y2: float, color: str = MUTED, width: int = 4) -> str:
    """Line plus triangle tip. Avoids SVG marker clipping on short segments."""
    import math

    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    tip = 10.0
    bx, by = x2 - ux * tip, y2 - uy * tip
    px, py = -uy, ux
    half = 5.0
    p1 = (bx + px * half, by + py * half)
    p2 = (bx - px * half, by - py * half)
    sx, sy = x2 - ux * (tip + 1.0), y2 - uy * (tip + 1.0)
    return (
        f'    <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{sx:.1f}" y2="{sy:.1f}" '
        f'stroke="{color}" stroke-width="{width}" stroke-linecap="round"/>\n'
        f'    <polygon points="{x2:.1f},{y2:.1f} {p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}" fill="{color}"/>'
    )


def _elbow(points: list[tuple[float, float]], color: str = MUTED, width: int = 4) -> str:
    """Orthogonal polyline with an arrow on the final segment."""
    if len(points) < 2:
        raise ValueError("need >=2 points")
    segs: list[str] = []
    for (x1, y1), (x2, y2) in zip(points[:-2], points[1:-1]):
        segs.append(
            f'    <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{width}" stroke-linecap="round"/>'
        )
    x1, y1 = points[-2]
    x2, y2 = points[-1]
    segs.append(_arrow(x1, y1, x2, y2, color, width))
    return "\n".join(segs)


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


LANG = "en"  # set by main()


def write(name: str, content: str) -> None:
    stem = Path(name).stem
    suffix = "_ja" if LANG == "ja" else ""
    path = OUT / f"{stem}{suffix}.svg"
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
    # True top-down (TD) stack: one column, centered arrows, no elbows.
    cx = 480
    card_w, card_h, gap = 880, 78, 36
    x0 = cx - card_w // 2
    stages = [
        (TEAL, T("1. CT input", "1. CT入力"), T("DICOM / SQLite", "DICOM / SQLite"), T("HU window + stacks", "HU窓処理とスライス積上げ")),
        (SKY, T("2. Split", "2. 分割"), T("study GroupKFold", "検査単位のGroupKFold"), T("leakage audit", "情報リークの監査")),
        (AMBER, T("3. Model", "3. モデル"), T("ConvNeXt 2.5D", "ConvNeXt 2.5D"), T("weighted logloss", "加重対数損失")),
        (LIME, T("4. Calibrate", "4. 校正"), T("temperature + ECE", "温度スケーリングとECE"), T("MC Dropout", "MC Dropout")),
        (ROSE, T("5. Publish", "5. 公開"), T("reports + checks", "報告書と検査"), T("no patient images", "患者画像なし")),
    ]
    y0 = 100
    boxes = []
    ys = []
    for i, (color, title, a, b) in enumerate(stages):
        y = y0 + i * (card_h + gap)
        ys.append(y)
        detail = f"{a}  ·  {b}"
        boxes.append(
            f"""    <rect x="{x0}" y="{y}" rx="14" ry="14" width="{card_w}" height="{card_h}" fill="{CARD}" stroke="{color}" stroke-width="3"/>
    <circle cx="{cx}" cy="{y + 22}" r="7" fill="{color}"/>
    <text x="{cx}" y="{y + 48}" fill="{TEXT}" font-size="20" font-weight="700" text-anchor="middle">{_esc(title)}</text>
    <text x="{cx}" y="{y + 68}" fill="{MUTED}" font-size="14" text-anchor="middle">{_esc(detail)}</text>"""
        )
    links = "\n".join(
        _arrow(cx, y + card_h, cx, y + card_h + gap) for y in ys[:-1]
    )
    footer_y = ys[-1] + card_h + 36
    height = footer_y + 24
    body = f"""    <text x="40" y="42" fill="{TEXT}" font-size="28" font-weight="700">{T("What happens in this pipeline", "本パイプラインの処理流れ")}</text>
{_multiline(40, 74, _wrap(T("CT slices to audited metrics. Medical images stay off the public repository.", "CTスライスから監査可能な指標まで。患者画像は公開物に含めない。"), 15, 880), fill=MUTED, size=15)}
{chr(10).join(boxes)}
{links}
{_multiline(40, footer_y, _wrap(T("Public bundle = code + configs + summary JSON. Datasets/, results/, *.pt excluded.", "公開物はコード・設定・要約JSONである。Datasets/、results/、*.pt は含めない。"), 13, 880), fill=MUTED, size=13)}"""
    write(
        "architecture.svg",
        _svg(960, height, T("RSNA pipeline architecture", "RSNAパイプラインの構成"), T("Five-stage pipeline overview.", "五段階の処理概要。"), body),
    )


def labels_taxonomy() -> None:
    labels = [
        (TEAL, "epidural", T("outside dura", "硬膜の外側")),
        (AMBER, "subdural", T("under dura", "硬膜の下")),
        (SKY, "subarachnoid", T("CSF spaces", "髄液腔")),
        (LIME, "intraparenchymal", T("inside brain", "脳実質内")),
        (ROSE, "intraventricular", T("ventricles", "脳室内")),
        ("#C084FC", "any", T("union of subtypes", "サブタイプのいずれかの出血")),
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
    body = f"""    <text x="40" y="42" fill="{TEXT}" font-size="28" font-weight="700">{T("Six labels the model predicts", "モデルが予測する6ラベル")}</text>
    <text x="40" y="74" fill="{MUTED}" font-size="16">{T("Schematic only — not a patient image. Classes follow Flanders et al., 2020.", "模式図であり実患者画像ではない。クラス定義は Flanders et al., 2020 に従う。")}</text>
    <ellipse cx="250" cy="300" rx="150" ry="140" fill="#111827" stroke="{SKY}" stroke-width="6"/>
    <ellipse cx="250" cy="300" rx="105" ry="95" fill="#1E293B" stroke="{TEAL}" stroke-width="3"/>
    <ellipse cx="250" cy="300" rx="36" ry="48" fill="#0EA5E9" opacity="0.35"/>
    <path d="M140 220 Q250 160 360 220" fill="none" stroke="{ROSE}" stroke-width="5" stroke-linecap="round"/>
    <path d="M150 380 Q250 430 350 380" fill="none" stroke="{AMBER}" stroke-width="5" stroke-linecap="round"/>
    <circle cx="250" cy="280" r="16" fill="{TEAL}" opacity="0.75"/>
    <circle cx="250" cy="320" r="12" fill="{LIME}" opacity="0.8"/>
    <text x="190" y="470" fill="{MUTED}" font-size="14">{T("axial schematic", "軸位の模式図")}</text>
{chr(10).join(rows)}
    <text x="40" y="500" fill="{MUTED}" font-size="13">{T("Loss weights subtypes at 1 and any at 2 (community convention).", "損失の重みはサブタイプを1、any を2とする（参加者間の慣例）。")}</text>"""
    write(
        "labels_taxonomy.svg",
        _svg(960, 530, T("RSNA ICH label taxonomy", "RSNA ICHラベル体系"), T("Schematic head with six hemorrhage labels.", "6出血ラベルを示す模式図。"), body),
    )


def eval_protocol() -> None:
    # 2×2 grid: 200px-wide 1×4 cards clipped Japanese body text.
    card_w, card_h, gap_x, gap_y = 440, 150, 20, 18
    x0, y0 = 40, 110
    cards = [
        (ROSE, T("1. One epoch only", "1. 1エポックのみ"), T("Fine-tune once per fold and per seed.", "各fold・各シードで1回だけ追加学習する。"), T("Subsample of 6,852 images.", "対象は6,852枚の部分標本である。")),
        (AMBER, T("2. Shared init", "2. 共通の初期重み"), T("Every run starts from the same checkpoint.", "全実行で同一の初期重みを用いる。"), T("Trained on about 98% of studies.", "検査の約98%で学習済みである。")),
        (SKY, T("3. Same holdout", "3. 同一の検証データ"), T("Temperature fit uses the ECE evaluation set.", "温度推定とECE評価は同一集合である。"), T("Calibration metrics may be optimistic.", "校正指標は楽観的になりうる。")),
        (TEAL, T("4. Not a device", "4. 医療機器ではない"), T("Research and portfolio use only.", "研究およびポートフォリオ用途である。"), T("No clinical validation.", "臨床妥当性は未検証である。")),
    ]
    parts = []
    for i, (color, title, a, b) in enumerate(cards):
        col, row = i % 2, i // 2
        x = x0 + col * (card_w + gap_x)
        y = y0 + row * (card_h + gap_y)
        body_w = card_w - 36
        a_lines = _wrap(a, 14, body_w)
        b_lines = _wrap(b, 14, body_w)
        body_y = y + 78
        parts.append(
            f"""    <rect x="{x}" y="{y}" rx="14" ry="14" width="{card_w}" height="{card_h}" fill="{CARD}" stroke="{color}" stroke-width="3"/>
    <rect x="{x}" y="{y}" width="{card_w}" height="8" fill="{color}"/>
    <text x="{x + 18}" y="{y + 44}" fill="{TEXT}" font-size="18" font-weight="700">{_esc(title)}</text>
{_multiline(x + 18, body_y, a_lines + b_lines, fill=MUTED, size=14)}"""
        )
    footer_y = y0 + 2 * (card_h + gap_y) + 8
    height = footer_y + 28
    sub = T(
        "Headline metrics measure fine-tuning stability, not unseen-study performance.",
        "代表指標は追加学習の安定性を示すものであり、未知の検査に対する性能ではない。",
    )
    body = f"""    <text x="40" y="42" fill="{TEXT}" font-size="28" font-weight="700">{T("Read numbers with these constraints", "数値解釈における制約")}</text>
{_multiline(40, 74, _wrap(sub, 15, 880), fill=MUTED, size=15)}
{chr(10).join(parts)}
    <text x="40" y="{footer_y}" fill="{MUTED}" font-size="13">{T("Details: docs/evaluation_notes.md · Source: rsna_cv_summary.json", "詳細は docs/evaluation_notes_ja.md。出典は rsna_cv_summary.json。")}</text>"""
    write(
        "eval_constraints.svg",
        _svg(960, height, T("Evaluation protocol caveats", "評価手順上の制約"), T("Four constraint cards for metrics.", "指標解釈のための四つの制約。"), body),
    )


def split_study() -> None:
    split_title = T("How the split avoids image-level leakage", "画像単位の情報リークを避ける分割")
    split_sub = T("Grouping key = StudyInstanceUID. Patient-level split is not implemented yet.", "集約鍵は StudyInstanceUID である。患者単位分割は未実装である。")
    body = f"""{_multiline(40, 42, _wrap(split_title, 26, 880), fill=TEXT, size=26, weight="700")}
{_multiline(40, 74, _wrap(split_sub, 15, 880), fill=MUTED, size=15)}

    <rect x="40" y="110" rx="16" ry="16" width="420" height="220" fill="{CARD}" stroke="{TEAL}" stroke-width="3"/>
    <text x="64" y="150" fill="{TEAL}" font-size="20" font-weight="700">{T("This repo (study split)", "本リポジトリ（検査単位分割）")}</text>
    <rect x="64" y="175" rx="8" width="100" height="40" fill="{TEAL}"/>
    <text x="82" y="201" fill="{INK}" font-size="14" font-weight="700">{T("Study A", "検査A")}</text>
    <rect x="180" y="175" rx="8" width="100" height="40" fill="{TEAL}"/>
    <text x="198" y="201" fill="{INK}" font-size="14" font-weight="700">{T("Study B", "検査B")}</text>
    <rect x="296" y="175" rx="8" width="100" height="40" fill="{SKY}"/>
    <text x="314" y="201" fill="{INK}" font-size="14" font-weight="700">{T("Study C", "検査C")}</text>
    <text x="64" y="250" fill="{MUTED}" font-size="14">{T("train = A,B · val = C", "学習はAとB、検証はC")}</text>
    <text x="64" y="280" fill="{MUTED}" font-size="14">{T("All slices of a study stay on one side.", "同一検査のスライスは一方にのみ属する。")}</text>

    <rect x="500" y="110" rx="16" ry="16" width="420" height="220" fill="{CARD}" stroke="{ROSE}" stroke-width="3"/>
    <text x="524" y="150" fill="{ROSE}" font-size="20" font-weight="700">{T("Risky (image split)", "危険な例（画像単位分割）")}</text>
    <rect x="524" y="175" rx="6" width="55" height="34" fill="{ROSE}"/>
    <rect x="592" y="175" rx="6" width="55" height="34" fill="{AMBER}"/>
    <rect x="660" y="175" rx="6" width="55" height="34" fill="{ROSE}"/>
    <rect x="728" y="175" rx="6" width="55" height="34" fill="{AMBER}"/>
    <rect x="796" y="175" rx="6" width="55" height="34" fill="{ROSE}"/>
    <text x="524" y="250" fill="{MUTED}" font-size="14">{T("Slices from one study can land in", "同一検査のスライスが")}</text>
    <text x="524" y="280" fill="{MUTED}" font-size="14">{T("both train and val → optimistic scores.", "学習と検証の双方に入り、成績を過大評価しうる。")}</text>

    <text x="40" y="370" fill="{MUTED}" font-size="13">{T("Audit: core/pipeline/tools/audit_rsna_split.py", "監査スクリプト: core/pipeline/tools/audit_rsna_split.py")}</text>"""
    write(
        "split_study.svg",
        _svg(960, 400, T("Study-level split vs image-level split", "検査単位分割と画像単位分割の対比"), T("Comparison of study grouping versus image splits.", "検査単位集約と画像単位分割の比較。"), body),
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
        (40, 100, SKY, T("Weighted logloss ↓", "加重対数損失 ↓"), logloss, T("5 folds", "5分割"), 0.12),
        (260, 100, LIME, T("Mean AUC ↑", "平均AUC ↑"), auc, T("5 folds", "5分割"), 0.99),
        (480, 100, AMBER, T("Error AUROC ↑", "誤り検出AUROC ↑"), auroc, T("10 seeds", "10シード"), 0.94),
        (700, 100, ROSE, T("ECE (any) ↓", "ECE（any） ↓"), ece, T("10 seeds", "10シード"), 0.08),
    ]
    parts = []
    for x, y, color, title, value, n, fill in cards:
        bar = int(160 * fill)
        title_lines = _wrap(title, 12, 168)
        value_size = 15 if _text_width(value, 17) > 168 else 17
        parts.append(
            f"""    <rect x="{x}" y="{y}" rx="14" ry="14" width="200" height="200" fill="{CARD}" stroke="{color}" stroke-width="2"/>
{_multiline(x + 16, y + 30, title_lines, fill=MUTED, size=12)}
    <text x="{x + 16}" y="{y + 90}" fill="{TEXT}" font-size="{value_size}" font-weight="700">{_esc(value)}</text>
    <text x="{x + 16}" y="{y + 116}" fill="{MUTED}" font-size="13">{_esc(n)}</text>
    <rect x="{x + 16}" y="{y + 148}" width="168" height="12" rx="6" fill="{LINE}"/>
    <rect x="{x + 16}" y="{y + 148}" width="{min(bar, 168)}" height="12" rx="6" fill="{color}"/>"""
        )
    body = f"""    <text x="40" y="42" fill="{TEXT}" font-size="28" font-weight="700">{T("Headline metrics snapshot", "代表指標の要約")}</text>
    <text x="40" y="74" fill="{MUTED}" font-size="15">{T("Mean ± sample SD (ddof=1). Source: core/pipeline/reports/rsna_cv_summary.json", "平均 ± 標本標準偏差（ddof=1）。出典は rsna_cv_summary.json。")}</text>
{chr(10).join(parts)}
    <text x="40" y="340" fill="{MUTED}" font-size="13">{T("1-epoch fine-tuning from a shared checkpoint; calibration fitted on the evaluation holdout.", "共通の初期重みからの1エポック追加学習。校正は評価と同一の検証データで推定する。")}</text>"""
    write(
        "results_snapshot.svg",
        _svg(960, 370, T("RSNA metrics snapshot", "RSNA代表指標の要約"), T("Four headline metrics as cards.", "代表指標を四枚のカードで示す。"), body),
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
    body = f"""{_multiline(40, 42, _wrap(T("Per-fold view (5-fold GroupKFold)", "fold別の結果（検査単位GroupKFold、5分割）"), 24, 880), fill=TEXT, size=24, weight="700")}
{_multiline(40, 74, _wrap(T("Lower logloss and higher AUC are better. From rsna_cv_summary.json.", "対数損失は低いほど、AUCは高いほど良い。出典は rsna_cv_summary.json。"), 14, 880), fill=MUTED, size=14)}
    <text x="80" y="110" fill="{SKY}" font-size="16" font-weight="700">{T("Weighted logloss ↓", "加重対数損失 ↓")}</text>
    <text x="520" y="110" fill="{LIME}" font-size="16" font-weight="700">{T("Mean AUC ↑", "平均AUC ↑")}</text>
    <line x1="60" y1="280" x2="430" y2="280" stroke="{LINE}" stroke-width="2"/>
    <line x1="500" y1="280" x2="870" y2="280" stroke="{LINE}" stroke-width="2"/>
{chr(10).join(bars_l)}
{chr(10).join(bars_r)}
    <text x="40" y="350" fill="{MUTED}" font-size="13">{T("Fold SD understates variance (Bengio &amp; Grandvalet, 2004).", "fold間の標準偏差は分散を過小評価しうる（Bengio &amp; Grandvalet, 2004）。")}</text>"""
    write(
        "cv_folds.svg",
        _svg(960, 380, T("Cross-validation fold metrics", "交差検証のfold別指標"), T("Bar charts of logloss and AUC.", "対数損失とAUCの棒グラフ。"), body),
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
    body = f"""    <text x="40" y="42" fill="{TEXT}" font-size="26" font-weight="700">{T("Uncertainty sweep across 10 seeds", "10シードにわたる不確実性評価")}</text>
{_multiline(40, 74, _wrap(T("Amber = error-detection AUROC ↑. Rose = ECE ↓ (plotted inverted).", "琥珀は誤り検出AUROC（高いほど良い）。薔薇色はECE（低いほど良い；図では反転表示）。"), 14, 880), fill=MUTED, size=14)}
    <line x1="50" y1="260" x2="910" y2="260" stroke="{LINE}" stroke-width="2"/>
{chr(10).join(dots)}
    <rect x="70" y="320" width="12" height="12" fill="{AMBER}"/><text x="90" y="331" fill="{MUTED}" font-size="13">{T("AUROC", "AUROC")}</text>
    <rect x="180" y="320" width="12" height="12" fill="{ROSE}"/><text x="200" y="331" fill="{MUTED}" font-size="13">{T("ECE (inverted)", "ECE（反転表示）")}</text>
    <text x="40" y="365" fill="{MUTED}" font-size="13">{T("MC Dropout n=30 · temperature on same holdout · rsna_cv_summary.json", "MC Dropoutの標本数は30。温度は同一検証データで推定する。出典は rsna_cv_summary.json。")}</text>"""
    write(
        "uncertainty_seeds.svg",
        _svg(960, 390, T("Uncertainty metrics by seed", "シード別の不確実性指標"), T("Per-seed AUROC and ECE markers.", "シードごとのAUROCとECE。"), body),
    )


def repo_map() -> None:
    nodes = [
        (40, 110, 200, SKY, "README", T("overview", "構成の概要"), T("then smoke test", "続いて動作確認")),
        (270, 110, 200, TEAL, "docs/", T("figures · notes", "図と評価ノート"), T("glossary · audit", "用語集と監査")),
        (500, 110, 210, AMBER, "core/pipeline/", T("train · eval · API", "学習・評価・API"), T("tests · reports", "試験と報告")),
        (740, 110, 180, ROSE, "scripts/", T("CI guards", "継続的統合の検査"), T("smoke · claims", "動作確認と指標照合")),
    ]
    parts = []
    for x, y, w, color, title, a, b in nodes:
        parts.append(
            f"""    <rect x="{x}" y="{y}" rx="14" ry="14" width="{w}" height="150" fill="{CARD}" stroke="{color}" stroke-width="3"/>
    <text x="{x + 16}" y="{y + 42}" fill="{TEXT}" font-size="20" font-weight="700">{title}</text>
    <text x="{x + 16}" y="{y + 80}" fill="{MUTED}" font-size="14">{a}</text>
    <text x="{x + 16}" y="{y + 106}" fill="{MUTED}" font-size="14">{b}</text>"""
        )
    body = f"""    <text x="40" y="42" fill="{TEXT}" font-size="28" font-weight="700">{T("Where to look", "参照すべき場所")}</text>
    <text x="40" y="74" fill="{MUTED}" font-size="16">{T("Four primary areas. Review left to right for a short pass.", "主要な四領域である。短時間で確認する場合は左から順に参照する。")}</text>
{chr(10).join(parts)}
    <text x="40" y="300" fill="{MUTED}" font-size="13">{T("rsna_ich/ mirrors this landing page for portfolio reviewers.", "rsna_ich/ はポートフォリオ向けの入口である。")}</text>"""
    write("repo_map.svg", _svg(960, 340, T("Repository map", "リポジトリの案内"), T("Four primary directories.", "主要な四つの領域。"), body))


def reader_path() -> None:
    rows = [
        (100, SKY, T("Hiring manager", "採用担当"), T("Pictures → metrics → caveats", "図 → 指標 → 制約"), T("15 min", "15分程度")),
        (180, TEAL, T("ML engineer", "機械学習技術者"), T("Smoke test → split audit → code", "動作確認 → 分割監査 → 実装"), T("45 min", "45分程度")),
        (260, AMBER, T("Researcher", "研究者"), T("evaluation_notes → JSON → protocol", "評価ノート → JSON → 評価手順"), T("2 h", "2時間程度")),
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
    body = f"""    <text x="40" y="42" fill="{TEXT}" font-size="28" font-weight="700">{T("Reviewer paths", "想定読者と閲覧の順序")}</text>
    <text x="40" y="74" fill="{MUTED}" font-size="16">{T("Same repository, three depths. Every path ends at the evaluation caveats.", "同一リポジトリに三つの閲覧深度がある。いずれの経路も評価上の制約の確認に帰着する。")}</text>
{chr(10).join(parts)}"""
    write("reader_path.svg", _svg(960, 360, T("Reader paths", "想定読者と閲覧経路"), T("Three reviewer personas.", "三つの想定読者。"), body))


def included_excluded() -> None:
    body = f"""    <text x="40" y="42" fill="{TEXT}" font-size="28" font-weight="700">{T("What ships publicly", "公開範囲")}</text>
    <text x="40" y="74" fill="{MUTED}" font-size="16">{T("If it can identify a patient or recreate a scan, it stays private.", "患者を識別しうる情報、または画像を再構成しうるデータは非公開とする。")}</text>
    <rect x="40" y="110" rx="16" ry="16" width="420" height="200" fill="{CARD}" stroke="{LIME}" stroke-width="3"/>
    <text x="70" y="155" fill="{LIME}" font-size="22" font-weight="700">{T("Included", "含むもの")}</text>
    <text x="70" y="195" fill="{TEXT}" font-size="16">{T("source · configs · tests · CI", "ソース・設定・試験・継続的統合")}</text>
    <text x="70" y="225" fill="{TEXT}" font-size="16">{T("summary JSON · static figures", "要約JSON・静的な図")}</text>
    <text x="70" y="255" fill="{TEXT}" font-size="16">{T("evaluation notes · checklists", "評価ノート・点検表")}</text>
    <rect x="500" y="110" rx="16" ry="16" width="420" height="200" fill="{CARD}" stroke="{ROSE}" stroke-width="3"/>
    <text x="530" y="155" fill="{ROSE}" font-size="22" font-weight="700">{T("Excluded", "含めないもの")}</text>
    <text x="530" y="195" fill="{TEXT}" font-size="16">{T("RSNA DICOM / labels", "RSNAのDICOMおよびラベル")}</text>
    <text x="530" y="225" fill="{TEXT}" font-size="16">Datasets/ · runs/ · results/ · logs/</text>
    <text x="530" y="255" fill="{TEXT}" font-size="16">{T("trained weights (*.pt)", "学習済み重み（*.pt）")}</text>"""
    write(
        "included_excluded.svg",
        _svg(960, 360, T("Included versus excluded artifacts", "公開物と非公開物"), T("Public versus private content.", "公開する内容としない内容。"), body),
    )


def evidence_chain() -> None:
    # Single row; connectors span the full 60px gaps between 180-wide cards.
    boxes = [
        (40, 110, SKY, T("1. README claim", "1. READMEの記載")),
        (280, 110, TEAL, T("2. mean ± sd", "2. 平均 ± 標準偏差")),
        (520, 110, AMBER, T("3. summary JSON", "3. 要約JSON")),
        (760, 110, ROSE, T("4. fold / seed", "4. fold / シード")),
    ]
    parts = []
    for x, y, color, title in boxes:
        title_lines = _wrap(title, 15, 148)
        ty = y + 40 if len(title_lines) == 1 else y + 34
        parts.append(
            f"""    <rect x="{x}" y="{y}" rx="12" width="180" height="90" fill="{CARD}" stroke="{color}" stroke-width="3"/>
{_multiline(x + 16, ty, title_lines, fill=TEXT, size=15, weight="700")}"""
        )
    links = "\n".join(
        [
            _arrow(220, 155, 280, 155, width=3),
            _arrow(460, 155, 520, 155, width=3),
            _arrow(700, 155, 760, 155, width=3),
        ]
    )
    body = f"""    <text x="40" y="42" fill="{TEXT}" font-size="28" font-weight="700">{T("How a README number is checked", "READMEの数値が照合される経路")}</text>
    <text x="40" y="74" fill="{MUTED}" font-size="16">{T("CI recomputes every headline value from the per-fold / per-seed JSON.", "継続的統合は、fold別・シード別JSONから代表値を再計算する。")}</text>
{chr(10).join(parts)}
{links}
    <text x="40" y="250" fill="{MUTED}" font-size="13">{T("Guard: scripts/check_metric_claims.py · Hygiene: scripts/check_public_hygiene.py", "照合: scripts/check_metric_claims.py · 衛生: scripts/check_public_hygiene.py")}</text>
    <text x="40" y="278" fill="{MUTED}" font-size="13">{T("Caveats must sit in the same Markdown section as the numbers they qualify.", "制約の記述は、対象数値と同一の節に置く。")}</text>"""
    write(
        "evidence_chain.svg",
        _svg(960, 310, T("Evidence chain for metrics", "指標の根拠連鎖"), T("Flow from README claim to scalars.", "READMEの記載からスカラー値への経路。"), body),
    )


def smoke_flow() -> None:
    """Static replacement for Mermaid flowchart TD (GitHub often fails to render JA Mermaid)."""
    cx = 480
    main_w, main_h, gap = 400, 56, 44
    ys = [70 + i * (main_h + gap) for i in range(5)]
    colors = [SKY, TEAL, AMBER, LIME, ROSE]
    titles = [
        T("1. Install pinned deps", "1. 固定した依存を入れる"),
        T("2. Smoke test (dummy data)", "2. ダミーデータで動作確認"),
        T("3. Import every module", "3. 全モジュールをimport"),
        T("4. Recompute metrics from JSON", "4. JSONから指標を再計算"),
        T("5. All checks pass?", "5. すべて合格か"),
    ]
    parts = []
    for y, color, title in zip(ys, colors, titles):
        x = cx - main_w // 2
        parts.append(
            f"""    <rect x="{x}" y="{y}" rx="14" ry="14" width="{main_w}" height="{main_h}" fill="{CARD}" stroke="{color}" stroke-width="3"/>
    <text x="{cx}" y="{y + 36}" fill="{TEXT}" font-size="16" font-weight="700" text-anchor="middle">{_esc(title)}</text>"""
        )
    branch_w = 360
    by = ys[-1] + main_h + gap + 8
    yes_x, no_x = cx - 400, cx + 40
    yes_t = T("Yes → read evaluation notes", "はい → 評価ノートを読む")
    no_t = T("No → inspect failing output", "いいえ → 失敗した出力を見る")
    parts.append(
        f"""    <rect x="{yes_x}" y="{by}" rx="14" ry="14" width="{branch_w}" height="{main_h}" fill="{CARD}" stroke="{LIME}" stroke-width="3"/>
    <text x="{yes_x + branch_w / 2}" y="{by + 36}" fill="{TEXT}" font-size="15" font-weight="700" text-anchor="middle">{_esc(yes_t)}</text>
    <rect x="{no_x}" y="{by}" rx="14" ry="14" width="{branch_w}" height="{main_h}" fill="{CARD}" stroke="{ROSE}" stroke-width="3"/>
    <text x="{no_x + branch_w / 2}" y="{by + 36}" fill="{TEXT}" font-size="15" font-weight="700" text-anchor="middle">{_esc(no_t)}</text>"""
    )
    links = [_arrow(cx, y + main_h, cx, y + main_h + gap) for y in ys[:-1]]
    yes_cx = yes_x + branch_w // 2
    no_cx = no_x + branch_w // 2
    jy = ys[-1] + main_h + gap // 2
    links.append(_elbow([(cx, ys[-1] + main_h), (cx, jy), (yes_cx, jy), (yes_cx, by)]))
    links.append(_elbow([(cx, jy), (no_cx, jy), (no_cx, by)]))
    height = by + main_h + 30
    body = f"""    <text x="40" y="42" fill="{TEXT}" font-size="28" font-weight="700">{T("Smoke-test path (no medical data)", "動作確認の手順（医療データなし）")}</text>
{chr(10).join(parts)}
{chr(10).join(links)}"""
    write(
        "smoke_flow.svg",
        _svg(960, height, T("Smoke-test flowchart", "動作確認の流れ図"), T("Top-down path without medical data.", "医療データを用いない上から下への確認手順。"), body),
    )


def main() -> None:
    global LANG
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    for lang in ("en", "ja"):
        LANG = lang
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
        smoke_flow()


if __name__ == "__main__":
    main()
