#!/usr/bin/env python3
"""Publication hygiene checks for the tracked tree.

1. No local absolute paths (/Users/, /Volumes/, /home/<name>/, /mnt/<x>/, Windows user profiles).
2. No patient- or site-identifying DICOM keywords in code or data files.
3. No secret-like strings (cloud keys, tokens, private keys).
4. Source packages are not hidden by .gitignore, on both case-sensitive and
   case-insensitive filesystems, while data directories stay ignored.
5. Every relative Markdown link resolves to a tracked path.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]

LOCAL_PATH = re.compile(
    r"/Users/[A-Za-z]|/Volumes/[A-Za-z]|/home/[A-Za-z][A-Za-z0-9_.-]*/|/mnt/[a-z]/|[A-Za-z]:\\Users\\"
)
PATIENT_TAGS = re.compile(
    r"\b(PatientName|PatientID|PatientBirthDate|OtherPatientIDs|AccessionNumber|"
    r"InstitutionName|ReferringPhysicianName)\b"
)
SECRET = re.compile(
    r"AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9_-]{20,}"
    r"|xox[baprs]-[A-Za-z0-9-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----"
)
# Documentation that explains the patient-tag rule itself.
PATIENT_TAG_ALLOW = {"docs/audit/2026-11-repo-integrity.md", "scripts/check_public_hygiene.py"}

# Checked with --no-index: plain check-ignore always reports tracked files as not ignored.
# The probe is an untracked path, so a new module in the package is covered as well.
MUST_NOT_IGNORE = (
    "core/pipeline/src/datasets/rsna_ich_dataset.py",
    "core/pipeline/src/datasets/__init__.py",
    "core/pipeline/src/datasets/_new_module_probe.py",
)
MUST_IGNORE = (
    "Datasets/x",
    "datasets/x",
    "rsna_ich/Datasets/x",
    "core/Datasets/x",
    "core/pipeline/Datasets/x",
    "core/pipeline/datasets/x",
    "core/pipeline/tools/Datasets/x",
    "core/pipeline/data/x",
    "core/pipeline/results/x",
)

MD_LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def _tracked() -> list[str]:
    return _git("ls-files").stdout.splitlines()


def check_text(files: list[str]) -> list[str]:
    errors: list[str] = []
    for rel in files:
        path = ROOT / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if LOCAL_PATH.search(line):
                errors.append(f"{rel}:{i}: local absolute path")
            if SECRET.search(line):
                errors.append(f"{rel}:{i}: secret-like string")
            if rel not in PATIENT_TAG_ALLOW and PATIENT_TAGS.search(line):
                errors.append(f"{rel}:{i}: patient-identifying DICOM keyword")
    return errors


def _ignored(rel: str, ignorecase: str) -> bool:
    proc = _git("-c", f"core.ignorecase={ignorecase}", "check-ignore", "--no-index", "-q", rel)
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"git check-ignore failed for {rel}: {proc.stderr.strip()}")
    return proc.returncode == 0


def check_ignore_rules() -> list[str]:
    errors: list[str] = []
    for ignorecase in ("true", "false"):
        for rel in MUST_NOT_IGNORE:
            if _ignored(rel, ignorecase):
                errors.append(f"{rel} is ignored (core.ignorecase={ignorecase})")
        for rel in MUST_IGNORE:
            if not _ignored(rel, ignorecase):
                errors.append(f"{rel} is not ignored (core.ignorecase={ignorecase})")
    return errors


def check_links(files: list[str]) -> list[str]:
    tracked = set(files)
    tracked_dirs = {p.as_posix() for f in files for p in Path(f).parents}
    errors: list[str] = []
    for rel in files:
        if not rel.endswith(".md"):
            continue
        text = (ROOT / rel).read_text(encoding="utf-8")
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        for target in MD_LINK.findall(text):
            if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
                continue
            path_part = unquote(target.split("#", 1)[0])
            if not path_part:
                continue
            resolved = (ROOT / Path(rel).parent / path_part).resolve()
            try:
                key = resolved.relative_to(ROOT).as_posix()
            except ValueError:
                errors.append(f"{rel}: link escapes repository: {target}")
                continue
            if key not in tracked and key.rstrip("/") not in tracked_dirs:
                errors.append(f"{rel}: broken relative link: {target}")
    return errors


def main() -> int:
    files = _tracked()
    errors = check_text(files) + check_ignore_rules() + check_links(files)
    if errors:
        print("public hygiene check FAILED:")
        for e in errors:
            print(f"- {e}")
        return 1
    print(f"public hygiene check passed ({len(files)} tracked files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
