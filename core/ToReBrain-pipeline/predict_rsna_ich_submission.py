from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    # Make `src` importable regardless of current working directory.
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))

    from src.inference.predict_rsna_ich_submission import app

    app()


if __name__ == "__main__":
    main()
