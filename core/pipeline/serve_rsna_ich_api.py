from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))

    from src.inference.serve_rsna_ich_api import main as serve_main

    serve_main()


if __name__ == "__main__":
    main()