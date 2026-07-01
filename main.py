from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from guardrails_sensitive_data.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
