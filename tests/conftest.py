import sys
from pathlib import Path

# Ensure both project root and "src" are on sys.path so imports
# of both styles work: `import src.nlmfsp...` and `import nlmfsp...`.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for p in (SRC, ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# tests/archive/ holds superseded tests kept as development history. One of them
# imports nlmfsp.sde, a module that no longer exists, which aborts collection for
# the whole suite. Skip the directory rather than delete the history.
collect_ignore_glob = ["archive/*"]
