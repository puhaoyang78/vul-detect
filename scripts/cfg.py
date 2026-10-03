"""Short entry point; all work stays in the formal CFG runner."""
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    os.chdir(root)
    from vulnmechanism.cfg_experiment import main

    raise SystemExit(main())
