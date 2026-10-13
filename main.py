"""Run the server from a checkout: `python main.py` (stdio) or `python main.py --transport http`."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from decisionnode_mcp.main import main  # noqa: E402

if __name__ == "__main__":
    main()
