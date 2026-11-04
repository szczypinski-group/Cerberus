"""Entry point for ``python -m predcc_dft``."""

import sys

from predcc_dft.workflow.cli import main

if __name__ == "__main__":
    sys.exit(main())
