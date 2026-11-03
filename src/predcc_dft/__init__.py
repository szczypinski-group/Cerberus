"""PreDCC DFT: DFT benchmarking of imine-type equilibria in acetonitrile.

Implements the computational workflow of the SOP "Imine-Type Equilibria in
Acetonitrile" (Part II, sections 5-8): species lists, SMARTS products,
coordinates, CREST/CENSO conformers, ORCA frequencies, structure checks,
DLPNO-CCSD(T)/CBS reference, the functional/basis benchmark matrix and the
error analysis that ends in a production-method candidate.
"""

__version__ = "0.2.0"
SCHEMA_VERSION = 1


def main() -> int:
    """Run the command-line interface."""
    from predcc_dft.cli import main as _main

    return _main()


__all__ = ["SCHEMA_VERSION", "__version__", "main"]
