"""One Bruker 1D spectrum: read it with nmrglue and integrate ppm ranges.

Processed data (pdata/<procno>/1r, phased and baseline-corrected in TopSpin)
is used when present; otherwise the raw FID is processed here: digital
filter removal, exponential line broadening, zero filling to twice the size,
Fourier transform, automatic phasing (ACME) and a first-order baseline under
each integral. nmrglue is imported here only (``uv sync --extra nmr``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class SpectrumError(ValueError):
    """The experiment folder cannot be read as a Bruker 1D spectrum."""


@dataclass
class Spectrum:
    """Real spectrum on a ppm axis (decreasing, as plotted)."""

    ppm: np.ndarray
    intensity: np.ndarray
    source: str  # "processed" or "raw"
    path: Path


def _nmrglue():  # noqa: ANN202
    try:
        import nmrglue
    except ImportError as exc:
        raise SpectrumError(
            "nmrglue is not installed: uv sync --extra nmr"
        ) from exc
    return nmrglue


def read_processed(exp: Path, procno: int = 1) -> Spectrum:
    """pdata/<procno>/1r with its ppm axis from procs (OFFSET, SW_p, SF)."""
    ng = _nmrglue()
    pdir = exp / "pdata" / str(procno)
    if not (pdir / "1r").exists():
        raise SpectrumError(f"no processed spectrum {pdir / '1r'}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dic, data = ng.bruker.read_pdata(str(pdir), scale_data=True)
    procs = dic["procs"]
    data = np.asarray(data, dtype=float).ravel()
    si = data.size
    step = procs["SW_p"] / procs["SF"] / si
    ppm = procs["OFFSET"] - np.arange(si) * step
    return Spectrum(ppm, data, "processed", exp)


def read_raw(exp: Path, line_broadening_hz: float = 0.3) -> Spectrum:
    """Process the raw FID of an experiment folder into a real spectrum."""
    ng = _nmrglue()
    if not (exp / "fid").exists():
        raise SpectrumError(f"no raw FID {exp / 'fid'}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dic, fid = ng.bruker.read(str(exp))
        fid = ng.bruker.remove_digital_filter(dic, fid)
    acqus = dic["acqus"]
    sw_hz = float(acqus["SW_h"])
    fid = ng.proc_base.em(fid, lb=line_broadening_hz / sw_hz)
    fid = ng.proc_base.zf_size(fid, 2 * fid.size)
    spec = ng.proc_base.rev(ng.proc_base.fft(fid))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = ng.proc_autophase.autops(spec, "acme", disp=False)
    n = spec.size
    sw_ppm = sw_hz / float(acqus["SFO1"])
    centre = float(acqus["O1"]) / float(acqus["BF1"])
    ppm = centre + sw_ppm / 2 - np.arange(n) * sw_ppm / n
    return Spectrum(ppm, np.asarray(spec.real, dtype=float), "raw", exp)


def read(
    exp: Path,
    prefer: str = "auto",
    procno: int = 1,
    line_broadening_hz: float = 0.3,
) -> Spectrum:
    """Spectrum of a Bruker experiment folder.

    prefer: "processed" (pdata only), "raw" (FID only) or "auto"
    (processed if present, else raw).
    """
    if not exp.is_dir():
        raise SpectrumError(f"no experiment folder {exp}")
    has_pdata = (exp / "pdata" / str(procno) / "1r").exists()
    if prefer == "processed" or (prefer == "auto" and has_pdata):
        return read_processed(exp, procno)
    if prefer in ("raw", "auto"):
        return read_raw(exp, line_broadening_hz)
    raise SpectrumError(f"spectrum must be auto, processed or raw: {prefer}")


def integrate(spec: Spectrum, lo: float, hi: float) -> float:
    """Area of a ppm range above a straight baseline between its edges."""
    lo, hi = min(lo, hi), max(lo, hi)
    mask = (spec.ppm >= lo) & (spec.ppm <= hi)
    if mask.sum() < 3:
        raise SpectrumError(f"range {lo}-{hi} ppm is outside the spectrum")
    x, y = spec.ppm[mask], spec.intensity[mask]
    order = np.argsort(x)
    x, y = x[order], y[order]
    edge = max(1, len(x) // 50)  # baseline from the outer 2 % at each end
    base = np.interp(
        x,
        [x[:edge].mean(), x[-edge:].mean()],
        [y[:edge].mean(), y[-edge:].mean()],
    )
    return float(np.trapezoid(y - base, x))
