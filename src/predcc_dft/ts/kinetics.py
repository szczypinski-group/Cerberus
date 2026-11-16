"""Microkinetic model of one condensation (SOP 10.4): rates and concentrations.

Kinetic species (1 M standard state, mol/L):

    A  aldehyde        N  free amine        H  hemiaminal
    I  imine           W  water             T  TFA (free acid)
    S  ammonium trifluoroacetate ion pair (R-NH3+ CF3COO-)

Every channel is an elementary step with mass-action kinetics,

    r = k_f prod[reactants] - k_r prod[products],
    k_f = (kB T / h) exp(-dG_act / RT),   k_r = k_f / K,   K = exp(-dG / RT),

with dG_act measured from the separated reactants (shuttle included), so a
shuttled step is termolecular (addition) or bimolecular (dehydration) and its
rate scales with the shuttle's concentration. k_r from K keeps every channel
thermodynamically consistent: all channels lead to the same equilibrium.

The acid-base pair T + N <=> S is a diffusion-controlled channel with K from
the computed free energies; TFA (2 mol% by default) therefore splits between
free acid and salt by the amine's basicity, and both forms act as shuttles.
Water starts at zero and is produced with the imine, so [W] = [I] when no
water is added.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from predcc_dft.protocol.constants import R_KCAL

SPECIES = ("A", "N", "H", "I", "W", "T", "S")
KB_OVER_H = 2.083661912e10  # Boltzmann / Planck, 1/(K s)


@dataclass
class Channel:
    """One elementary step: reactants -> products through one TS."""

    name: str
    step: str  # addition | dehydration | acid_base
    reactants: list[str]
    products: list[str]
    dg_act: float  # kcal/mol, TS - separated reactants
    dg_rxn: float  # kcal/mol, products - reactants
    k_forward: float | None = None  # fixed k_f (diffusion), else Eyring
    notes: list[str] = field(default_factory=list)

    @property
    def order(self) -> int:
        """Molecularity of the forward reaction."""
        return len(self.reactants)


def eyring(dg_act: float, temperature: float) -> float:
    """Rate constant in M^(1-n) s^-1 from a 1 M standard-state barrier."""
    return KB_OVER_H * temperature * math.exp(-dg_act / (R_KCAL * temperature))


def rate_constants(
    channel: Channel, temperature: float, cap: float
) -> tuple[float, float]:
    """(k_f, k_r) of a channel; k_f is capped at the diffusion limit."""
    if channel.k_forward is not None:
        kf = channel.k_forward
    else:
        kf = eyring(channel.dg_act, temperature)
        if channel.order > 1 and kf > cap:
            kf = cap
    k_eq = math.exp(-channel.dg_rxn / (R_KCAL * temperature))
    return kf, kf / k_eq


def _prod(c: np.ndarray, idx: list[int]) -> float:
    out = 1.0
    for i in idx:
        out *= c[i]
    return out


def _dprod(c: np.ndarray, idx: list[int], j: int) -> float:
    """d/dc_j of prod(c[idx]) (an index may repeat)."""
    total = 0.0
    for pos, i in enumerate(idx):
        if i == j:
            total += _prod(c, idx[:pos] + idx[pos + 1 :])
    return total


class Network:
    """Mass-action ODE system of a list of channels."""

    def __init__(
        self, channels: list[Channel], temperature: float, cap: float
    ) -> None:
        """Index the species and fix every channel's rate constants."""
        self.channels = channels
        pos = {s: i for i, s in enumerate(SPECIES)}
        self.react = [[pos[s] for s in ch.reactants] for ch in channels]
        self.prods = [[pos[s] for s in ch.products] for ch in channels]
        self.k = [rate_constants(ch, temperature, cap) for ch in channels]
        n = len(SPECIES)
        self.nu = np.zeros((n, len(channels)))
        for j, (r, p) in enumerate(zip(self.react, self.prods, strict=True)):
            for i in r:
                self.nu[i, j] -= 1
            for i in p:
                self.nu[i, j] += 1

    def rates(self, c: np.ndarray) -> np.ndarray:
        """Net rate of every channel, mol/(L s)."""
        c = np.clip(c, 0.0, None)
        return np.array(
            [
                kf * _prod(c, r) - kr * _prod(c, p)
                for (kf, kr), r, p in zip(
                    self.k, self.react, self.prods, strict=True
                )
            ]
        )

    def rhs(self, _t: float, c: np.ndarray) -> np.ndarray:
        """dc/dt."""
        return self.nu @ self.rates(c)

    def jac(self, _t: float, c: np.ndarray) -> np.ndarray:
        """d(dc/dt)/dc."""
        c = np.clip(c, 0.0, None)
        n = len(SPECIES)
        dr = np.zeros((len(self.channels), n))
        for m, ((kf, kr), r, p) in enumerate(
            zip(self.k, self.react, self.prods, strict=True)
        ):
            for j in set(r) | set(p):
                dr[m, j] = kf * _dprod(c, r, j) - kr * _dprod(c, p, j)
        return self.nu @ dr


class IntegrationError(RuntimeError):
    """No stiff solver could integrate the network (unphysical input)."""


def _solve(net: Network, c0: np.ndarray, t_end: float, t_eval: np.ndarray):  # noqa: ANN202
    """BDF, then Radau: the first that finishes."""
    messages = []
    for method in ("BDF", "Radau"):
        with warnings.catch_warnings():  # zero rows at t = 0 are harmless
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message=".*[Ss]ingular matrix.*")
            sol = solve_ivp(
                net.rhs,
                (0.0, t_end),
                c0,
                method=method,
                jac=net.jac,
                t_eval=t_eval,
                rtol=1e-8,
                atol=1e-14,
            )
        if sol.success:
            return sol
        messages.append(f"{method}: {sol.message}")
    raise IntegrationError(
        "kinetics integration failed (" + "; ".join(messages) + ")"
    )


def _crossing(t: np.ndarray, y: np.ndarray, target: float) -> float | None:
    """First time y reaches target (log-linear interpolation)."""
    above = np.nonzero(y >= target)[0]
    if not above.size:
        return None
    i = int(above[0])
    if i == 0:
        return float(t[0])
    t0, t1, y0, y1 = t[i - 1], t[i], y[i - 1], y[i]
    if y1 == y0:
        return float(t1)
    f = (target - y0) / (y1 - y0)
    if t0 > 0:
        return float(math.exp(math.log(t0) + f * math.log(t1 / t0)))
    return float(t0 + f * (t1 - t0))


def time_grid(t_end: float, points: int) -> np.ndarray:
    """0 plus log-spaced times from 1 ms (or t_end/1e9) to t_end."""
    start = min(1e-3, t_end * 1e-9)
    return np.concatenate(([0.0], np.geomspace(start, t_end, points - 1)))


def simulate(
    channels: list[Channel],
    c0: dict[str, float],
    temperature: float,
    t_end: float,
    points: int = 200,
    cap: float = 1e9,
    t_equilibrium: float = 1e12,
) -> dict:
    """Concentrations over time, equilibrium, half-lives and fluxes.

    Returns times, concentration arrays per species, the equilibrium state
    (long-time limit), t50/t95 of the approach of [I] to its equilibrium
    value, and each channel's share of its step's integrated net flux.
    """
    net = Network(channels, temperature, cap)
    y0 = np.array([c0.get(s, 0.0) for s in SPECIES], dtype=float)
    t = time_grid(t_end, points)
    sol = _solve(net, y0, t_end, t)
    eq = _solve(net, y0, t_equilibrium, np.array([t_equilibrium])).y[:, -1]
    conc = {s: sol.y[i] for i, s in enumerate(SPECIES)}
    i_eq = float(eq[SPECIES.index("I")])
    rates = np.array([net.rates(sol.y[:, k]) for k in range(len(t))])
    flux = np.trapezoid(rates, t, axis=0)
    share: dict[str, float] = {}
    for step in {ch.step for ch in channels}:
        members = [j for j, ch in enumerate(channels) if ch.step == step]
        total = sum(flux[j] for j in members)
        for j in members:
            share[channels[j].name] = float(flux[j] / total) if total else 0.0
    return {
        "t_s": t,
        "conc": conc,
        "equilibrium": {s: float(eq[i]) for i, s in enumerate(SPECIES)},
        "t50_s": _crossing(t, conc["I"], 0.5 * i_eq) if i_eq > 0 else None,
        "t95_s": _crossing(t, conc["I"], 0.95 * i_eq) if i_eq > 0 else None,
        "flux_share": share,
        "rate_constants": {
            ch.name: k for ch, k in zip(channels, net.k, strict=True)
        },
    }


def shifted(channels: list[Channel], delta: float) -> list[Channel]:
    """Same channels with every TS barrier shifted by delta kcal/mol."""
    return [
        Channel(
            ch.name,
            ch.step,
            ch.reactants,
            ch.products,
            ch.dg_act + (delta if ch.k_forward is None else 0.0),
            ch.dg_rxn,
            ch.k_forward,
            ch.notes,
        )
        for ch in channels
    ]


def acid_base_channel(dg_salt: float, k_diffusion: float) -> Channel:
    """T + N <=> S at the diffusion limit, K from the computed dG."""
    return Channel(
        "acid_base", "acid_base", ["T", "N"], ["S"], 0.0, dg_salt, k_diffusion
    )
