"""
Light-curve model for the bicoherence patterns: simulation and bispectrum estimates.

Implements:

    x(t) = C e^{g - s^2/2} [1 + a1 cos(phi) + a2 cos(2 phi + th)]      (flow)
           + 2 Re[ kappa2 e^{i arg} (G * q2)(t) e^{i 2 pi nu0 t} ]     (locked)
           + D (1 + d(t))                                              (slow)
           -> Poisson(x * rate * dt)                                   (counts)

    g   : Gaussian red-noise field (Timmer-Koenig, Lorentzian PSD, or with
          g_psd="powerlaw" a broken power law: flat below pl_fb, f^-pl_slope above)
    phi : 2 pi nu0 t + psi_i, psi_i uniform per segment (coherent-tone limit)
    q2  : g^2 - <g^2>, the quadratic content of the field
    G   : tone-band kernel. 'lorentzian' -> gamma/(gamma + 2 pi i (nu-nu0))
          (driven mode, biphase rotates); 'flat' -> unit response over
          nu0 +/- dband (static/parametric locking, biphase flat).

Self-checks (`python sim_lightcurve.py` runs all of them):
  A  web        trio only (kappa2=0), nu0 in band: both loci, biphase ~ 0,
                cross-arm b2 vs the zero-parameter bilinear formula,
                harmonic-point biphase = -th.
  B  cross      nu0 above band, a1 large: arms alive, ridge at floor.
  B0 nothing    same but a1 small: everything at floor.
  C  hyp-res    a1 small, kappa2 on, Lorentzian G: ridge with product law,
                arms at floor, transverse biphase = arctan(2 pi d / gamma).
  D  hyp-flat   flat G, arg kappa2 = pi/2: ridge flat biphase = -pi/2.
"""

from __future__ import annotations

import json, os, sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

OUT = os.path.dirname(os.path.abspath(__file__))
DT, NSEG, K = 1.0 / 64, 2048, 300
NTOT = NSEG * K
DF = 1.0 / (NSEG * DT)

# (fp, power, poisson, components): rms-normalized power spectrum and its
# per-term decomposition {name: power}, as returned by metrics()
PSD = tuple[np.ndarray, np.ndarray, float, dict[str, np.ndarray]]
# (f, b2, biphase, floor, nu0, psd), as returned by metrics() for plotting
PlotData = tuple[np.ndarray, np.ndarray, np.ndarray, float, float, PSD]


@dataclass(frozen=True)
class Params:
    """Parameters of the light-curve model (terms as in the module docstring).

    The defaults are the `BASE` run. Instances are immutable: derive a variant
    with ``replace(p, nu0=6.0, ...)`` (``dataclasses.replace``, re-exported
    here).

    Parameters
    ----------
    nu0 : float
        Tone (QPO) frequency (Hz).
    a1 : float
        Rms fractional amplitude of the fundamental. The tone factor is
        ``M = 1 + a1 sqrt(2) cos(phi) + a2 sqrt(2) cos(2 phi + th)``.
    a2 : float
        Rms fractional amplitude of the ``2 nu0`` harmonic.
    th : float
        Harmonic phase (rad).
    g_psd : {"lorentzian", "powerlaw"}
        PSD shape of the flow field ``g``, which multiplies the tone through
        ``exp(g - sig**2 / 2)``. See `g_field`.
    sig_g : float
        Rms of ``g``. With ``g_psd="powerlaw"`` it is used only if `pl_norm`
        is None.
    cents : tuple of float
        ``"lorentzian"``: Lorentzian centres (Hz), mirrored about ``f = 0``;
        0 gives zero-centred red noise.
    widths : tuple of float
        ``"lorentzian"``: Lorentzian half-widths (Hz), one per centre.
    wts : tuple of float
        ``"lorentzian"``: relative Lorentzian weights. Only the shape matters,
        since the rms is set by `sig_g`.
    pl_norm : float or None
        ``"powerlaw"``: PSD level below the break, which sets the rms of
        ``g``. If None, only the shape is used and the rms is `sig_g`.
    pl_fb : float
        ``"powerlaw"``: break frequency (Hz); the PSD is flat below it.
    pl_slope : float
        ``"powerlaw"``: index above the break, ``P ~ f**-pl_slope``.
    kappa2 : float
        Rms of the locked term ``kappa2 e^{i argk} (G * q2)``, with
        ``q2 = g**2 - <g**2>``. 0 turns it off.
    kernel : {"lorentzian", "flat"}
        Tone-band kernel ``G``: ``"lorentzian"`` is a driven mode (biphase
        rotates across the ridge); ``"flat"`` is static/parametric locking
        (biphase flat).
    Qres : float
        Quality factor of the Lorentzian kernel, ``gamma = pi nu0 / Qres``.
    argk : float
        Phase of ``kappa2`` (rad).
    dband : float
        Half-width (Hz) of the flat kernel's band ``nu0 +/- dband``.
    D : float
        Mean level of the slow term ``D (1 + nd_rms d(t))``, relative to the
        flow's ``C = 1``. ``d(t)`` is unit-rms red noise.
    nd_rms : float
        Fractional rms of the slow term.
    nd_psd : {"lorentzian", "powerlaw"}
        PSD shape of ``d(t)``. ``"lorentzian"``: Lorentzian of half-width
        `nd_w` centred on `nd_cent` (or flat below `nd_fc`, if set). ``"powerlaw"``:
        broken power law ``pl_psd(1, nd_fb, nd_slope, nd_slope_lo)``,
        truncated above `nd_fc` if set.
    nd_w : float
        ``"lorentzian"``: half-width (Hz) of ``d(t)``'s Lorentzian PSD.
        Unused if `nd_fc` is set.
    nd_cent : float
        ``"lorentzian"``: centre (Hz) of ``d(t)``'s Lorentzian PSD, mirrored
        about ``f = 0``. Default 0 (zero-centred red noise); a positive centre
        gives a peaked component. Unused if `nd_fc` is set.
    nd_fc : float or None
        ``"lorentzian"``: if set, ``d(t)`` is instead flat below `nd_fc` (Hz),
        keeping it below the tone. ``"powerlaw"``: if set, the PSD is zeroed
        at and above `nd_fc`.
    nd_fb : float
        ``"powerlaw"``: break frequency (Hz) of ``d(t)``'s PSD.
    nd_slope : float
        ``"powerlaw"``: index above the break, ``P ~ f**-nd_slope``.
    nd_slope_lo : float
        ``"powerlaw"``: index below the break, ``P ~ f**-nd_slope_lo``
        (0 is flat).
    rate : float
        Poisson count rate (counts/s) per unit ``x``.
    """

    nu0: float = 2.0
    a1: float = 0.25
    a2: float = 0.08
    th: float = 0.7

    g_psd: Literal["lorentzian", "powerlaw"] = "lorentzian"
    sig_g: float = 0.30
    cents: tuple[float, ...] = (0.0, 0.0)
    widths: tuple[float, ...] = (0.15, 1.2)
    wts: tuple[float, ...] = (1.0, 1.0)
    pl_norm: float | None = None
    pl_fb: float = 1.0
    pl_slope: float = 2.0

    kappa2: float = 0.0
    kernel: Literal["lorentzian", "flat"] = "lorentzian"
    Qres: float = 8.0
    argk: float = 0.0
    dband: float = 0.25

    D: float = 0.15
    nd_rms: float = 0.10
    nd_psd: Literal["lorentzian", "powerlaw"] = "lorentzian"
    nd_w: float = 0.15
    nd_cent: float = 0.0
    nd_fc: float | None = None
    nd_fb: float = 0.15
    nd_slope: float = 2.0
    nd_slope_lo: float = 0.0

    rate: float = 6000.0


def tk_noise(
    psd_func: Callable[[np.ndarray], np.ndarray],
    n: int,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Timmer-Koenig Gaussian noise realization with a given PSD shape.

    Draws complex Gaussian Fourier amplitudes scaled by
    ``sqrt(psd_func(f) / 2)``, zeroes the DC bin, keeps the Nyquist bin
    real for even `n`, and inverts.

    Parameters
    ----------
    psd_func : callable
        PSD shape ``P(f)``, evaluated on the rfft frequency grid. The
        ``f = 0`` value is evaluated at the first nonzero frequency.
    n : int
        Number of samples.
    dt : float
        Sample spacing (s).
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    ndarray
        Length-`n` realization with zero mean and unit standard deviation,
        so only the shape of `psd_func` matters.
    """
    f = np.fft.rfftfreq(n, dt)
    amp = np.sqrt(psd_func(np.maximum(f, f[1])) / 2.0)
    F = amp * (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f)))
    F[0] = 0.0
    if n % 2 == 0:
        F[-1] = F[-1].real
    x = np.fft.irfft(F, n)
    return x / np.std(x)


def lor_psd(
    cents: Sequence[float], widths: Sequence[float], weights: Sequence[float]
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a PSD made of Lorentzians mirrored about zero frequency.

    Parameters
    ----------
    cents : sequence of float
        Lorentzian centre frequencies (Hz). Zero-centred components give the
        band-limited red noise used for the flow field ``g`` and the slow
        term ``d(t)``.
    widths : sequence of float
        Lorentzian half-widths (Hz), one per component.
    weights : sequence of float
        Component weights, one per component.

    Returns
    -------
    callable
        ``P(f)``, the sum over components of
        ``a (w / pi) / ((f - c)**2 + w**2)`` plus its mirror at ``-c``.
    """

    def P(f: np.ndarray) -> np.ndarray:
        out = np.zeros_like(f)
        for c, w, a in zip(cents, widths, weights):
            out += a * (w / np.pi) / ((f - c) ** 2 + w**2) + a * (w / np.pi) / (
                (f + c) ** 2 + w**2
            )
        return out

    return P


def pl_psd(
    norm: float, fb: float, slope: float, slope_lo: float = 0.0
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a broken power-law PSD: ``f**-slope_lo`` below the break,
    ``f**-slope`` above.

    Parameters
    ----------
    norm : float
        PSD level at the break (units of ``g**2 / Hz``, one-sided); with
        ``slope_lo = 0`` this is the level everywhere below it.
    fb : float
        Break frequency (Hz).
    slope : float
        Power-law index above the break, ``P ~ f**-slope``.
    slope_lo : float, optional
        Power-law index below the break, ``P ~ f**-slope_lo``. Default 0
        (flat).

    Returns
    -------
    callable
        ``P(f) = norm (f / fb)**-slope_lo`` for ``f <= fb`` and
        ``norm (f / fb)**-slope`` above. Must be evaluated at ``f > 0``
        unless ``slope_lo = 0``.
    """

    def P(f: np.ndarray) -> np.ndarray:
        x = f / fb
        return norm * np.where(x <= 1, x ** -slope_lo if slope_lo else 1.0,
                               np.maximum(x, 1.0) ** -slope)

    return P


def g_field(
    p: Params, rng: np.random.Generator, n: int | None = None
) -> tuple[np.ndarray, float]:
    """Draw the Gaussian flow field ``g`` and return it with its rms.

    ``p.g_psd`` selects the PSD shape:

    - ``"lorentzian"`` : ``lor_psd(cents, widths, wts)``, rms ``sig_g``.
    - ``"powerlaw"`` : ``pl_psd(pl_norm, pl_fb, pl_slope)``. The rms is
      ``sqrt(sum P(f) df)`` over the realizable rfft grid (``df`` to
      Nyquist), so ``pl_norm`` sets the amplitude. If ``pl_norm`` is None
      only the shape is used and the rms is ``sig_g``.

    Parameters
    ----------
    p : Params
        Model parameters.
    rng : numpy.random.Generator
        Random number generator.
    n : int, optional
        Number of samples; defaults to the module global `NTOT`.

    Returns
    -------
    g : ndarray, shape (n,)
        The field.
    sig : float
        Its rms, for the lognormal mean correction ``exp(g - sig**2 / 2)``.
    """
    n = NTOT if n is None else n
    psd, sig = _g_psd(p, n)
    return tk_noise(psd, n, DT, rng) * sig, sig


def _g_psd(
    p: Params, n: int
) -> tuple[Callable[[np.ndarray], np.ndarray], float]:
    """PSD shape and rms of the flow field ``g`` on an `n`-sample grid (see
    `g_field`)."""
    kind = p.g_psd
    if kind == "lorentzian":
        return lor_psd(p.cents, p.widths, p.wts), p.sig_g
    if kind == "powerlaw":
        psd = pl_psd(p.pl_norm or 1.0, p.pl_fb, p.pl_slope)
        if p.pl_norm is None:
            return psd, p.sig_g
        f = np.fft.rfftfreq(n, DT)[1:]
        return psd, float(np.sqrt(np.sum(psd(f)) * (f[1] - f[0])))
    raise ValueError(f"unknown g_psd {kind!r}")


def _d_psd(p: Params) -> Callable[[np.ndarray], np.ndarray]:
    """PSD shape of the unit-rms slow noise ``d(t)`` (see `Params.nd_psd`)."""
    # Lorentzian centred on nd_cent (default 0), or (if nd_fc is set) flat and
    # band-limited to f < nd_fc, i.e. confined below the tone; or a broken
    # power law, optionally truncated at nd_fc
    if p.nd_psd == "powerlaw":
        bpl = pl_psd(1.0, p.nd_fb, p.nd_slope, p.nd_slope_lo)
        return (lambda f: bpl(f) * (f < p.nd_fc)) if p.nd_fc else bpl
    if p.nd_psd != "lorentzian":
        raise ValueError(f"unknown nd_psd {p.nd_psd!r}")
    if p.nd_fc:
        return lambda f: (f < p.nd_fc).astype(float)
    return lor_psd((p.nd_cent,), (p.nd_w,), (1.0,))


def bicoherence(
    x: np.ndarray, dt: float, nseg: int, fmax: float, shot_noise: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Segment-averaged squared bicoherence and biphase of a time series.

    Splits `x` into length-`nseg` segments (each mean-subtracted, no taper)
    and accumulates ``B = <X(f1) X(f2) X*(f1 + f2)>`` on the grid
    ``f1, f2 = k / (nseg dt)``, ``k = 1..nf``, with `nf` chosen so that
    ``f1 + f2`` reaches about ``2 fmax``.

    Parameters
    ----------
    x : ndarray
        Evenly sampled time series. Samples past the last whole segment are
        dropped.
    dt : float
        Sample spacing (s).
    nseg : int
        Segment length in samples.
    fmax : float
        Highest frequency (Hz) on each axis of the grid.
    shot_noise : bool, optional
        Treat `x` as Poisson counts and subtract the shot-noise bispectrum
        from ``B`` before forming ``b2`` and the biphase. Default False.

    Notes
    -----
    Counts drawn from a varying rate ``lam(t)`` carry a shot-noise term in
    the bispectrum (Wirnitzer 1985, JOSA A 2, 14, eqs. 15 and 17; the same
    term appears in Matarrese, Verde & Heavens 1997, MNRAS 290, 651),
    here in per-segment count units:

        B_shot(k, l) = P_lam(k) + P_lam(l) + P_lam(k + l) + S,

    with ``P_lam`` the power of the rate alone and ``S`` the counts per
    segment (the Poisson power level). It is real, so it biases the biphase
    toward 0, and a tone in ``lam`` puts it on the cross arms and the
    hypotenuse. With ``P_lam = P - S``, the correction subtracted is
    ``P(k) + P(l) + P(k + l) - 2 S``, using the segment-averaged power ``P``
    of `x`. The correction is linear, so this equals summing Wirnitzer's
    per-frame estimator over segments. The denominator is left uncorrected,
    so ``b2`` stays normalized by the total (signal plus Poisson) power.
    Dead time breaks the Poisson statistics this assumes.

    Returns
    -------
    f : ndarray, shape (nf,)
        Bin frequencies (Hz). ``b2[i, j]`` is at ``(f[i], f[j])``, so the grid
        index is ``k - 1``.
    b2 : ndarray, shape (nf, nf)
        Squared bicoherence ``|B|**2 / (<|X1 X2|**2> <|X3|**2>)``. Its noise
        floor is about ``1 / ns``.
    biphase : ndarray, shape (nf, nf)
        ``arg B`` (rad).
    psd : ndarray, shape (nseg // 2 + 1,)
        Unnormalized mean ``|X|**2`` per rfft bin. Indexed by rfft bin, so
        bin ``k`` is ``psd[k]``.
    ns : int
        Number of segments averaged.
    """
    nf3 = int(2 * fmax * nseg * dt)
    nf = nf3 // 2
    ns = len(x) // nseg
    segs = x[: ns * nseg].reshape(ns, nseg)
    segs = segs - segs.mean(axis=1, keepdims=True)
    X = np.fft.rfft(segs, axis=1)
    k = np.arange(1, nf + 1)
    idx3 = k[:, None] + k[None, :]
    num = np.zeros((nf, nf), complex)
    d12 = np.zeros((nf, nf))
    d3 = np.zeros((nf, nf))
    for i in range(ns):
        Xi = X[i]
        x1x2 = Xi[k][:, None] * Xi[k][None, :]
        X3 = Xi[idx3]
        num += x1x2 * np.conj(X3)
        d12 += np.abs(x1x2) ** 2
        d3 += np.abs(X3) ** 2
    psd = (np.abs(X) ** 2).mean(axis=0)
    if shot_noise:
        # S: mean counts per segment (taken before the mean subtraction)
        S = x[: ns * nseg].sum() / ns
        b_shot = psd[k][:, None] + psd[k][None, :] + psd[idx3] - 2 * S
        num = num - ns * b_shot
    b2 = np.abs(num) ** 2 / (d12 * d3)
    return k / (nseg * dt), b2, np.angle(num), psd, ns


def build(
    p: Params, seed: int = 11, components: bool = False
) -> (
    tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]
):
    """Simulate a light curve from the model.

    Implements the flow, locked, slow and counts terms of the module
    docstring. Length and binning come from the module globals `NTOT`,
    `NSEG`, `K` and `DT` (callers may override `K` and `NTOT`).

    Parameters
    ----------
    p : Params
        Model parameters, e.g. a `RUNS` entry.
    seed : int, optional
        Seed for the random number generator.
    components : bool, optional
        Also return the expected-count time series of each additive term.
        Default False.

    Returns
    -------
    cts : ndarray, shape (NTOT,)
        Poisson counts per `DT` bin.
    flow_cts : ndarray, shape (NTOT,)
        Expected counts of the flow envelope ``exp(g - sig**2 / 2)`` alone,
        without the tone, the locked or slow terms, or Poisson noise. Used for
        the bilinear and product-law predictions.
    comps : dict of ndarray, shape (NTOT,)
        Only if `components`. Expected counts of each term, without Poisson
        noise: ``"flow"`` (``ell``), ``"QPO"`` (``ell a1 sqrt(2) cos phi``),
        ``"harmonic"`` (``ell a2 sqrt(2) cos(2 phi + th)``), ``"quadratic"``
        (the locked term) and ``"dilution"`` (the slow term
        ``D (1 + nd_rms d)``). Terms whose amplitude is zero are omitted.
        They sum to the expected counts (before the clip at 0.01).

    Notes
    -----
    ``C = 1`` is implicit. The locked term is normalized to rms `kappa2`
    before it is added.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(NTOT) * DT
    g, sig = g_field(p, rng)
    ell = np.exp(g - 0.5 * sig**2)

    # segment-random tone phase (coherent within each segment)
    psi = np.repeat(rng.uniform(0, 2 * np.pi, K), NSEG)
    phi = 2 * np.pi * p.nu0 * t + psi
    M = (
        1.0
        + p.a1 * np.sqrt(2) * np.cos(phi)
        + p.a2 * np.sqrt(2) * np.cos(2 * phi + p.th)
    )
    x = ell * M
    xq = np.zeros(NTOT)

    # locked tone component: kappa2 * (G * q2), tone-band confined by construction
    if p.kappa2 > 0:
        q2 = g * g - np.mean(g * g)
        Q2 = np.fft.rfft(q2)
        f = np.fft.rfftfreq(NTOT, DT)
        if p.kernel == "lorentzian":
            gam = np.pi * p.nu0 / p.Qres
            win = np.abs(f - p.nu0) < 4 * gam / (2 * np.pi)
            G = gam / (gam + 2j * np.pi * (f - p.nu0))
        else:  # flat over nu0 +/- dband
            win = np.abs(f - p.nu0) < p.dband
            G = np.ones_like(f, complex)
        # one-sided masked spectrum -> irfft gives Re[e^{i argk} analytic signal]
        xq = np.fft.irfft(np.where(win, Q2 * G * np.exp(1j * p.argk), 0), NTOT)
        xq = xq / np.std(xq)
        x = x + p.kappa2 * xq

    # slow incoherent power d(t)
    slow = p.D * (1.0 + tk_noise(_d_psd(p), NTOT, DT, rng) * p.nd_rms)
    x = x + slow
    lam = np.clip(x, 0.01, None) * p.rate * DT
    cts = rng.poisson(lam).astype(float)
    flow_cts = ell * p.rate * DT  # flow field alone, count units
    if not components:
        return cts, flow_cts
    terms = {
        "dilution": (p.D, slow),
        "flow": (1.0, ell),
        "QPO": (p.a1, ell * p.a1 * np.sqrt(2) * np.cos(phi)),
        "harmonic": (p.a2, ell * p.a2 * np.sqrt(2) * np.cos(2 * phi + p.th)),
        "quadratic": (p.kappa2, p.kappa2 * xq),
    }
    comps = {k: v * p.rate * DT for k, (amp, v) in terms.items() if amp != 0}
    return cts, flow_cts, comps


def controls(
    p: Params, f: np.ndarray | float | None = None, poisson: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dilution fraction ``eta(nu0 - f)`` and coupling ratio ``Q(f)`` of a
    configuration, as defined in the paper (``eq:dilution_fraction`` and
    ``eq:coupling_ratio``).

    Computed from the model PSDs, not from a realization:

        eta(f) = D^2 P_d(f) / P_0(f),
        Q(f)   = 4 |K2| / (C a1^2) * P_g(f) P_g(f2) / (P_g(f) + P_g(f2)),

    with ``f2 = nu0 - f`` and ``C = 1``. ``P_0`` is the undiluted power to
    linear order in ``g`` (as in the paper's Fourier expansion): the flow
    noise, its sidebands at ``+/- nu0`` and ``+/- 2 nu0``, and the locked
    term ``|K2 G(f)|^2 P_{g^2}(f)``; the tone lines are left out. ``|K2|``
    is the time-domain coupling the sim implies, ``kappa2`` divided by the
    rms of ``G * q2``. ``P_g`` in `Q` is two-sided power per bin at the
    bicoherence resolution `DF`, so `Q` depends on the segment length (the
    tone is coherent, the noise is not).

    Parameters
    ----------
    p : Params
        Model parameters.
    f : float or ndarray, optional
        Frequencies ``f1`` (Hz). Default: the bicoherence bins ``k DF`` up to
        ``nu0 / 2``, the range the paper uses.
    poisson : bool, optional
        Count the Poisson level ``2 (1 + D) / rate`` as dilution, as the paper
        does. Default True.

    Returns
    -------
    f : ndarray
        The frequencies.
    eta : ndarray
        ``eta(nu0 - f)``.
    Q : ndarray
        ``Q(f)``. 0 if ``kappa2 = 0``.
    """
    nu0 = p.nu0
    if f is None:
        f = np.arange(1, int(nu0 / 2 / DF) + 1) * DF
    f = np.atleast_1d(np.asarray(f, float))

    # one-sided densities on the full-length grid, normalized as in tk_noise
    fg = np.fft.rfftfreq(NTOT, DT)
    dfg = fg[1]
    psd_g, sig = _g_psd(p, NTOT)
    shape = psd_g(np.maximum(fg, dfg))
    shape[0] = 0.0
    norm_g = sig**2 / (shape.sum() * dfg)

    def S_g(x: np.ndarray) -> np.ndarray:
        x = np.abs(x)
        return np.where(x < dfg / 2, 0.0, norm_g * psd_g(np.maximum(x, dfg)))

    # PSD of q2 = g^2 - <g^2>: autocovariance 2 R_g^2
    v = shape * norm_g * dfg  # per-bin variance of g
    X = NTOT * v / 2
    X[-1] = NTOT * v[-1]
    Rg = np.fft.irfft(X, NTOT)
    vq = 2 * np.fft.rfft(2 * Rg**2).real / NTOT
    vq[0] = 0.0
    S_q2 = vq / dfg

    # locked term: same kernel and window as build
    def kern(x: np.ndarray) -> np.ndarray:
        if p.kernel == "lorentzian":
            gam = np.pi * nu0 / p.Qres
            win = np.abs(x - nu0) < 4 * gam / (2 * np.pi)
            return np.where(win, gam / (gam + 2j * np.pi * (x - nu0)), 0.0)
        return (np.abs(x - nu0) < p.dband).astype(complex)

    K2 = 0.0
    if p.kappa2 > 0:
        K2 = p.kappa2 / np.sqrt(np.sum(np.abs(kern(fg)) ** 2 * vq))

    def S_0(x: np.ndarray) -> np.ndarray:
        return (
            S_g(x)
            + p.a1**2 / 2 * (S_g(x - nu0) + S_g(x + nu0))
            + p.a2**2 / 2 * (S_g(x - 2 * nu0) + S_g(x + 2 * nu0))
            + K2**2 * np.abs(kern(x)) ** 2 * np.interp(x, fg, S_q2)
        )

    d_psd = _d_psd(p)
    norm_d = p.nd_rms**2 / (np.sum(d_psd(fg[1:])) * dfg)

    def S_dil(x: np.ndarray) -> np.ndarray:
        out = p.D**2 * norm_d * d_psd(np.maximum(x, dfg))
        return out + 2 * (1 + p.D) / p.rate if poisson else out

    fm = nu0 - f
    eta = S_dil(fm) / S_0(fm)
    # two-sided per-bin power at resolution DF: P_g = S_g DF / 2
    P1, P2 = S_g(f) * DF / 2, S_g(fm) * DF / 2
    Q = 4 * K2 / p.a1**2 * P1 * P2 / (P1 + P2)
    return f, eta, Q


BASE = Params()

# BASE with a broken power-law flow field: flat to 1 Hz, f^-2 above, level
# chosen so the rms of g is 0.30 (same as BASE's sig_g) on the default grid
BASE_PL = replace(BASE, g_psd="powerlaw", pl_norm=0.0457, pl_fb=1.0, pl_slope=2.0)

_CROSS = replace(
    BASE, nu0=6.0, widths=(0.15, 0.8), D=1.0, nd_rms=0.8, nd_w=1.0, rate=2500.0, a2=0.0
)
_HYP = replace(BASE, nu0=6.0, widths=(0.25, 1.5), a1=0.03, a2=0.0, D=0.10, rate=12000.0)

RUNS: dict[str, Params] = {
    "A_web": BASE,
    "B_cross": replace(_CROSS, a1=0.30),
    "B0_none": replace(_CROSS, a1=0.06),
    "C_hypres": replace(_HYP, kappa2=0.30, kernel="lorentzian"),
    "C0_ctl": replace(_HYP, kappa2=0.0),
    "D_hypflat": replace(_HYP, kappa2=0.30, kernel="flat", argk=np.pi / 2),
}


def metrics(
    name: str, p: Params, shot_noise: bool = False
) -> tuple[dict[str, Any], PlotData]:
    """Build one run and measure its bicoherence on the predicted loci.

    The bicoherence is computed up to ``fmax = 1.3 nu0``, or ``2.3 nu0``
    when ``nu0 <= 4`` so that the harmonic point is on the grid.

    Parameters
    ----------
    name : str
        Run name. ``"A_web"``, ``"C_hypres"`` and ``"D_hypflat"`` add extra
        checks (see Returns).
    p : Params
        Model parameters.
    shot_noise : bool, optional
        Subtract the Poisson shot-noise bispectrum (see `bicoherence`).
        Default False.

    Returns
    -------
    out : dict
        Mean ``b2`` (``*_b2``) and circular-mean biphase (``*_biph``) on:

        - ``arm`` : the cross arm ``(f, nu0)``, ``f`` in [0.15, 0.7] ``nu0``.
        - ``ridge`` : the hypotenuse ``f1 + f2 = nu0``, ``f1`` in
          [0.2, 0.8] ``nu0``.
        - ``mid`` : the middle of the ridge, ``f1`` in [0.35, 0.65] ``nu0``.
        - ``off`` : an off-locus diagonal control, ``f`` in [0.15, 0.35]
          ``nu0`` (``b2`` only).
        - ``harm`` : the harmonic point ``(nu0, nu0)``, only when ``a2 > 0``.

        Also ``name``, ``K`` (segments), ``floor`` (``1 / K``) and ``nu0``.
        For ``"A_web"``, ``arm_meas_over_pred`` is the measured arm ``b2``
        over the zero-parameter bilinear prediction. For ``"C_hypres"`` and
        ``"D_hypflat"``, ``transverse`` is the biphase and ``b2`` profile
        across the mid ridge out to +/-1.2 Hz, ``pred_rot`` the Lorentzian
        prediction ``arctan(2 pi df / gamma)``, and ``corr_product_law`` and
        ``corr_sum_law`` the correlations of ridge ``b2`` with the product
        and sum laws.
    pdata : tuple
        ``(f, b2, biphase, floor, nu0, psd)``, for the plotting functions.
        ``psd = (fp, power, poisson, components)`` is the segment-averaged
        power spectrum of the same counts in fractional rms normalization
        (``(rms/mean)**2 / Hz``), on the rfft grid without the DC bin, with
        ``poisson = 2 / rate`` its expected Poisson-noise level.
        ``components`` maps each term returned by ``build(components=True)``
        to its own power spectrum, normalized by the same total mean count
        rate. The terms are not all independent (the quadratic term shares
        ``g`` with the flow), so the components need not sum to ``power``.
    """
    cts, flow, comps = build(p, components=True)
    nu0 = p.nu0
    k0 = int(round(nu0 / DF))
    fmax = 1.3 * nu0 if nu0 > 4 else 2.3 * nu0
    f, b2, biph, psd_tot, ns = bicoherence(cts, DT, NSEG, fmax, shot_noise)
    nf = len(f)
    floor = 1.0 / ns
    out = dict(name=name, K=ns, floor=floor, nu0=nu0)

    # cross arm (f, nu0): column k2 = k0, f in [0.15, 0.7] nu0
    sel = np.arange(max(int(0.15 * k0), 2), int(0.7 * k0))
    out["arm_b2"] = float(np.mean(b2[sel, k0 - 1]))
    out["arm_biph"] = float(np.angle(np.mean(np.exp(1j * biph[sel, k0 - 1]))))
    # ridge k1+k2 = k0, 0.2-0.8 nu0
    r1 = np.arange(max(int(0.2 * k0), 1), int(0.8 * k0))
    r2 = k0 - r1
    out["ridge_b2"] = float(np.mean(b2[r1 - 1, r2 - 1]))
    out["ridge_biph"] = float(np.angle(np.mean(np.exp(1j * biph[r1 - 1, r2 - 1]))))
    # off-locus control region (k1+k2 well below nu0, off arms)
    o1 = np.arange(max(int(0.15 * k0), 1), int(0.35 * k0))
    out["off_b2"] = float(np.mean(b2[o1 - 1, o1 - 1]))
    # mid-ridge fill (0.35-0.65 nu0): the Q-channel's home
    m1 = np.arange(int(0.35 * k0), int(0.65 * k0))
    m2 = k0 - m1
    out["mid_b2"] = float(np.mean(b2[m1 - 1, m2 - 1]))
    out["mid_biph"] = float(np.angle(np.mean(np.exp(1j * biph[m1 - 1, m2 - 1]))))
    # harmonic point
    if p.a2 > 0:
        out["harm_biph"] = float(biph[k0 - 1, k0 - 1])
        out["harm_b2"] = float(b2[k0 - 1, k0 - 1])

    # A only: zero-parameter bilinear check on the cross arm
    if name == "A_web":
        segs = flow[: ns * NSEG].reshape(ns, NSEG)
        segs = segs - segs.mean(axis=1, keepdims=True)
        psd_flow = (np.abs(np.fft.rfft(segs, axis=1)) ** 2).mean(axis=0)
        tone_frac = 1 - (psd_tot[k0 - 2] + psd_tot[k0 + 1]) / (2 * psd_tot[k0])  # crude
        tau = max(tone_frac, 0.0)
        # psd arrays are rfft-indexed (0..), b2 k-index starts at 1
        pred = (
            tau
            * (p.a1 ** 2 / 2)
            * (psd_flow[sel] + psd_flow[sel + k0]) ** 2
            / (psd_tot[sel] * psd_tot[sel + k0])
        )
        out["arm_meas_over_pred"] = float(np.mean(b2[sel - 1, k0 - 1] / pred))

    # C/D: transverse biphase profile across the ridge
    if name in ("C_hypres", "D_hypflat"):
        prof = []
        MT = int(round(1.2 / DF))  # +/-1.2 Hz: the widest offset on the HXMT slide
        for m in range(-MT, MT + 1):
            kk = k0 + m
            q1 = np.arange(int(0.35 * k0), int(0.65 * k0))
            q2_ = kk - q1
            ok = (q2_ >= 1) & (q2_ <= nf)
            ph = np.angle(np.mean(np.exp(1j * biph[q1[ok] - 1, q2_[ok] - 1])))
            bb = float(np.mean(b2[q1[ok] - 1, q2_[ok] - 1]))
            prof.append((m, float(ph), bb))
        out["transverse"] = prof
        gam = np.pi * nu0 / p.Qres
        out["pred_rot"] = [
            (m, float(np.arctan2(2 * np.pi * m * DF, gam))) for m in range(-MT, MT + 1)
        ]
        # product-law check along ridge: b2 ~ rho1*rho2 => corr with P(f1)P(f2)/(Ptot Ptot)
        segs = flow[: ns * NSEG].reshape(ns, NSEG)
        segs = segs - segs.mean(axis=1, keepdims=True)
        psd_flow = (np.abs(np.fft.rfft(segs, axis=1)) ** 2).mean(axis=0)
        # Q-channel per-bin law: b2 ~ rho1 rho2 P1 P2; bilinear: [P1+P2]^2/(Ptot Ptot)
        lawQ = (psd_flow[r1] * psd_flow[r2]) ** 2 / (psd_tot[r1] * psd_tot[r2])
        lawM = (psd_flow[r1] + psd_flow[r2]) ** 2 / (psd_tot[r1] * psd_tot[r2])
        bb = b2[r1 - 1, r2 - 1]
        out["corr_product_law"] = float(np.corrcoef(bb, lawQ)[0, 1])
        out["corr_sum_law"] = float(np.corrcoef(bb, lawM)[0, 1])
    # rms-normalized power spectrum of the same segments (Poisson level 2/rate)
    mu = cts[: ns * NSEG].mean()
    fp = np.fft.rfftfreq(NSEG, DT)[1:]
    pw = 2 * DT * psd_tot[1:] / (NSEG * mu**2)
    pw_comp = {}
    for key, c in comps.items():
        segs = c[: ns * NSEG].reshape(ns, NSEG)
        segs = segs - segs.mean(axis=1, keepdims=True)
        pc = (np.abs(np.fft.rfft(segs, axis=1)) ** 2).mean(axis=0)
        pw_comp[key] = 2 * DT * pc[1:] / (NSEG * mu**2)
    return out, (f, b2, biph, floor, nu0, (fp, pw, 2 * DT / mu, pw_comp))


# ----------------------------------------------------------------------------
# Bicoherence-map plotting
# ----------------------------------------------------------------------------
# Physical label for each run, used in titles.
LABELS = {
    "A_web": "web  (trio, tone in band)",
    "B_cross": "cross  (arms, no ridge)",
    "B0_none": "null  (weak drive)",
    "C_hypres": "hypotenuse  (Lorentzian lock)",
    "C0_ctl": "control  (lock off)",
    "D_hypflat": "hypotenuse  (flat lock)",
}


# Colour for each PSD component in _draw_psd(decompose=True), fixed per term
COMP_COLORS = {
    "dilution": "#2a78d6",
    "flow": "#eb6834",
    "QPO": "#1baf7a",
    "harmonic": "#eda100",
    "quadratic": "#e87ba4",
}


def _draw_pair(
    ax_b2: Axes,
    ax_ph: Axes,
    f: np.ndarray,
    b2: np.ndarray,
    biph: np.ndarray,
    floor: float,
    nu0: float,
    title: str,
    b2_log: bool = False,
    b2_vmin: float | None = None,
    b2_vmax: float | None = None,
    b2_label: str | None = None,
    ph_symlog: bool = False,
    ph_linthresh: float = 0.1,
    ph_label: str | None = None,
    ph_thresh: float | None = None,
    ph_hatch: str | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Draw the b2 map and the biphase map for one run onto two axes.

    Parameters
    ----------
    ax_b2, ax_ph : matplotlib.axes.Axes
        Axes for the ``b2`` map and the biphase map.
    f : ndarray, shape (nf,)
        Bin frequencies (Hz).
    b2, biph : ndarray, shape (nf, nf)
        Squared bicoherence and biphase (rad), as from `bicoherence`.
    floor : float
        ``b2`` noise floor. Sets the minimum top of the ``b2`` colour scale
        (``5 floor``) and the default biphase mask threshold.
    nu0 : float
        Tone frequency (Hz), marked with dotted lines.
    title : str
        Title prefix for both panels.
    b2_log : bool, optional
        Use a logarithmic ``b2`` colour scale. Default False (linear).
    b2_vmin, b2_vmax : float, optional
        Limits of the ``b2`` colour scale. Defaults: ``vmin`` is 0 (linear)
        or ``floor`` (log); ``vmax`` is the larger of the 99.5th percentile
        of ``b2`` and ``5 floor``.
    b2_label : str, optional
        Label for the ``b2`` colorbar. Default: no label.
    ph_symlog : bool, optional
        Use a symmetric-log biphase colour scale (linear within
        ``±ph_linthresh``). Default False (linear).
    ph_linthresh : float, optional
        Half-width (rad) of the linear region of the symlog biphase scale.
        Default 0.1. Ignored unless `ph_symlog`.
    ph_label : str, optional
        Label for the biphase colorbar. Default: no label.
    ph_thresh : float, optional
        The biphase is masked where ``b2 < ph_thresh``. Default ``3 floor``.
    ph_hatch : str, optional
        Matplotlib hatch pattern (e.g. ``"////"``, ``"xx"``) drawn in the
        masked cells. Default None: masked cells are solid light grey.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing `ax_b2`.
    axes : tuple of matplotlib.axes.Axes
        ``(ax_b2, ax_ph)``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize, SymLogNorm

    ext = [f[0], f[-1], f[0], f[-1]]
    # squared bicoherence: transpose so x=f1, y=f2; clip colour scale off floor
    vmax = b2_vmax if b2_vmax is not None else max(
        float(np.percentile(b2, 99.5)), 5 * floor
    )
    if b2_log:
        vmin = b2_vmin if b2_vmin is not None else floor
        norm_b: Normalize = LogNorm(vmin=vmin, vmax=vmax)
    else:
        vmin = b2_vmin if b2_vmin is not None else 0.0
        norm_b = Normalize(vmin=vmin, vmax=vmax)
    im_b = ax_b2.imshow(
        b2.T,
        origin="lower",
        extent=ext,
        aspect="equal",
        cmap="inferno",
        norm=norm_b,
        interpolation="nearest",
    )
    ax_b2.set_title(title + r"   $b^2$", fontsize=9)
    ax_b2.set_xlabel(r"$f_1$ (Hz)")
    ax_b2.set_ylabel(r"$f_2$ (Hz)")
    ax_b2.axhline(nu0, color="c", lw=0.5, ls=":")
    ax_b2.axvline(nu0, color="c", lw=0.5, ls=":")
    cb_b = plt.colorbar(im_b, ax=ax_b2, fraction=0.046, pad=0.04)
    if b2_label is not None:
        cb_b.set_label(b2_label)

    # biphase: mask where b2 is at/near the noise floor (phase meaningless there)
    thresh = ph_thresh if ph_thresh is not None else 3 * floor
    ph = np.ma.masked_where(b2.T < thresh, biph.T)
    cmap = plt.get_cmap("twilight_shifted").copy()
    if ph_hatch is None:
        cmap.set_bad("0.85")
    else:
        # masked cells transparent over a hatched background patch
        cmap.set_bad((0, 0, 0, 0))
        ax_ph.add_patch(
            plt.Rectangle(
                (ext[0], ext[2]), ext[1] - ext[0], ext[3] - ext[2],
                facecolor="white", edgecolor="0.6", hatch=ph_hatch, lw=0, zorder=0,
            )
        )
    if ph_symlog:
        norm_p: Normalize = SymLogNorm(
            linthresh=ph_linthresh, vmin=-np.pi, vmax=np.pi, base=10
        )
    else:
        norm_p = Normalize(vmin=-np.pi, vmax=np.pi)
    im_p = ax_ph.imshow(
        ph,
        origin="lower",
        extent=ext,
        aspect="equal",
        cmap=cmap,
        norm=norm_p,
        interpolation="nearest",
        zorder=1,
    )
    ax_ph.set_title(title + r"   biphase", fontsize=9)
    ax_ph.set_xlabel(r"$f_1$ (Hz)")
    ax_ph.set_ylabel(r"$f_2$ (Hz)")
    ax_ph.axhline(nu0, color="k", lw=0.5, ls=":")
    ax_ph.axvline(nu0, color="k", lw=0.5, ls=":")
    cb = plt.colorbar(
        im_p, ax=ax_ph, fraction=0.046, pad=0.04, ticks=[-np.pi, 0, np.pi]
    )
    cb.ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
    if ph_label is not None:
        cb.set_label(ph_label)
    fig = ax_b2.get_figure(root=True)
    assert fig is not None
    return fig, (ax_b2, ax_ph)


def _draw_psd(
    ax: Axes, 
    psd: PSD, 
    nu0: float, 
    fmax: float, 
    title: str = "",
    scale: Literal['p','fp'] = 'p',
    subtract_poisson: bool = False,
    decompose: bool = False,
) -> Axes:
    """Draw the rms-normalized power spectrum for one run onto `ax`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    psd : tuple
        ``(fp, power, poisson)``, as in the `metrics` plot data.
    nu0 : float
        Tone frequency (Hz); ``nu0`` and ``2 nu0`` are marked with dotted lines.
    fmax : float
        Highest frequency on the bicoherence map axes (Hz). The plot runs to
        ``2 fmax``, the highest sum frequency ``f1 + f2`` on the map.
    title : str, optional
        Title prefix.
    scale : {"p", "fp"}, optional
        Plot ``P`` (``"p"``, default) or ``f P`` (``"fp"``).
    subtract_poisson : bool, optional
        Subtract the Poisson level from the power instead of drawing it as a
        dashed line. Bins at or below zero after subtraction are not drawn
        on the log axes. Default False.
    decompose : bool, optional
        Also draw the power spectrum of each model term (dilution, flow,
        QPO, harmonic, quadratic) as a coloured dashed line, with a legend.
        These are noise-free, so `subtract_poisson` does not change them.
        Default False.

    Returns
    -------
    matplotlib.axes.Axes
        `ax`.
    """
    fp, pw, pn, comps = psd
    sel = fp <= 2 * fmax
    if subtract_poisson:
        # log axes: mask bins the subtraction leaves at or below zero
        pw = np.ma.masked_less_equal(pw - pn, 0.0)
    if scale == 'p':
        ax.loglog(fp[sel], pw[sel], color="0.15", lw=1.0, drawstyle="steps-mid")
    else:
        ax.loglog(fp[sel], fp[sel]*pw[sel], color="0.15", lw=1.0, drawstyle="steps-mid")
    if decompose:
        for key, pc in comps.items():
            pc = np.ma.masked_less_equal(pc, 0.0)
            y = pc[sel] if scale == 'p' else fp[sel] * pc[sel]
            ax.loglog(
                fp[sel], y, color=COMP_COLORS[key], lw=1.0, ls="--",
                drawstyle="steps-mid", label=key,
            )
        ax.legend(fontsize=7, loc="best", framealpha=0.85, edgecolor="none")
    if not subtract_poisson:
        if scale == 'p':
            ax.axhline(pn, color="0.5", lw=1.0, ls="--")
        else:
            ax.plot(fp[sel],pn*fp[sel],lw=1.,ls='--',color="0.5")
        ax.annotate(
            "Poisson", (fp[0], pn if scale == 'f' else fp[0] * pn),
            xytext=(3, 3), textcoords="offset points",
            ha="left", va="bottom", fontsize=8, color="0.4",
        )
    for m in (1, 2):
        if m * nu0 <= 2 * fmax:
            ax.axvline(m * nu0, color="c", lw=0.6, ls=":")
    ax.set_xlim(fp[0], 2 * fmax)
    ax.set_title(title + r"   power spectrum", fontsize=9)
    ax.set_xlabel(r"$f$ (Hz)")
    if scale=='p':
        ax.set_ylabel(r"$P$ ((rms/mean)$^2$ Hz$^{-1}$)")
    else:
        ax.set_ylabel(r"$f \times P$ ((rms/mean)$^2$)")
    ax.grid(True, which="major", color="0.9", lw=0.5)
    return ax


def plot_run(
    name: str, pdata: PlotData, path: str | None = None
) -> tuple[Figure, tuple[Axes, Axes, Axes]] | None:
    """Plot a three-panel (b2 | biphase | power spectrum) figure for one run.

    Parameters
    ----------
    name : str
        Run name, used to look up the title in `LABELS`.
    pdata : tuple
        ``(f, b2, biphase, floor, nu0, psd)``, the second value returned
        by `metrics`.
    path : str, optional
        Output image path. If given, the figure is saved there and closed.
        If omitted, the figure is left open and returned.

    Returns
    -------
    tuple of (Figure, (Axes, Axes, Axes)) or None
        ``(fig, (b2 axes, biphase axes, PSD axes))`` if `path` is omitted;
        otherwise None.
    """
    import matplotlib.pyplot as plt

    f, b2, biph, floor, nu0, psd = pdata
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(16, 4.6))
    _draw_pair(a0, a1, f, b2, biph, floor, nu0, LABELS.get(name, name))
    _draw_psd(a2, psd, nu0, f[-1], LABELS.get(name, name))
    fig.tight_layout()
    if path is None:
        return fig, (a0, a1, a2)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return None


def plot_combined(
    items: list[tuple[str, PlotData]],
    path: str | None = None,
    pair_kw: dict[str, Any] | Sequence[dict[str, Any]] | None = None,
    psd_kw: dict[str, Any] | Sequence[dict[str, Any]] | None = None,
) -> tuple[Figure, np.ndarray] | None:
    """Plot one figure with a (b2 | biphase | power spectrum) row per run.

    Parameters
    ----------
    items : list of tuple
        ``(name, pdata)`` pairs, with `pdata` as returned by `metrics`.
    path : str, optional
        Output image path. If given, the figure is saved there and closed.
        If omitted, the figure is left open and returned.
    pair_kw, psd_kw : dict or sequence of dict, optional
        Extra keyword arguments for `_draw_pair` and `_draw_psd`. A single
        dict applies to every row; a sequence gives one dict per row. Keys
        override the defaults set here (e.g. ``title``).

    Returns
    -------
    tuple of (Figure, ndarray of Axes) or None
        ``(fig, axes)`` if `path` is omitted; otherwise None.
        `axes` has shape ``(len(items), 3)``: one row per run, with the
        ``b2``, biphase and power-spectrum axes as columns.
    """
    import matplotlib.pyplot as plt

    n = len(items)

    def per_row(kw: dict[str, Any] | Sequence[dict[str, Any]] | None) -> list:
        if kw is None:
            return [{}] * n
        if isinstance(kw, dict):
            return [kw] * n
        if len(kw) != n:
            raise ValueError(f"got {len(kw)} kwarg dicts for {n} rows")
        return list(kw)

    pair_rows, psd_rows = per_row(pair_kw), per_row(psd_kw)
    fig, axes = plt.subplots(n, 3, figsize=(16, 4.3 * n), squeeze=False)
    for r, (name, pdata) in enumerate(items):
        f, b2, biph, floor, nu0, psd = pdata
        title = LABELS.get(name, name)
        _draw_pair(
            axes[r, 0], axes[r, 1], f, b2, biph, floor, nu0,
            **{"title": title, **pair_rows[r]},
        )
        _draw_psd(axes[r, 2], psd, nu0, f[-1], **{"title": title, **psd_rows[r]})
    fig.tight_layout()
    if path is None:
        return fig, axes
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return None


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    noplot = "--noplot" in sys.argv[1:]
    names = args or list(RUNS)
    res = []
    plots = []
    for n in names:
        r, pdata = metrics(n, RUNS[n])
        res.append(r)
        plots.append((n, pdata))
        print(json.dumps(r, indent=1))
        if not noplot:
            plot_run(n, pdata, os.path.join(OUT, f"bic_{n}.png"))
    with open(os.path.join(OUT, "sim_lightcurve_results.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    if not noplot:
        plot_combined(plots, os.path.join(OUT, "bic_all.png"))
        print("wrote bic_<run>.png for each run and bic_all.png")
