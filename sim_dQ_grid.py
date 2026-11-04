"""
Grid check of the two-parameter hyp/cross rule:

    b2_hyp / b2_cross  ~=  W (1 + Q)^2 / (1 + d)          (arg kappa2 = 0)

  Q = |B^Q / B^M| on the ridge (QPO amplitude vs quadratic coupling),
  d = dilution power / everything else, at the ridge partner f2 = nu0 - f1,
  W = undiluted bilinear ratio, P0(f3)/P0(f2) [(S1+S2)/(S1+S3)]^2.

Dials: a1 (moves Q, Q ~ a1^-2) and the rms of dilution confined below nu0
(moves d). Every predicted quantity comes from the per-term power spectra of
the same simulated light curve, so there are
no free parameters. Each run is also classified web / cross / hyp / none from
the measured arm and ridge against the off-locus floor.

Writes fig_dQ_grid.png and sim_dQ_grid.json here. With --dense, runs a
16 x 19 grid instead and writes sim_dQ_grid_dense.json only (paper_figs.ipynb
draws its plane with plot_plane); --extra-dense halves the dense log spacing in
both dials (31 x 36 grid, sim_dQ_grid_extra_dense.json). --plot replots from the
saved grid; --workers N sets the number of processes (default: all CPUs).
"""

import json, os
import numpy as np
import sim_lightcurve as sm

OUT = os.path.dirname(os.path.abspath(__file__))
KSEG = 3000  # floor 1/K = 3.3e-4; set on sm only when this script runs the grid

# Hypotenuse state, with dilution flat below nu0 (zero above)
BASE = sm.Params(nu0=5.0, a1=0.2, a2=0.0, g_psd="powerlaw", pl_fb=1.5, pl_slope=5.0,
                 sig_g=0.1, kappa2=0.1, kernel="flat", dband=0.05, argk=0.0,
                 D=1.0, nd_fc=5.0, nd_rms=0.0)
A1 = (0.0125, 0.025, 0.05, 0.1, 0.2, 0.4)
NDRMS = (0.0, 0.03, 0.06, 0.12, 0.2, 0.35)
A1_DENSE = tuple(np.round(np.geomspace(0.008, 0.5, 16), 5))
_ND_STEP = (0.4 / 0.01) ** (1 / 14)  # spacing of the main dilution columns
NDRMS_DENSE = ((0.0,) + tuple(np.round(0.01 * _ND_STEP ** -np.arange(3, 0, -1), 5))
               + tuple(np.round(np.geomspace(0.01, 0.4, 15), 5)))
# dense grid with twice the resolution in both dials (contains every dense point)
A1_XDENSE = tuple(np.round(np.geomspace(0.008, 0.5, 31), 5))
NDRMS_XDENSE = ((0.0,) + tuple(np.round(0.01 * _ND_STEP ** -(np.arange(6, 0, -1) / 2), 5))
                + tuple(np.round(np.geomspace(0.01, 0.4, 29), 5)))
GRIDS = {  # name: (a1 values, nd_rms values, json file)
    "coarse": (A1, NDRMS, "sim_dQ_grid.json"),
    "dense": (A1_DENSE, NDRMS_DENSE, "sim_dQ_grid_dense.json"),
    "extra_dense": (A1_XDENSE, NDRMS_XDENSE, "sim_dQ_grid_extra_dense.json"),
    # for a WEB-based `base` (D = 0.3): nd_rms / 0.3 keeps the dilution rms D * nd_rms
    # of "extra_dense"
    "extra_dense_web": (A1_XDENSE, tuple(np.round(np.array(NDRMS_XDENSE) / 0.3, 5)),
                        "sim_dQ_grid_extra_dense_web.json"),
}
NSIG = 3.0  # detection threshold, in standard errors of a locus mean
DOM = 3.0   # one locus "dominates" when it is DOM times the other


def run(p):
    r, (f, b2, biph, floor, nu0, psd) = sm.metrics("run", p, shot_noise=True)
    fp, P, pn, comps = psd
    S = comps["flow"]
    dil = comps["dilution"]
    Pk = lambda k: P[k - 1]                       # fp has no DC bin
    Sk = lambda k: S[abs(k) - 1] if k else 0.0
    Dk = lambda k: dil[k - 1]
    k0 = int(round(p.nu0 / sm.DF))
    tau = comps["QPO"][k0 - 1] / Pk(k0)
    lam = comps["quadratic"][k0 - 1] / Pk(k0)
    J = np.arange(-len(S) + k0 + 1, len(S) - 1)
    Pgg = 2 * sum(Sk(j) * Sk(k0 - j) for j in J)
    a = p.a1**2 / 2

    # f1 < nu0/2, away from f1 -> 0: arm (k, k0, k+k0) vs ridge (k, k0-k, k0)
    q = np.arange(max(int(0.15 * k0), 2), int(0.45 * k0))
    Q, d, W, pred = [], [], [], []
    for k in q:
        k2, k3 = k0 - k, k + k0
        S1, S2, S3 = Sk(k), Sk(k2), Sk(k3)
        Qk = np.sqrt(4 * lam * S1**2 * S2**2 / (Pgg * tau * a * (S1 + S2) ** 2))
        dk = Dk(k2) / (Pk(k2) - Dk(k2))
        d3 = Dk(k3) / (Pk(k3) - Dk(k3))           # ~0: dilution confined below nu0
        Wk = (Pk(k3) - Dk(k3)) / (Pk(k2) - Dk(k2)) * ((S1 + S2) / (S1 + S3)) ** 2
        Q.append(Qk); d.append(dk); W.append(Wk)
        pred.append(Wk * (1 + Qk) ** 2 * (1 + d3) / (1 + dk))

    # measured: means over the same bins, floor from the off-locus control
    # floor: off-locus patch f1 < f2 in [0.1, 0.4] nu0 (f1 + f2 <= 0.8 nu0, ~1000 bins)
    o = np.arange(int(0.1 * k0), int(0.4 * k0))
    iu = np.triu_indices(len(o), 1)
    fl = float(b2[o[iu[0]] - 1, o[iu[1]] - 1].mean())
    arm = b2[q - 1, k0 - 1].mean() - fl
    ridge = b2[q - 1, k0 - q - 1].mean() - fl
    se = fl / np.sqrt(len(q))                    # b2 under the null ~ exponential, sd = mean
    det_arm, det_ridge = bool(arm > NSIG * se), bool(ridge > NSIG * se)
    # ratio, or a bound on it when one locus is below the detection threshold
    if det_arm and det_ridge:
        meas, bound = ridge / arm, "none"
    elif det_ridge:
        meas, bound = ridge / (NSIG * se), "lower"
    elif det_arm:
        meas, bound = NSIG * se / arm, "upper"
    else:
        meas, bound = float("nan"), "none"
    return dict(
        a1=p.a1, nd_rms=p.nd_rms, tau=float(tau), Lam=float(lam),
        Q=float(np.median(Q)), d=float(np.median(d)), W=float(np.median(W)),
        pred=float(np.mean(pred)), arm=float(arm), ridge=float(ridge),
        meas=float(meas), bound=bound, det_arm=det_arm, det_ridge=det_ridge,
        se=float(se), floor=float(fl), pattern=classify(meas, det_arm, det_ridge),
    )


def classify(ratio, det_arm, det_ridge):
    """Pattern by which locus dominates: ridge/arm > DOM is a hypotenuse,
    < 1/DOM a cross; if only one locus is detected, the pattern is that locus."""
    if not (det_arm and det_ridge):
        return "hyp" if det_ridge else "cross" if det_arm else "none"
    if ratio > DOM:
        return "hyp"
    return "cross" if ratio < 1 / DOM else "web"


PAT = {  # palette: identity follows the pattern
    "web": ("#2a78d6", "o", "web"),
    "cross": ("#eb6834", "s", "cross"),
    "hyp": ("#1baf7a", "D", "hypotenuse"),
    "none": ("0.6", "x", "nothing detected"),
}


def mark(ax, x, y, key, filled, ms=9):
    """One run's marker: filled if both loci were detected, open if one was not."""
    c, m, _ = PAT[key]
    if m == "x":
        ax.plot(x, y, m, color=c, ms=ms - 1, mew=1.5, ls="none")
    else:
        # always fill
        filled = True
        ax.plot(x, y, m, ms=ms, ls="none", mew=1.5 if ms >= 7 else 1.0,
                color=c if filled else "white", mec="white" if filled else c)


# marker area for size_by_b2: linear in log10 b2 between these limits (pt^2)
B2_LIM = (1e-4, 0.3)
S_LIM = (6.0, 140.0)


def b2_overall(r):
    """Strength of the map a run shows: the larger floor-subtracted locus mean."""
    return max(r["arm"], r["ridge"])


def b2_size(b2, scale=1.0):
    """Marker area for b2; scale multiplies the whole S_LIM range (denser grids)."""
    lo, hi = np.log10(B2_LIM)
    u = np.clip((np.log10(np.maximum(b2, B2_LIM[0])) - lo) / (hi - lo), 0, 1)
    return scale * (S_LIM[0] + u * (S_LIM[1] - S_LIM[0]))


def size_handles(levels=(1e-3, 1e-2, 1e-1), size_scale=1.0):
    """Legend entries for the size_by_b2 marker areas (grey circles)."""
    import matplotlib.pyplot as plt
    return [plt.scatter([], [], s=float(b2_size(v, size_scale)), color="0.55",
                        edgecolors="white", linewidths=0.8 * np.sqrt(size_scale), label=f"$10^{{{int(np.log10(v))}}}$")
            for v in levels]


def legend_handles(ms=8):
    import matplotlib.pyplot as plt
    return [plt.Line2D([], [], marker=m, color=c, ls="none", ms=ms,
                       mec="white" if m != "x" else c, mew=1.5, label=lab)
            for c, m, lab in PAT.values()]


def plot_plane(ax, res, ms=9, label_a1=None, region_labels=True, size_by_b2=False,
               size_scale=1.0):
    """The (d, Q) plane: shading is the pattern the rule predicts (ratio beyond
    DOM or 1/DOM, with W at its grid median), markers are the measured pattern
    of each run at its predicted (d, Q). Undiluted runs sit at the left edge.
    label_a1: the a1 values whose rows get a label at the right (default all).
    size_by_b2: marker area set by b2_overall (log-scaled, see b2_size) instead of ms;
    size_scale multiplies those areas (and edge widths by its square root)."""
    from matplotlib.colors import ListedColormap
    Wm = np.median([r["W"] for r in res])
    dmin = 10**-2.85
    dd, QQ = np.logspace(-2.95, 2.5, 300), np.logspace(-2.5, 1.5, 300)
    Dg, Qg = np.meshgrid(dd, QQ)
    Z = Wm * (1 + Qg) ** 2 / (1 + Dg)
    zone = np.digitize(Z, [1 / DOM, DOM])  # 0 cross, 1 web, 2 hyp
    cmap = ListedColormap([PAT["cross"][0], PAT["web"][0], PAT["hyp"][0]])
    ax.pcolormesh(Dg, Qg, zone, cmap=cmap, alpha=0.13, shading="auto", lw=0,
                  rasterized=True)
    ax.contour(Dg, Qg, Z, levels=[1 / DOM, DOM], colors="0.5", linewidths=0.8)
    for r in res:
        x = max(r["d"], dmin)
        if not size_by_b2:
            mark(ax, x, r["Q"], r["pattern"], r["bound"] == "none", ms)
            continue
        c, m, _ = PAT[r["pattern"]]
        sz = float(b2_size(b2_overall(r), size_scale))
        lw = np.sqrt(size_scale)
        if m == "x":
            ax.scatter(x, r["Q"], s=sz, marker=m, color=c, linewidths=1.0 * lw, zorder=3)
        else:
            ax.scatter(x, r["Q"], s=sz, marker=m, color=c, edgecolors="white",
                       linewidths=0.6 * lw, zorder=3)
    nmax = max(r["nd_rms"] for r in res)
    for r in res:
        if r["nd_rms"] == nmax and (label_a1 is None or r["a1"] in label_a1):
            ax.annotate(f"$a_1$ = {r['a1']:.2g}", (r["d"], r["Q"]), xytext=(9, 0),
                        textcoords="offset points", fontsize=7.5, color="0.35", va="center")
    if region_labels:
        for x, y, t in [(3e-2, 6, "hypotenuse"), (3e-2, 0.05, "web"), (30, 0.05, "cross")]:
            ax.text(x, y, t, fontsize=10, color="0.25", weight="bold", zorder=5,
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(10**-2.95, 10**2.5); ax.set_ylim(10**-2.5, 10**1.5)
    ax.set_xlabel(r"dilution contrast  $\mathcal{d}(\nu_0-f_1)$")
    ax.set_ylabel(r"coupling ratio  $\mathcal{Q} = |B^Q/B^M|_{\rm ridge}$")


def plot(res, path):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))

    # (a) measured vs predicted ratio; open markers + arrows are bounds
    lim = [1e-2, 3e2]
    ax1.axhspan(DOM, lim[1], color=PAT["hyp"][0], alpha=0.07, lw=0)
    ax1.axhspan(lim[0], 1 / DOM, color=PAT["cross"][0], alpha=0.07, lw=0)
    for r in res:
        if r["pattern"] == "none":
            continue
        mark(ax1, r["pred"], r["meas"], r["pattern"], r["bound"] == "none")
        if r["bound"] != "none":
            s = 1.8 if r["bound"] == "lower" else 1 / 1.8
            ax1.annotate("", (r["pred"], r["meas"] * s), (r["pred"], r["meas"]),
                         arrowprops=dict(arrowstyle="->", color="0.45", lw=1))
    ax1.plot(lim, lim, color="0.3", lw=1)
    ax1.set_xscale("log"); ax1.set_yscale("log"); ax1.set_xlim(lim); ax1.set_ylim(lim)
    ax1.text(1.3e-2, DOM * 1.15, "hypotenuse-dominated", fontsize=8, color="0.35")
    ax1.text(2.5e2, 1.3e-2, "cross-dominated", fontsize=8, color="0.35", ha="right")
    ax1.set_xlabel(r"predicted  $W\,(1+\mathcal{Q})^2/(1+\mathcal{d})$")
    ax1.set_ylabel(r"measured  $b^2_{\rm hyp}/b^2_{\rm cross}$  (floor-subtracted)")
    ax1.set_title("(a) two-parameter rule vs simulation  (open = one locus undetected)",
                  fontsize=10, loc="left")
    ax1.legend(handles=legend_handles(), fontsize=8.5, frameon=False, loc="upper left")

    # (b) the (d, Q) plane
    plot_plane(ax2, res)
    ax2.set_xlabel(r"dilution contrast  $\mathcal{d}(\nu_0-f_1)$   (undiluted runs at left edge)")
    ax2.set_title(r"(b) shading: predicted pattern;  markers: measured pattern",
                  fontsize=10, loc="left")

    for ax in (ax1, ax2):
        ax.grid(True, which="major", color="0.92", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _init_worker():
    sm.K, sm.NTOT = KSEG, sm.NSEG * KSEG


def run_grid(grid="dense", path=None, workers=None, base=None):
    """Run one of GRIDS ("coarse", "dense", "extra_dense", "extra_dense_web") at
    K = KSEG and save it as JSON (default: that grid's file here). Runs are spread
    over `workers` processes (default: all CPUs; 1 runs serially here); each run
    has a fixed seed, so the result does not depend on the worker count. sm.K and
    sm.NTOT are restored afterwards, so a caller's settings survive. `base` (default
    BASE) supplies every parameter but the two dials. Returns the list of per-run
    results, in grid order."""
    from concurrent.futures import ProcessPoolExecutor
    a1s, nds, fname = GRIDS[grid]
    if path is None:
        path = os.path.join(OUT, fname)
    base = BASE if base is None else base
    ps = [sm.replace(base, a1=float(a), nd_rms=float(n)) for a in a1s for n in nds]
    workers = workers or os.cpu_count()
    if workers == 1:
        saved = sm.K, sm.NTOT
        _init_worker()
        try:
            res = [run(p) for p in ps]
        finally:
            sm.K, sm.NTOT = saved
    else:
        # one BLAS/FFT thread per process; spawned workers inherit this
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS"):
            os.environ.setdefault(v, "1")
        res = []
        with ProcessPoolExecutor(workers, initializer=_init_worker) as ex:
            for i, r in enumerate(ex.map(run, ps, chunksize=4), 1):
                res.append(r)
                if i % len(nds) == 0:
                    print(f"a1 = {r['a1']} done ({i}/{len(ps)})", flush=True)
    with open(path, "w") as fh:
        json.dump(dict(base=repr(base), K=KSEG, runs=res), fh, indent=1)
    return res

if __name__ == "__main__":
    import sys
    import matplotlib
    matplotlib.use("Agg")
    grid = ("extra_dense" if "--extra-dense" in sys.argv
            else "dense" if "--dense" in sys.argv else "coarse")
    jpath = os.path.join(OUT, GRIDS[grid][2])
    if "--plot" in sys.argv:  # replot from the saved grid
        with open(jpath) as fh:
            res = json.load(fh)["runs"]
    else:
        nw = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else None
        res = run_grid(grid, jpath, nw)
    if grid == "coarse":
        plot(res, os.path.join(OUT, "fig_dQ_grid.png"))
    for r in res:
        print(f"a1={r['a1']:<7} nd={r['nd_rms']:<5} Q={r['Q']:<7.3g} d={r['d']:<7.3g} "
              f"pred={r['pred']:<7.3g} meas={r['meas']:<7.3g} ({r['bound']}) {r['pattern']}")
