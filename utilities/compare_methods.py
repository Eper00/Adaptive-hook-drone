"""Compare the three director methods and show where the safety filter saves the RL.

Methods (all on the same scenarios, seed by seed):

    director         RL director policy
    director_mpc     director MPC (ARX model)
    director_safety  RL director + predictive safety filter (ResNet model)

(``--methods`` also accepts director_resnet_mpc.) The episodes run in
parallel processes; the raw episodes are cached in ``episodes.pkl`` so the
plots can be redone with ``--replot``.

The comparison is built around the question "when the RL director fails,
does the safety filter save the drone?". The RL and RL + filter runs of a
seed are identical until the filter's first intervention, so each seed falls
into one of

    kept         RL succeeds, RL + filter succeeds
    rescued      RL fails,    RL + filter succeeds
    not rescued  RL fails,    RL + filter fails
    broken       RL succeeds, RL + filter fails

Figures (results/analysis/methods/):
    overview.png         outcomes, success vs payload mass, time to goal,
                         max tilt, command roughness, computation per step
    rescue_matrix.png    RL outcome -> RL + filter outcome (and -> MPC)
    paired_outcomes.png  every scenario, every method
    rescue_gallery.png   every RL failure: tilt and height of RL vs RL + filter,
                         with the filter's interventions (why) and the crash
    rescue_timing.png    how early the filter acts before the RL failure,
                         what it does, and what it costs on RL successes
    scenario_map.png     where the scenarios are and what happened
    summary.json         all numbers

Usage:
    python -m utilities.compare_methods                       # 100 seeds, 25 s
    python -m utilities.compare_methods --episodes 40 --workers 10
    python -m utilities.compare_methods --replot              # plots from episodes.pkl

An interrupted run continues where it stopped (finished chunks are kept in
chunks/); ``--fresh`` starts over, e.g. after retraining a policy or model or
changing the safety filter.

A safety-filter worker needs 2-3.5 GB, so the number of workers is capped by
the available memory (``--worker_mem`` GB per worker, default 3.5).
"""

import argparse
import json
import multiprocessing
import os
import pickle
import time
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

METHODS = ["director", "director_mpc", "director_safety"]
NAMES = {"director": "RL director", "director_mpc": "Director MPC",
         "director_resnet_mpc": "Director ResNet-MPC", "director_safety": "RL + safety filter"}
COLORS = {"director": "#3b6fb6", "director_mpc": "#8e5bb5",
          "director_resnet_mpc": "#9a9a2e", "director_safety": "#1ba39c"}
PAIR = ["kept", "rescued", "not rescued", "broken"]
PAIR_COLORS = {"kept": "#b9dcb3", "rescued": "#2e9e4a", "not rescued": "#d1453b",
               "broken": "#e8a33d"}


################################################################################
# RUNNING
################################################################################

def available_memory_gb():
    """MemAvailable of /proc/meminfo [GB] (None where unknown)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return None


def memory_limited_workers(workers, worker_mem, reserve=3.0):
    """At most as many workers as fit into the available memory (keeping
    ``reserve`` GB for the main process and the desktop). A safety-filter
    worker needs 2-3.5 GB (PPO policies, ResNet model, CasADi functions);
    ten of them on a 32 GB machine ran the system out of memory."""
    avail = available_memory_gb()
    if avail is None:
        return workers
    fit = max(1, int((avail - reserve) / worker_mem))
    if fit < workers:
        print(f"[memory] {avail:.1f} GB available, {worker_mem:g} GB per worker:"
              f" {workers} -> {fit} workers (--worker_mem to change)")
    return min(workers, fit)


def _run_chunk(kind, seeds, max_time, model_path):
    """Worker: fly ``seeds`` with one method (own env / controller)."""
    import matplotlib
    matplotlib.use("Agg")
    from multi_drone_mujoco.envs.predictive_safety_filter import hook_obstacle_contact
    from utilities.analyse import Overlay, make_controller, run_director_episode

    env, controller = make_controller(kind, Namespace(render=False, model_path=model_path))
    step = controller.step
    rec = {}

    def timed_step():
        t0 = time.perf_counter()
        r = step()
        rec["dt"].append(time.perf_counter() - t0)
        # hook touching the payload assembly (any part / connectors + plate only)
        rec["c_any"].append(hook_obstacle_contact(env, False))
        rec["c_rest"].append(hook_obstacle_contact(env, True))
        return r

    controller.step = timed_step
    out = []
    for seed in seeds:
        rec.update(dt=[], c_any=[], c_rest=[])
        ep = run_director_episode(controller, env, seed, max_time, False, Overlay(env))
        att = ep["attached"].astype(bool)
        ep["step_time"] = np.asarray(rec["dt"])
        ep["contact"] = np.where(att, rec["c_rest"], rec["c_any"])
        ep["qpos"] = ep["qpos"].astype(np.float32)
        ep["kind"] = kind
        out.append(ep)
    env.close()
    return kind, out


def run_all(methods, seeds, max_time, workers, model_path, chunk, chunk_dir):
    """Fly all (method, seed) pairs. Every finished chunk is saved in
    ``chunk_dir`` and reused on a restart (``--fresh`` deletes them)."""
    jobs = [(k, seeds[i:i + chunk]) for k in methods for i in range(0, len(seeds), chunk)]
    # slow methods first, so the pool stays busy
    jobs.sort(key=lambda j: j[0] not in ("director_safety", "director_resnet_mpc"))
    os.makedirs(chunk_dir, exist_ok=True)

    def chunk_file(kind, s):
        return os.path.join(chunk_dir, f"{kind}_{s[0]}_{s[-1]}_{max_time:g}s.pkl")

    results = {k: [] for k in methods}
    todo = []
    for k, s in jobs:
        if os.path.exists(chunk_file(k, s)):
            with open(chunk_file(k, s), "rb") as f:
                results[k] += pickle.load(f)
        else:
            todo.append((k, s))
    if len(todo) < len(jobs):
        print(f"reusing {len(jobs) - len(todo)} finished chunks from {chunk_dir}")
    t0 = time.time()
    # one BLAS / OpenMP thread per worker; set here, because the spawned
    # workers import numpy (this module) before they run a chunk
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=ctx) as ex:
        futs = {ex.submit(_run_chunk, k, s, max_time, model_path): (k, s) for k, s in todo}
        jobs = todo
        for n, f in enumerate(as_completed(futs), 1):
            kind, eps = f.result()
            with open(chunk_file(*futs[f]), "wb") as fh:
                pickle.dump(eps, fh)
            results[kind] += eps
            out = ", ".join(f"{e['seed']}:{e['outcome']}" for e in eps)
            print(f"[{n:3d}/{len(jobs)} {time.time() - t0:6.0f}s] {NAMES[kind]:20s} {out}",
                  flush=True)
    for k in results:
        results[k].sort(key=lambda e: e["seed"])
    return results


################################################################################
# ANALYSIS
################################################################################

def tilt_deg(ep):
    r, p = ep["rpy"][:, 0], ep["rpy"][:, 1]
    return np.degrees(np.arccos(np.clip(np.cos(r) * np.cos(p), -1, 1)))


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def pair_category(ep_rl, ep_f):
    ok_rl, ok_f = ep_rl["outcome"] == "success", ep_f["outcome"] == "success"
    return {(True, True): "kept", (False, True): "rescued",
            (False, False): "not rescued", (True, False): "broken"}[(ok_rl, ok_f)]


def first_intervention(ep):
    idx = np.nonzero(ep["intervened"].astype(bool))[0]
    return (float(ep["t"][idx[0]]), int(idx[0])) if idx.size else (None, None)


def summarize(results):
    from utilities.analyse import OUTCOMES, REASONS
    s = {"methods": {}}
    for k, eps in results.items():
        n = len(eps)
        counts = {o: sum(e["outcome"] == o for e in eps) for o in OUTCOMES}
        tg = [e["event_time"] for e in eps if e["outcome"] == "success"]
        lo, hi = wilson(counts["success"], n)
        s["methods"][k] = {
            "name": NAMES[k], "episodes": n, **counts,
            "success_rate": counts["success"] / max(n, 1), "success_ci95": [lo, hi],
            "time_to_goal_median": float(np.median(tg)) if tg else None,
            "max_tilt_median_deg": float(np.median([tilt_deg(e).max() for e in eps])),
            "max_swing_median_deg": float(np.median([np.degrees(np.abs(e["swing"]).max())
                                                     for e in eps])),
            "step_time_median_ms": float(1e3 * np.median(np.concatenate([e["step_time"]
                                                                          for e in eps]))),
            "filter_active_percent": float(100 * np.mean(np.concatenate(
                [e["intervened"] for e in eps]))),
            "interventions_by_reason_steps": {
                r: int(sum((e["reason"] == REASONS.index(r)).sum() for e in eps))
                for r in REASONS[1:]},
        }
    if "director" in results and "director_safety" in results:
        rows = []
        for a, b in zip(results["director"], results["director_safety"]):
            t1, _ = first_intervention(b)
            rows.append({"seed": int(a["seed"]), "rl": a["outcome"], "filter": b["outcome"],
                         "category": pair_category(a, b), "mass": float(a["mass"]),
                         "rl_event_time": float(a["event_time"]),
                         "first_intervention": t1,
                         "filter_active_percent": float(100 * b["intervened"].mean())})
        s["pairs"] = rows
        s["pair_counts"] = {c: sum(r["category"] == c for r in rows) for c in PAIR}
    return s


################################################################################
# PLOTS
################################################################################

def _style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "font.size": 9,
                         "axes.titlesize": 10, "axes.titleweight": "bold",
                         "legend.frameon": False})


def plot_overview(results, out):
    import matplotlib.pyplot as plt
    from utilities.analyse import OUTCOME_COLORS, OUTCOMES

    from multi_drone_mujoco.envs.predictive_safety_filter import SafetyConstraints
    lim = SafetyConstraints()
    kinds = list(results)
    names = [NAMES[k] for k in kinds]
    fig, axs = plt.subplots(2, 3, figsize=(17, 9.5))

    ax = axs[0, 0]                                   # outcomes (stacked)
    for j, k in enumerate(kinds):
        eps, bottom = results[k], 0
        n = len(eps)
        for o in OUTCOMES:
            c = sum(e["outcome"] == o for e in eps)
            ax.bar(j, 100 * c / n, bottom=bottom, color=OUTCOME_COLORS[o], width=0.6,
                   edgecolor="white", label=o if j == 0 else None)
            if c:
                ax.text(j, bottom + 50 * c / n, str(c), ha="center", va="center", fontsize=8,
                        color="white" if o != "unfinished" else "black")
            bottom += 100 * c / n
        lo, hi = wilson(sum(e["outcome"] == "success" for e in eps), n)
        ax.errorbar(j + 0.36, 100 * (lo + hi) / 2, yerr=50 * (hi - lo), color="k", capsize=3,
                    lw=1)
    ax.set_xticks(range(len(kinds)), names)
    ax.set_ylabel("episodes [%]")
    ax.set_title(f"Outcomes ({len(results[kinds[0]])} scenarios, bar: 95% CI of success)")
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.07))
    ax.set_ylim(0, 100)

    ax = axs[0, 1]                                   # success vs payload mass
    masses = np.array([e["mass"] for e in results[kinds[0]]])
    edges = np.quantile(masses, np.linspace(0, 1, 6))
    edges[-1] += 1e-9
    centers = 0.5 * (edges[:-1] + edges[1:])
    for k in kinds:
        rate = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = [e for e in results[k] if lo <= e["mass"] < hi]
            rate.append(100 * np.mean([e["outcome"] == "success" for e in sel]) if sel else np.nan)
        ax.plot(centers, rate, "o-", color=COLORS[k], label=NAMES[k])
    ax.set_xlabel("payload mass [kg] (quintile bins)")
    ax.set_ylabel("success [%]")
    ax.set_title("Success vs payload mass")
    ax.legend(fontsize=8)

    def box(ax, data, title, ylabel, hline=None, log=False):
        data = [d if len(d) else [np.nan] for d in data]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=True)
        for patch, k in zip(bp["boxes"], kinds):
            patch.set_facecolor(COLORS[k])
            patch.set_alpha(0.55)
        for m in bp["medians"]:
            m.set_color("k")
        ax.set_xticks(range(1, len(kinds) + 1), names)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if hline is not None:
            ax.axhline(hline, color="#d1453b", ls=":", lw=1.2)
        if log:
            from matplotlib.ticker import ScalarFormatter
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_formatter(ScalarFormatter())

    box(axs[0, 2], [[e["event_time"] for e in results[k] if e["outcome"] == "success"]
                    for k in kinds], "Time to the goal with the payload (successes)", "[s]")
    box(axs[1, 0], [[tilt_deg(e).max() for e in results[k]] for k in kinds],
        "Max tilt per episode (dotted: filter limit)", "[deg]", np.degrees(lim.max_tilt))
    box(axs[1, 1], [[np.mean(np.linalg.norm(np.diff(e["cmd"], axis=0), axis=1)) * 48
                     for e in results[k]] for k in kinds],
        "Command roughness  mean |d cmd/dt|", "[m/s²]")
    box(axs[1, 2], [[1e3 * np.median(e["step_time"]) for e in results[k]] for k in kinds],
        "Computation per control step (episode median)", "[ms]", 1e3 / 48, log=True)
    axs[1, 2].text(0.02, 1e3 / 48 * 1.1, "real time (20.8 ms)", color="#d1453b", fontsize=8,
                   transform=axs[1, 2].get_yaxis_transform())
    fig.suptitle("Director methods on identical scenarios", fontsize=13, fontweight="bold")
    fig.tight_layout()
    f = os.path.join(out, "overview.png")
    fig.savefig(f, dpi=140)
    plt.close(fig)
    return [f]


def plot_rescue_matrix(results, out):
    import matplotlib.pyplot as plt
    from utilities.analyse import OUTCOMES

    others = [k for k in results if k != "director"]
    fig, axs = plt.subplots(1, len(others), figsize=(5.2 * len(others), 4.6))
    axs = np.atleast_1d(axs)
    for ax, k in zip(axs, others):
        M = np.zeros((len(OUTCOMES), len(OUTCOMES)), int)
        for a, b in zip(results["director"], results[k]):
            M[OUTCOMES.index(a["outcome"]), OUTCOMES.index(b["outcome"])] += 1
        # colour: green = improvement to success, red = new failure
        C = np.full(M.shape + (3,), 1.0)
        for i in range(len(OUTCOMES)):
            for j in range(len(OUTCOMES)):
                if M[i, j] == 0:
                    continue
                if i == j:
                    base = np.array([0.85, 0.85, 0.85])
                elif OUTCOMES[j] == "success":
                    base = np.array([0.18, 0.62, 0.29])
                elif OUTCOMES[i] == "success":
                    base = np.array([0.82, 0.27, 0.23])
                else:
                    base = np.array([0.91, 0.64, 0.24])
                C[i, j] = 1 - (1 - base) * min(1.0, 0.35 + M[i, j] / M.max())
        ax.imshow(C)
        for i in range(len(OUTCOMES)):
            for j in range(len(OUTCOMES)):
                if M[i, j]:
                    ax.text(j, i, M[i, j], ha="center", va="center", fontsize=12,
                            fontweight="bold")
        ax.set_xticks(range(len(OUTCOMES)), OUTCOMES)
        ax.set_yticks(range(len(OUTCOMES)), OUTCOMES)
        ax.set_xlabel(NAMES[k])
        ax.set_ylabel("RL director")
        ax.grid(False)
        fails = M[1:].sum()
        saved = M[1:, 0].sum()
        broken = M[0, 1:].sum()
        ax.set_title(f"RL failures turned into success: {saved}/{fails}\n"
                     f"RL successes lost: {broken}/{M[0].sum()}")
    fig.suptitle("Same scenario, outcome of the RL director → other method",
                 fontweight="bold")
    fig.tight_layout()
    f = os.path.join(out, "rescue_matrix.png")
    fig.savefig(f, dpi=140)
    plt.close(fig)
    return [f]


def plot_paired(results, out):
    import matplotlib
    import matplotlib.pyplot as plt
    from utilities.analyse import OUTCOME_COLORS, OUTCOMES, _outcome_patches

    kinds = list(results)
    order = np.argsort([OUTCOMES.index(e["outcome"]) for e in results["director"]],
                       kind="stable")
    code = {o: i for i, o in enumerate(OUTCOMES)}
    M = np.array([[code[results[k][i]["outcome"]] for i in order] for k in kinds])
    fig, ax = plt.subplots(figsize=(max(9, 0.13 * M.shape[1] + 3), 0.55 * len(kinds) + 1.9))
    cmap = matplotlib.colors.ListedColormap([OUTCOME_COLORS[o] for o in OUTCOMES])
    ax.imshow(M, aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(OUTCOMES) - 0.5,
              interpolation="nearest")
    ax.set_yticks(range(len(kinds)), [NAMES[k] for k in kinds])
    seeds = [results["director"][i]["seed"] for i in order]
    ax.set_xticks(range(len(seeds)), seeds, fontsize=5, rotation=90)
    ax.set_xlabel("scenario (seed), sorted by the RL director's outcome")
    ax.grid(False)
    if "director_safety" in kinds:
        r = kinds.index("director_safety")
        for x, i in enumerate(order):
            if pair_category(results["director"][i], results["director_safety"][i]) == "rescued":
                ax.plot(x, r, marker="*", color="white", ms=7, mec="k", mew=0.5)
    ax.legend(handles=_outcome_patches() + [matplotlib.lines.Line2D(
        [], [], marker="*", color="white", mec="k", ls="", ms=8, label="rescued by the filter")],
        ncol=5, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    f = os.path.join(out, "paired_outcomes.png")
    fig.savefig(f, dpi=140)
    plt.close(fig)
    return [f]


def _spans(mask, t):
    from utilities.analyse import _intervals
    return _intervals(mask, t)


def plot_gallery(results, out, max_cases=24):
    """Every RL failure: RL vs RL + filter (tilt, height), interventions shaded."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from utilities.analyse import REASON_COLORS, REASONS

    rl, fl = results["director"], results["director_safety"]
    cases = [(a, b) for a, b in zip(rl, fl) if a["outcome"] != "success"]
    cases += [(a, b) for a, b in zip(rl, fl)
              if a["outcome"] == "success" and b["outcome"] != "success"]
    cases.sort(key=lambda ab: (PAIR.index(pair_category(*ab)), ab[0]["seed"]))
    cases = cases[:max_cases]
    if not cases:
        return []
    ncol = 4
    nrow = int(np.ceil(len(cases) / ncol))
    fig = plt.figure(figsize=(4.6 * ncol, 3.9 * nrow))
    gs = fig.add_gridspec(2 * nrow, ncol, height_ratios=[1.6, 1] * nrow, hspace=0.12,
                          wspace=0.28)
    for n, (a, b) in enumerate(cases):
        r, c = divmod(n, ncol)
        ax1 = fig.add_subplot(gs[2 * r, c])
        ax2 = fig.add_subplot(gs[2 * r + 1, c], sharex=ax1)
        cat = pair_category(a, b)
        for ax in (ax1, ax2):
            for reason in REASON_COLORS:
                for t0, t1 in _spans(b["reason"] == REASONS.index(reason), b["t"]):
                    ax.axvspan(t0, t1 + 1 / 48, color=REASON_COLORS[reason], alpha=0.22, lw=0)
            for t0, t1 in _spans(a["contact"].astype(bool), a["t"]):
                ax.axvspan(t0, t1 + 1 / 48, ymin=0.94, ymax=1.0, color="k", lw=0)
        ax1.plot(a["t"], tilt_deg(a), color=COLORS["director"], lw=1.1, label="RL")
        ax1.plot(b["t"], tilt_deg(b), color=COLORS["director_safety"], lw=1.1,
                 label="RL + filter")
        ax1.axhline(15, color="#d1453b", ls=":", lw=0.9)
        ax2.plot(a["t"], a["p"][:, 2], color=COLORS["director"], lw=1.1)
        ax2.plot(b["t"], b["p"][:, 2], color=COLORS["director_safety"], lw=1.1)
        pz = a["payload_pos"][:, 2] - a["payload_start"][2]
        ax2.plot(a["t"], pz, color=COLORS["director"], lw=0.8, ls="--")
        ax2.plot(b["t"], b["payload_pos"][:, 2] - b["payload_start"][2],
                 color=COLORS["director_safety"], lw=0.8, ls="--")
        for ep, ax in ((a, ax1), (b, ax1)):
            if ep["outcome"] == "stability":
                ax.plot(ep["t"][-1], min(tilt_deg(ep)[-1], 95), "X", ms=9,
                        color=COLORS[ep["kind"]], mec="k", mew=0.6)
        ax1.set_ylim(0, 100)
        ax1.tick_params(labelbottom=False)
        ax1.set_title(f"seed {a['seed']}  m={a['mass']:.2f} kg\n"
                      f"RL: {a['outcome']} → filter: {b['outcome']}",
                      color=PAIR_COLORS[cat] if cat != "kept" else "k", fontsize=9)
        if c == 0:
            ax1.set_ylabel("tilt [deg]")
            ax2.set_ylabel("z [m]")
        if r == nrow - 1:
            ax2.set_xlabel("t [s]")
    handles = [plt.Line2D([], [], color=COLORS["director"], label="RL director"),
               plt.Line2D([], [], color=COLORS["director_safety"], label="RL + safety filter"),
               plt.Line2D([], [], color="gray", ls="--", label="payload lift (z - rest)"),
               plt.Line2D([], [], marker="X", color="gray", ls="", mec="k", label="flip / crash"),
               Patch(color="k", label="RL: hook touches payload assembly (top strip)")]
    handles += [Patch(color=REASON_COLORS[r], alpha=0.4, label=f"filter: {r}")
                for r in REASON_COLORS]
    fig.legend(handles=handles, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               fontsize=9)
    fig.suptitle("Every scenario the RL director fails (and any the filter breaks): "
                 "what the safety filter changed", fontsize=13, fontweight="bold", y=1.03)
    f = os.path.join(out, "rescue_gallery.png")
    fig.savefig(f, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return [f]


def plot_timing(results, out):
    import matplotlib.pyplot as plt
    from utilities.analyse import REASON_COLORS, REASONS

    rl, fl = results["director"], results["director_safety"]
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))

    ax = axs[0]                                  # lead time before the RL failure
    for cat, mk in (("rescued", "o"), ("not rescued", "s")):
        xs, ys = [], []
        for a, b in zip(rl, fl):
            if pair_category(a, b) != cat:
                continue
            t1, _ = first_intervention(b)
            xs.append(a["event_time"])
            ys.append(np.nan if t1 is None else a["event_time"] - t1)
        ax.scatter(xs, ys, marker=mk, s=40, color=PAIR_COLORS[cat], edgecolor="k", lw=0.5,
                   label=f"{cat} ({len(xs)})")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("time of the RL failure [s]")
    ax.set_ylabel("RL failure time - first intervention [s]")
    ax.set_title("How early the filter acts (> 0: before the RL fails)")
    ax.legend(fontsize=8)

    ax = axs[1]                                  # what the filter does, per group
    groups = {c: [b for a, b in zip(rl, fl) if pair_category(a, b) == c] for c in PAIR}
    groups = {c: g for c, g in groups.items() if g}
    x = np.arange(len(groups))
    bottom = np.zeros(len(groups))
    for reason in REASON_COLORS:
        v = np.array([100 * np.mean(np.concatenate([e["reason"] == REASONS.index(reason)
                                                    for e in g])) for g in groups.values()])
        ax.bar(x, v, bottom=bottom, color=REASON_COLORS[reason], label=reason, width=0.6)
        bottom += v
    ax.set_xticks(x, [f"{c}\n({len(g)})" for c, g in groups.items()])
    ax.set_ylabel("steps with intervention [%]")
    ax.set_title("Why the filter intervenes, by scenario group")
    ax.legend(fontsize=8)

    ax = axs[2]                                  # cost on RL successes
    kept = [(a, b) for a, b in zip(rl, fl) if pair_category(a, b) == "kept"]
    if kept:
        d = [b["event_time"] - a["event_time"] for a, b in kept]
        ax.hist(d, bins=20, color=PAIR_COLORS["rescued"], edgecolor="white")
        ax.axvline(np.median(d), color="k", ls="--", lw=1)
        ax.text(np.median(d), ax.get_ylim()[1] * 0.92, f"  median {np.median(d):+.2f} s",
                fontsize=8)
    ax.set_xlabel("time to goal: RL + filter - RL [s]")
    ax.set_ylabel("scenarios")
    ax.set_title("Cost of the filter where the RL already succeeds")
    fig.tight_layout()
    f = os.path.join(out, "rescue_timing.png")
    fig.savefig(f, dpi=140)
    plt.close(fig)
    return [f]


def plot_map(results, out):
    import matplotlib.pyplot as plt

    rl, fl = results["director"], results["director_safety"]
    fig, ax = plt.subplots(figsize=(7.5, 7))
    for a, b in zip(rl, fl):
        cat = pair_category(a, b)
        p, g = a["payload_start"][:2], a["goal"][:2]
        ax.annotate("", g, p, arrowprops=dict(arrowstyle="-", color=PAIR_COLORS[cat],
                                              lw=0.6 if cat == "kept" else 1.3, alpha=0.8))
        ax.scatter(*p, s=30 + 500 * a["mass"], color=PAIR_COLORS[cat], edgecolor="k",
                   lw=0.4, zorder=3)
        ax.scatter(*g, marker="x", s=18, color=PAIR_COLORS[cat], zorder=3)
        if cat != "kept":
            ax.annotate(str(a["seed"]), p, fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.scatter(0, 0, marker="^", s=90, color="k", zorder=4)
    for c in PAIR:
        ax.scatter([], [], color=PAIR_COLORS[c], edgecolor="k", lw=0.4, s=50,
                   label=f"{c} ({sum(pair_category(a, b) == c for a, b in zip(rl, fl))})")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Scenarios: payload (circle, size ~ mass) → goal (x); ▲ take-off\n"
                 "colour: RL director vs RL + safety filter")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    f = os.path.join(out, "scenario_map.png")
    fig.savefig(f, dpi=140)
    plt.close(fig)
    return [f]


def report(results, out):
    _style()
    s = summarize(results)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(s, f, indent=1, default=float)
    files = plot_overview(results, out)
    if "director" in results:
        files += plot_rescue_matrix(results, out)
        files += plot_paired(results, out)
    if "director" in results and "director_safety" in results:
        files += plot_gallery(results, out)
        files += plot_timing(results, out)
        files += plot_map(results, out)

    from utilities.analyse import OUTCOMES
    print(f"\n{'method':22s}" + "".join(f"{o:>11s}" for o in OUTCOMES)
          + f"{'t_goal':>9s}{'tilt':>7s}{'ms/step':>9s}{'filter%':>9s}")
    for k, m in s["methods"].items():
        print(f"{m['name']:22s}" + "".join(f"{m[o]:11d}" for o in OUTCOMES)
              + f"{(m['time_to_goal_median'] or np.nan):9.1f}{m['max_tilt_median_deg']:7.1f}"
              + f"{m['step_time_median_ms']:9.1f}{m['filter_active_percent']:9.2f}")
    if "pair_counts" in s:
        print("RL -> RL + filter:", s["pair_counts"])
        for r in s["pairs"]:
            if r["category"] != "kept":
                print(f"  seed {r['seed']:3d} m={r['mass']:.2f}  {r['rl']:10s} -> {r['filter']:10s}"
                      f" [{r['category']}] RL event {r['rl_event_time']:5.1f}s,"
                      f" first intervention {r['first_intervention']}")
    for f in files + [os.path.join(out, "summary.json")]:
        print("saved:", f)
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0, help="first seed")
    p.add_argument("--max_time", type=float, default=25.0)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                   help="parallel workers (capped by the available memory)")
    p.add_argument("--worker_mem", type=float, default=3.5,
                   help="memory budget per worker [GB] for the cap")
    p.add_argument("--chunk", type=int, default=5, help="seeds per worker job")
    p.add_argument("--model_path", default=None, help="RL director policy (default: final)")
    p.add_argument("--out", default="results/analysis/methods")
    p.add_argument("--replot", action="store_true", help="plots from the cached episodes")
    p.add_argument("--fresh", action="store_true", help="ignore finished chunks of a previous run")
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    os.makedirs(a.out, exist_ok=True)
    cache = os.path.join(a.out, "episodes.pkl")
    if a.replot:
        with open(cache, "rb") as f:
            results = pickle.load(f)
    else:
        methods = [m.strip() for m in a.methods.split(",")]
        seeds = list(range(a.seed, a.seed + a.episodes))
        a.workers = memory_limited_workers(a.workers, a.worker_mem)
        print(f"{len(methods)} methods x {len(seeds)} scenarios, {a.max_time:.0f} s,"
              f" {a.workers} workers")
        chunk_dir = os.path.join(a.out, "chunks")
        if a.fresh and os.path.isdir(chunk_dir):
            import shutil
            shutil.rmtree(chunk_dir)
        results = run_all(methods, seeds, a.max_time, a.workers, a.model_path, a.chunk,
                          chunk_dir)
        with open(cache, "wb") as f:
            pickle.dump(results, f)
    report(results, a.out)


if __name__ == "__main__":
    main()
