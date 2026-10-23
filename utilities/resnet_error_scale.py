"""Prediction error of the ResNet model relative to the motion of the drone.

The RMS prediction errors of test_resnet.py are absolute numbers; this script
puts them on a scale. For every quantity (position, velocity, tilt, payload
swing; vector norms) and look-ahead it computes

    typical value   95th percentile of the magnitude in the flights
                    (speed, tilt, swing; not for the position)
    change          RMS of |x(k+h) - x(k)|, i.e. how much the quantity changes
                    during the look-ahead (the error of predicting "no change")
    error           RMS of |x_pred(k+h) - x(k+h)|
    relative        error / change

on two sets of flights that were not used for training:

    test   the excited test flights of test_resnet.py (test_flights.pkl)
    RL     flights of the RL director policy (seeds 900+), i.e. the fast,
           aggressive motion that the safety filter has to predict

Usage:
    python -m utilities.resnet_error_scale                 # flies the RL flights once (~1 min)
    python -m utilities.resnet_error_scale --horizon 20    # look-ahead [steps] (default: MPC horizon)
"""

import argparse
import csv
import os
import pickle

import numpy as np

OUT = "results/analysis/identification"
RL_SEEDS = tuple(range(900, 924))
QUANTITIES = [  # name, unit, scale, state indices, typical value shown
    ("position", "cm", 100.0, (0, 1, 2), False),
    ("velocity", "cm/s", 100.0, (3, 4, 5), True),
    ("tilt", "deg", np.degrees(1.0), (13, 14), True),
    ("swing", "deg", np.degrees(1.0), (9, 10), True),
]


def rl_flights(path):
    from multi_drone_mujoco.envs.resnet_mpc import (DIRECTOR_POLICY, RL_DURATION, _collect_rl,
                                                    collect_runs)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    runs = collect_runs([(_collect_rl, (s, DIRECTOR_POLICY, RL_DURATION)) for s in RL_SEEDS], workers=12)
    with open(path, "wb") as f:
        pickle.dump(runs, f)
    return runs


def scale_rows(models, runs, label, h):
    from utilities.test_resnet import window_errors
    err = window_errors(models, runs, stride=2)
    rows = []
    for mode in (0, 1):
        d = err[mode]
        T, P = d["true"], d["pred"]
        for name, unit, sc, idx, typical in QUANTITIES:
            if name == "swing" and mode == 0:
                continue
            idx = list(idx)
            change = np.linalg.norm(T[:, h, idx] - T[:, 0, idx], axis=1) * sc
            error = np.linalg.norm(P[:, h, idx] - T[:, h, idx], axis=1) * sc
            mag = np.linalg.norm(T[:, 0, idx], axis=1) * sc
            rows.append({"flights": label, "mode": "payload" if mode else "no payload", "quantity": name,
                         "unit": unit, "typical_p95": round(float(np.percentile(mag, 95)), 2) if typical else "",
                         "change_rms": round(float(np.sqrt(np.mean(change ** 2))), 2),
                         "error_rms": round(float(np.sqrt(np.mean(error ** 2))), 2),
                         "relative_pct": round(float(100 * np.sqrt(np.mean(error ** 2))
                                                     / np.sqrt(np.mean(change ** 2))), 1),
                         "windows": len(T)})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--horizon", type=int, default=20, help="look-ahead [steps] (MPC horizon: 20)")
    p.add_argument("--out", default=OUT)
    a = p.parse_args()

    import torch

    from multi_drone_mujoco.envs.resnet_mpc import load_model
    torch.set_num_threads(8)
    models = load_model()
    with open(os.path.join(a.out, "test_flights.pkl"), "rb") as f:
        test = [r for r in pickle.load(f) if np.any(r["phase"] == "load")]
    rl = rl_flights(os.path.join(a.out, "rl_flights.pkl"))
    rows = scale_rows(models, test, "test", a.horizon) + scale_rows(models, rl, "RL", a.horizon)
    path = os.path.join(a.out, f"error_scale_h{a.horizon}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"look-ahead {a.horizon / 48:.2f} s")
    for r in rows:
        print(f"  {r['flights']:4s} {r['mode']:10s} {r['quantity']:8s} typical {str(r['typical_p95']):>6s}"
              f"  change {r['change_rms']:6.2f}  error {r['error_rms']:5.2f} {r['unit']:4s}"
              f"  ({r['relative_pct']:5.1f} %)  [{r['windows']} windows]")
    print("saved:", path)


if __name__ == "__main__":
    main()
