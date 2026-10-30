"""One annotated movie of a scenario: RL director, MPC director, RL director +
safety filter, then the side-by-side comparison, one after the other.

Input: the videos and the logged episodes of
    python -m utilities.analyse --test scenario --seed <seed> --max_time 25 --video \
        --video_width 960 --tag seed_<seed>_video [--video_camera X Y Z DIST AZ EL]
(seed 1290: --video_camera 0.75 0.9 0.55 3.2 90 -20, as the rescue scene)
Every part starts with a title card; captions are timed from the logged
episode (grasp, overshoot, filter interventions, flip over, goal reached,
hold). Output: <video dir>/video_story.mp4 (1920 x 1080, 24 fps).

Usage:
    python -m utilities.make_scenario_movie --run results/analysis/scenario/seed_1290_video
"""

import argparse
import json
import os
import subprocess

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utilities.analyse import GOAL_HOLD_TIME, GOAL_TOLERANCE, REASONS

W, H, FPS = 1920, 1080, 24
FONT_DIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
font = lambda size, bold=False: ImageFont.truetype(
    os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), size)
BG = (22, 27, 38)
COLORS = {"director": (59, 111, 182), "director_mpc": (224, 138, 44), "director_safety": (27, 163, 156)}
NAMES = {"director": "RL director", "director_mpc": "MPC director",
         "director_safety": "RL director + safety filter"}
RED, MAGENTA, GREEN, WHITE = (220, 60, 50), (210, 60, 200), (70, 190, 90), (240, 240, 240)


################################################################################
# EVENTS (from the logged episode)
################################################################################

def events(kind, ep, res):
    """Captions of one method: [(t_start, t_end, text, color)] and the end
    time of its part [s]."""
    t, att = ep["t"], ep["attached"].astype(bool)
    goal = ep["goal"]
    dist = np.linalg.norm(ep["p"] - goal, axis=1)
    tilt = np.degrees(np.abs(ep["rpy"][:, 0:2]).max(axis=1))
    t_end = float(t[-1])
    ev = []
    if att.any():
        k_att = int(np.argmax(att))
        ev.append((t[k_att], t[k_att] + 1.2, "grasps and lifts the payload", WHITE))
    if kind == "director_safety":
        # intervention spans (gaps below 0.2 s merged), shown at least 0.8 s
        iv = ep["intervened"].astype(bool)
        spans, k = [], 0
        while k < len(t):
            if iv[k]:
                k0 = k
                while k + 1 < len(t) and (iv[k + 1] or (k + 5 < len(t) and iv[k + 1:k + 10].any())):
                    k += 1
                spans.append((k0, k))
            k += 1
        for k0, k1 in spans:
            why = REASONS[int(ep["reason"][k0])]
            label = {"constraint": "filter intervenes: predicted constraint violation",
                     "contact": "filter intervenes: hook contact, climb",
                     "stall": "filter: MPC takeover (stalled)"}.get(why, "filter intervenes")
            ev.append((t[k0], max(t[k1], t[k0] + 0.8), label, MAGENTA))
    if res["outcome"] == "success":
        tg = res["event_time"]
        ev.append((tg, tg + 1.5, f"reaches the goal at {tg:.1f} s", GREEN))
        ev.append((tg + 1.5, tg + GOAL_HOLD_TIME, f"holds the payload at the goal ({GOAL_HOLD_TIME:g} s)", GREEN))
        t_end = tg + GOAL_HOLD_TIME
    elif res["outcome"] == "stability":
        tf = res["event_time"]
        k_f = int(np.searchsorted(t, tf))
        # overshoot: closest approach to the goal with the payload before the flip
        cand = np.flatnonzero(att[:k_f])
        if len(cand):
            k_min = cand[np.argmin(dist[cand])]
            if dist[k_min] < 0.5 and dist[k_f - 1] > dist[k_min] + 0.2:
                k_tilt = k_min + int(np.argmax(tilt[k_min:k_f] > 35)) if (tilt[k_min:k_f] > 35).any() else k_f
                ev.append((t[k_min], t[k_tilt], "too fast: overshoots the goal", RED))
                if k_tilt < k_f:
                    ev.append((t[k_tilt], tf, "the swinging payload pulls the drone over", RED))
        ev.append((tf, tf + 2.0, f"flips over at {tf:.1f} s (tilt > 90°)", RED))
        t_end = tf + 2.0
    return ev, t_end


################################################################################
# DRAWING
################################################################################

def read_frames(path):
    """All frames of a video as (n, h, w, 3) uint8."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height", "-of", "csv=p=0", path],
                           capture_output=True, text=True, check=True).stdout.strip().split(",")
    w, h = int(probe[0]), int(probe[1])
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def caption(d, xy, text, color, size=40, pad=12):
    f = font(size, bold=True)
    x, y = xy
    b = d.textbbox((x, y), text, font=f)
    d.rounded_rectangle([b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad], radius=10, fill=(0, 0, 0, 170))
    d.text((x, y), text, font=f, fill=color)


def card(lines, sub=None):
    """Title card: (text, size, color) lines centered."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    y = H // 2 - sum(s + 24 for _, s, _ in lines) // 2
    for text, size, color in lines:
        f = font(size, bold=size >= 50)
        w = d.textlength(text, font=f)
        d.text(((W - w) / 2, y), text, font=f, fill=color)
        y += size + 24
    return np.asarray(img)


def side_panel(d, kind, ev, t, res):
    """Right panel of a single-method part: name, time, event log."""
    x0 = 1460
    d.rectangle([x0 - 20, 0, W, H], fill=BG)
    d.rectangle([x0 - 20, 0, x0 - 12, H], fill=COLORS[kind])
    d.text((x0, 40), NAMES[kind].replace(" + ", "\n+ "), font=font(34, True), fill=COLORS[kind])
    d.text((x0, 150), f"t = {t:5.2f} s", font=font(40, True), fill=WHITE)
    y = 240
    d.text((x0, y), "events", font=font(24), fill=(150, 160, 175))
    y += 40
    for t0, _, text, color in ev:
        if t0 <= t + 1e-9:
            for k, line in enumerate(_wrap(f"{t0:4.1f} s  {text}", 30)):
                d.text((x0, y), line, font=font(22), fill=color)
                y += 28
            y += 8


def _wrap(text, n):
    words, lines, cur = text.split(), [], ""
    for w_ in words:
        if len(cur) + len(w_) + 1 > n and cur:
            lines.append(cur)
            cur = w_
        else:
            cur = f"{cur} {w_}".strip()
    return lines + [cur]


def method_part(kind, frames, ep, res, writer):
    ev, t_end = events(kind, ep, res)
    n = min(len(frames), int(np.ceil(t_end * FPS)) + 1)
    for i in range(n):
        t = i * 2.0 / 48.0 + 1.0 / 48.0               # frame i shows control step 2 i
        img = Image.new("RGB", (W, H), BG)
        img.paste(Image.fromarray(frames[i]).resize((1440, 1080), Image.LANCZOS), (0, 0))
        d = ImageDraw.Draw(img, "RGBA")
        side_panel(d, kind, ev, t, res)
        active = [e for e in ev if e[0] <= t <= e[1]]
        for k, (_, _, text, color) in enumerate(active[-2:]):
            caption(d, (40, 960 - 80 * k), text, color)
        writer.write(np.asarray(img))


def comparison_part(frames, eps, ress, writer):
    order = ["director", "director_mpc", "director_safety"]
    n = len(frames)
    for i in range(n):
        t = i * 2.0 / 48.0 + 1.0 / 48.0
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img, "RGBA")
        d.text((40, 40), "Comparison: the same scenario, side by side", font=font(44, True), fill=WHITE)
        d.text((40, 110), f"t = {t:5.2f} s", font=font(36, True), fill=WHITE)
        img.paste(Image.fromarray(frames[i]).resize((1920, 480), Image.LANCZOS), (0, 200))
        for c, kind in enumerate(order):
            x = 40 + 640 * c
            d.text((x, 700), NAMES[kind], font=font(30, True), fill=COLORS[kind])
            ev, _ = events(kind, eps[kind], ress[kind])
            done = [e for e in ev if e[0] <= t + 1e-9]
            y = 750
            for t0, _, text, color in done[-5:]:
                for line in _wrap(f"{t0:4.1f} s  {text}", 38):
                    d.text((x, y), line, font=font(21), fill=color)
                    y += 27
                y += 6
        writer.write(np.asarray(img))


class Writer:
    def __init__(self, path):
        self.p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                                   "-pix_fmt", "yuv420p", "-crf", "20", path], stdin=subprocess.PIPE)

    def write(self, frame, repeat=1):
        b = np.ascontiguousarray(frame, dtype=np.uint8).tobytes()
        for _ in range(repeat):
            self.p.stdin.write(b)

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", default="results/analysis/scenario/seed_1290_video")
    p.add_argument("--card", type=float, default=3.0, help="duration of a title card [s]")
    a = p.parse_args()
    sc = json.load(open(os.path.join(a.run, "scenario.json")))
    ress = json.load(open(os.path.join(a.run, "results.json")))["methods"]
    kinds = ["director", "director_mpc", "director_safety"]
    eps = {k: dict(np.load(os.path.join(a.run, f"{k}.npz"))) for k in kinds}
    out = os.path.join(a.run, "video_story.mp4")
    wr = Writer(out)
    rep = int(a.card * FPS)
    mass = sc["payload_mass"] + 0.01
    wr.write(card([("A rescued scenario", 64, WHITE),
                   (f"payload {mass:.2f} kg, radius {100 * sc['payload_radius']:.1f} cm", 36, (180, 190, 205)),
                   ("take off  →  grasp the payload  →  carry it to the goal  →  hold it there for 5 s", 32,
                    (180, 190, 205)),
                   ("the same scenario flown by three controllers", 32, (180, 190, 205))]), rep)
    texts = {"director": "the learned high-level policy alone",
             "director_mpc": "ResNet MPC on the active waypoint",
             "director_safety": "the same RL policy, its commands checked by the predictive safety filter"}
    for n, kind in enumerate(kinds, 1):
        r = ress[kind]
        result = (f"result: goal reached at {r['event_time']:.1f} s" if r["outcome"] == "success"
                  else f"result: flips over at {r['event_time']:.1f} s")
        wr.write(card([(f"{n} / 4", 36, (150, 160, 175)), (NAMES[kind], 64, COLORS[kind]),
                       (texts[kind], 34, (200, 205, 215)), (result, 34, WHITE)]), rep)
        method_part(kind, read_frames(os.path.join(a.run, f"video_{kind}.mp4")), eps[kind], ress[kind], wr)
    wr.write(card([("4 / 4", 36, (150, 160, 175)), ("Comparison", 64, WHITE),
                   ("RL, MPC and RL + safety filter side by side", 34, (200, 205, 215))]), rep)
    comparison_part(read_frames(os.path.join(a.run, "video_comparison.mp4")), eps, ress, wr)
    lines = [("Summary", 60, WHITE)]
    for kind in kinds:
        r = ress[kind]
        res = (f"goal reached at {r['event_time']:.1f} s" if r["outcome"] == "success"
               else f"flips over at {r['event_time']:.1f} s")
        extra = f", filter active in {r['filter_active_percent']:.1f} % of the steps" if kind == "director_safety" else ""
        lines.append((f"{NAMES[kind]}:  {res}{extra}", 34, COLORS[kind]))
    wr.write(card(lines), int(1.5 * rep))
    wr.close()
    print("saved:", out)


if __name__ == "__main__":
    main()
