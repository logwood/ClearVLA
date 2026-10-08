"""Post-hoc, evaluator-only geometry/action/video audit of every NPZ episode.

No geometry, colour mask, phase label or oracle from this script enters a policy.
Distances are TCP-to-cube centre diagnostics, not grasp success definitions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def first(mask):
    ids = np.flatnonzero(mask)
    return int(ids[0]) if len(ids) else None


def colour_pixels(image, colour):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    b, g, r = image.astype(np.float32).transpose(2, 0, 1)
    # The wood table is saturated orange. Require narrow pure-colour hue and
    # channel dominance so it cannot masquerade as tens of thousands of cube pixels.
    hue = (((h < 4) | (h > 176)) & (r > 2*g) & (r > 2*b)) if colour == "red" else ((h > 40) & (h < 82) & (g > 1.8*r) & (g > 1.8*b))
    return int((hue & (s > 110) & (v > 45)).sum())


def analyze_episode(folder, output):
    output.mkdir(parents=True, exist_ok=True)
    a = dict(np.load(folder / "trajectory.npz", allow_pickle=False))
    report = json.loads((folder / "result.json").read_text())
    metrics = [json.loads(x)["metrics"] for x in (folder / "telemetry.jsonl").read_text().splitlines()]
    state, objects, commands = a["robot_obs_trajectory"], a["object_positions"], a["executed"]
    tcp, red, green = state[:, :3], objects[:, 0], objects[:, 1]
    xy = np.linalg.norm(tcp[:, :2] - red[:, :2], axis=1)
    dz = tcp[:, 2] - red[:, 2]
    grasp = np.asarray([x["is_cubeA_grasped"] for x in metrics], bool)
    forces = np.asarray([[np.linalg.norm(x[f"telemetry_cube_a_finger{i}_contact_force"]) for i in (1, 2)] for x in metrics])
    previous = state[:-1, -1]
    closes = np.flatnonzero((commands[:, -1] < 0) & (previous > 0))
    opens = np.flatnonzero((commands[:, -1] > 0) & (previous < 0))
    # A missed close is explicit geometry, not an inferred native grasp label.
    close_rows = [{"command_step": int(t), "xy_error_m": float(xy[t]),
                   "tcp_minus_red_z_m": float(dz[t]), "opening_m": float(state[t, 6]),
                   "any_grasp_next8": bool(grasp[t + 1:t + 9].any())} for t in closes]
    moving = np.linalg.norm(commands[:, :3], axis=1) > .01
    approach = np.arange(len(commands)) < (int(closes[0]) if len(closes) else 80)
    direction = red[:-1, :2] - tcp[:-1, :2]
    norms = np.linalg.norm(commands[:, :2], axis=1) * np.linalg.norm(direction, axis=1)
    cosine = (commands[:, :2] * direction).sum(1) / np.maximum(norms, 1e-12)
    mask = approach & moving & (norms > 1e-5)
    steps = [0, 24, 40, 64, 96, 160, 400]
    event_steps = sorted(set(steps + [int(np.argmin(xy))] + ([int(closes[0])] if len(closes) else []) + ([first(grasp)] if grasp.any() else [])))
    cap = cv2.VideoCapture(str(folder / "video.mp4"))
    frames, pixel_rows = {}, []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pixels = {f"{view}_{colour}": colour_pixels(frame[:336, lo:lo + 336], colour)
                  for view, lo in (("top", 0), ("wrist", 336)) for colour in ("red", "green")}
        pixel_rows.append(pixels)
        if index in event_steps:
            frames[index] = frame
        index += 1
    cap.release()
    assert index == len(state) == len(metrics)
    sheet = []
    for t in event_steps:
        if t not in frames:
            continue
        image = frames[t].copy()
        bar = np.full((32, image.shape[1], 3), 255, np.uint8)
        label = f"step {t} | XY error {xy[t]*100:.1f}cm | dz {dz[t]*100:.1f}cm | grasp {int(grasp[t])}"
        cv2.putText(bar, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (20, 20, 20), 1, cv2.LINE_AA)
        sheet.append(np.concatenate([bar, image]))
    cv2.imwrite(str(output / "stages.jpg"), np.concatenate(sheet), [cv2.IMWRITE_JPEG_QUALITY, 92])
    summary = {
        "seed": report["seed"], "steps": len(commands), "success": report["success"],
        "first_close_command_step": int(closes[0]) if len(closes) else None,
        "close_events": close_rows, "open_command_steps": opens.tolist(),
        "min_xy_error_m": float(xy.min()), "min_xy_step": int(np.argmin(xy)),
        "min_xy_below_red_plus_4cm_m": float(xy[dz < .04].min()) if (dz < .04).any() else None,
        "first_xy_within_2cm": first(xy < .02),
        "first_close_range": close_rows[0] if close_rows else None,
        "red_lift_max_m": float((red[:, 2] - red[0, 2]).max()),
        "red_xy_displacement_max_m": float(np.linalg.norm(red[:, :2] - red[0, :2], axis=1).max()),
        "grasp_observation_steps": np.flatnonzero(grasp).tolist(),
        "both_fingers_force_over_0p1N_steps": np.flatnonzero((forces > .1).all(1)).tolist(),
        "approach_xy_command_cosine_mean": float(cosine[mask].mean()) if mask.any() else None,
        "tcp_path_length_m": float(np.linalg.norm(np.diff(tcp, axis=0), axis=1).sum()),
        "top_red_pixels_reset": pixel_rows[0]["top_red"],
        "top_green_pixels_reset": pixel_rows[0]["top_green"],
        "colour_mask_note": "Compressed-video HSV pixel counts: diagnostic, not calibrated visibility/occlusion ground truth",
        "arm_command_rms": np.sqrt((commands[:, :6] ** 2).mean(0)).tolist(),
        "first8_mean_action": commands[:8].mean(0).tolist(),
        "initial_red_xyz": red[0].tolist(), "initial_green_xyz": green[0].tolist(),
    }
    np.savez_compressed(output / "derived.npz", xy_error=xy, height_error=dz, finger_forces=forces,
                        grasp=grasp, colour_pixel_counts=np.asarray([[row[k] for k in pixel_rows[0]] for row in pixel_rows]))
    (output / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(4, 2, figsize=(13, 12), constrained_layout=True)
    time = np.arange(len(state))
    ax = axes[0, 0]
    ax.plot(tcp[:, 0], tcp[:, 1], lw=1, label="TCP")
    ax.plot(red[:, 0], red[:, 1], "r", label="red cube")
    ax.plot(green[:, 0], green[:, 1], "g", label="green cube")
    ax.scatter(*tcp[0, :2], c="black", marker="x", label="TCP reset")
    ax.set(xlabel="world x (m)", ylabel="world y (m)", title="Full physical path", aspect="equal")
    ax.legend(fontsize=8)
    axes[0, 1].plot(time, xy * 100, label="horizontal TCP-to-red")
    axes[0, 1].plot(time, dz * 100, label="TCP height above red centre")
    axes[0, 1].set(ylabel="cm", title="Approach / alignment"); axes[0, 1].legend(fontsize=8)
    axes[1, 0].plot(time, state[:, 6] * 100, label="actual opening (cm)")
    axes[1, 0].step(time[:-1], commands[:, -1], label="command (+1 open / -1 close)")
    axes[1, 0].set(title="Gripper: physical state and applied command"); axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot(time, forces[:, 0], label="finger 1")
    axes[1, 1].plot(time, forces[:, 1], label="finger 2")
    axes[1, 1].set(ylabel="N", title="Red cube contact forces"); axes[1, 1].legend(fontsize=8)
    for j, name in enumerate("xyz"):
        axes[2, 0].plot(time[:-1], commands[:, j], label=name, lw=.8)
        axes[2, 1].plot(time[:-1], commands[:, j+3], label=name, lw=.8)
    axes[2, 0].set(title="Applied translation commands", ylabel="native normalized units")
    axes[2, 1].set(title="Applied rotation commands", ylabel="native normalized units")
    axes[2, 0].legend(fontsize=8); axes[2, 1].legend(fontsize=8)
    axes[3, 0].plot(time, (red[:, 2] - red[0, 2]) * 100, label="red lift (cm)")
    axes[3, 0].step(time, grasp.astype(int), label="native grasp flag")
    axes[3, 0].legend(fontsize=8); axes[3, 0].set(title="Native grasp and lift")
    for key in pixel_rows[0]:
        axes[3, 1].plot(time, [p[key] for p in pixel_rows], label=key, lw=.8)
    axes[3, 1].set(title="Visible colour support (compressed video)", ylabel="pixels")
    axes[3, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[1:]:
        for t in closes:
            ax.axvline(t, c="red", alpha=.15, lw=.8)
        ax.set_xlabel("executed control steps")
    fig.suptitle(f"Seed {report['seed']} — full 400-step trace (8-step replanning)")
    fig.savefig(output / "trace.png", dpi=140)
    plt.close(fig)
    return summary, tcp, red, green


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.panel / "manifest.json").read_text())
    assert manifest["complete"] and len(manifest["episodes"]) == manifest["requested_episodes"]
    summaries = []
    fig, axes = plt.subplots(3, 6, figsize=(20, 10), constrained_layout=True)
    for ep, ax in zip(manifest["episodes"], axes.flat):
        row, tcp, red, green = analyze_episode(args.panel / ep["directory"], args.output / ep["directory"])
        summaries.append(row)
        ax.plot(tcp[:, 0], tcp[:, 1], lw=1)
        ax.plot(red[:, 0], red[:, 1], c="red", lw=1)
        ax.scatter(red[0, 0], red[0, 1], c="red", marker="s", s=35)
        ax.scatter(green[0, 0], green[0, 1], c="green", marker="s", s=35)
        ax.scatter(tcp[0, 0], tcp[0, 1], c="black", marker="x", s=20)
        ax.set(xlim=(-.35, .35), ylim=(-.38, .38), aspect="equal", title=f"{row['seed']} | min XY {row['min_xy_error_m']*100:.1f}cm")
        ax.grid(alpha=.2)
        print(json.dumps({k: row[k] for k in ("seed", "first_close_range", "min_xy_error_m", "top_red_pixels_reset", "top_green_pixels_reset")}), flush=True)
    fig.suptitle("All 18 full paths — blue TCP; red target cube; green base cube; x reset TCP")
    fig.savefig(args.output / "all18_paths.png", dpi=150)
    plt.close(fig)
    (args.output / "panel_geometry.json").write_text(json.dumps({"source": str(args.panel), "episodes": summaries}, indent=2) + "\n")


if __name__ == "__main__":
    main()
