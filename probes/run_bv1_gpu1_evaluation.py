"""Adopt existing B-v1 training; place standard panels on the requested GPU1.

Only the verified old scheduling parent receives SIGTERM. The existing train
child completes unchanged. Its success is established by the wrapper's final
summary, complete epoch/exposure, finite checkpoint and pinned identities.
"""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import time


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def process(pid):
    root = Path("/proc") / str(pid)
    try:
        fields = (root / "stat").read_text().split(") ", 1)[1].split()
        if fields[0] == "Z":
            return None
        return dict(pid=pid, starttime=int(fields[19]), ppid=int(fields[1]),
                    uid=root.stat().st_uid,
                    command=(root / "cmdline").read_bytes().replace(b"\0", b" ").decode().strip())
    except FileNotFoundError:
        return None


def same_process(expected):
    current = process(expected["pid"])
    return current is not None and all(current[k] == expected[k]
                                      for k in ("pid", "starttime", "uid", "command"))


def terminate_scheduler(parent, child):
    """Narrow PID signal, never a process-group signal or a training signal."""
    if not same_process(parent) or not same_process(child):
        raise ValueError("scheduling/training process identity changed")
    if parent["uid"] != os.getuid() or child["uid"] != os.getuid():
        raise ValueError("handoff may only adopt this user's processes")
    if process(child["pid"])["ppid"] != parent["pid"]:
        raise ValueError("training is no longer the verified scheduling child")
    os.kill(parent["pid"], signal.SIGTERM)
    deadline = time.monotonic() + 15
    while same_process(parent):
        if time.monotonic() >= deadline:
            raise TimeoutError("old scheduling parent did not exit; do not launch a panel")
        time.sleep(.2)
    if not same_process(child):
        raise RuntimeError("training disappeared during handoff; inspect original logs")


def gpu_rows():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free,memory.total",
        "--format=csv,noheader,nounits"], text=True)
    return [dict(index=p[0], uuid=p[1], used=int(p[2]), free=int(p[3]), total=int(p[4]))
            for p in (list(map(str.strip, line.split(","))) for line in output.splitlines())]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    h = json.loads(args.receipt.read_text())
    r = json.loads(Path(h["original_receipt"]).read_text())
    source = Path(r["source"])
    old = Path(r["output"])
    out = Path(h["output"])
    if sha(h["original_receipt"]) != h["original_receipt_sha256"]:
        raise ValueError("original receipt changed")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != r["commit"]:
        raise ValueError("immutable training source differs")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip():
        raise ValueError("immutable training source dirty")
    for name, digest in r["sha256"].items():
        if sha(name) != digest:
            raise ValueError("pinned input changed: " + name)
    status = json.loads((old / "status.json").read_text())
    if status.get("stage") != "short1024" or status.get("pid") != h["training_process"]["pid"]:
        raise ValueError("old scheduling stage has changed")
    if not same_process(h["old_scheduler"]) or not same_process(h["training_process"]):
        raise ValueError("handoff process identities differ")
    if h["old_scheduler"]["pid"] == h["training_process"]["pid"]:
        raise ValueError("scheduler and training must be separate processes")
    if "run_bv1_address_short.py" not in h["old_scheduler"]["command"]:
        raise ValueError("not the declared old scheduler")
    if r["training_audit"] not in h["training_process"]["command"]:
        raise ValueError("not the declared training wrapper")
    if h["target_gpu"] != "GPU-24cf9b56-d67a-4e35-6761-225b0d4383b5":
        raise ValueError("this continuation is explicitly assigned to GPU1")
    if h["audit_gpus"] != ["GPU-455c6794-4552-89b9-21bc-63b6e213a425"]:
        raise ValueError("later audits must use GPU2, preserving GPU0 for others")
    if args.check_only:
        print(json.dumps(dict(check_only=True, ready=True, parent=h["old_scheduler"]["pid"],
                              training=h["training_process"]["pid"], target_gpu=h["target_gpu"])))
        return
    out.mkdir(exist_ok=False)
    held = None

    def state(value, **kw):
        dump(out / "status.json", dict(state=value, unix_time=time.time(), formal_promoted=False, **kw))

    def acquire(allowed, shared_panel=False, ordered=False):
        nonlocal held
        deadline = time.monotonic() + 72 * 3600
        while time.monotonic() < deadline:
            if ordered:
                me = json.loads((out / "panel-ready.json").read_text())
                earlier = False
                for peer in h["peer_outputs"]:
                    peer = Path(peer)
                    ready = peer / "panel-ready.json"
                    if not ready.exists() or (peer / "panel-complete.json").exists():
                        continue
                    ps = json.loads((peer / "status.json").read_text())
                    item = json.loads(ready.read_text())
                    if ps.get("state") != "failed" and (item["checkpoint_mtime_ns"], str(peer)) < (me["checkpoint_mtime_ns"], str(out)):
                        earlier = True
                if earlier:
                    state("waiting_gpu1", reason="earlier completed checkpoint has priority")
                    time.sleep(20)
                    continue
            for g in gpu_rows():
                usable = shared_panel or g["used"] < 512
                if g["uuid"] not in allowed or not usable:
                    continue
                lock_path = Path(h["panel_lock"]) if shared_panel else Path(r["gpu_locks"]) / (g["uuid"] + ".lock")
                handle = lock_path.open("a")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                fresh = next(x for x in gpu_rows() if x["uuid"] == g["uuid"])
                usable = shared_panel or fresh["used"] < 512
                if not usable:
                    handle.close()
                    continue
                held = handle
                return fresh
            state("waiting_gpu1" if shared_panel else "waiting_audit_gpu",
                  shared_panel=shared_panel, allowed_gpus=allowed)
            time.sleep(20)
        raise TimeoutError("no suitable GPU within72h; existing jobs were not stopped")

    def execute(stage, command, gpu, deterministic=False):
        env = dict(os.environ, PYTHONPATH=str(source), CUDA_VISIBLE_DEVICES=gpu["uuid"],
                   EGL_VISIBLE_DEVICES=gpu["index"], OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                   MKL_NUM_THREADS="4", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   PYTORCH_ALLOC_CONF="expandable_segments:True")
        env.pop("CUBLAS_WORKSPACE_CONFIG", None)
        if deterministic:
            env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        with (out / (stage + ".log")).open("x") as log:
            child = subprocess.Popen(command, cwd=source, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            state("running", stage=stage, pid=child.pid, gpu=gpu["uuid"], egl=gpu["index"], command=command)
            code = child.wait()
        if code:
            raise RuntimeError(stage + " failed; original logs retained")

    try:
        dump(out / "original-status.json", status)
        state("taking_over_scheduler", training_pid=h["training_process"]["pid"])
        terminate_scheduler(h["old_scheduler"], h["training_process"])
        # The original child keeps its log FDs/session and runs without a signal.
        handle = (Path(r["gpu_locks"]) / (status["gpu"] + ".lock")).open("a")
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        held = handle
        dump(old / "status.json", dict(state="handed_off", unix_time=time.time(),
             training_pid=h["training_process"]["pid"], continuation_receipt=str(args.receipt),
             continuation_status=str(out / "status.json"), target_gpu=h["target_gpu"],
             scope="only scheduling parent replaced; original training continues"))
        state("waiting_existing_training_and_offline", training_pid=h["training_process"]["pid"], training_gpu=status["gpu"])
        deadline = time.monotonic() + 48 * 3600
        while same_process(h["training_process"]):
            if time.monotonic() >= deadline:
                raise TimeoutError("existing training did not finish in48h")
            time.sleep(20)
        summary = json.loads((old / "short-updates/summary.json").read_text())
        if summary.get("complete") is not True or summary.get("updates") != 1024 or summary.get("batch_size") != 8:
            raise ValueError("existing training wrapper did not complete")
        held.close()
        held = None
        state("auditing_final_checkpoint")
        import torch
        from clearvla.mainline.config import load_config
        cfg = load_config(r["short_config"])
        run = Path(cfg.data.output_dir)
        ck = run / "checkpoints/latest.pt"
        payload = torch.load(ck, map_location="cpu", weights_only=False, mmap=True)
        if payload["epoch"] != 1 or payload["global_step"] != 13060 or payload["identity"]["git_commit"] != r["commit"] or payload["identity"]["config_digest"] != cfg.digest():
            raise ValueError("final checkpoint identity differs")
        if not all(bool(torch.isfinite(v).all()) for v in payload["model"].values() if isinstance(v, torch.Tensor) and v.is_floating_point()):
            raise ValueError("saved model has nonfinite values")
        rows = [json.loads(x) for x in (run / "metrics.jsonl").read_text().splitlines()]
        training = [x for x in rows if x.get("kind") == "train"]
        epochs = [x for x in rows if x.get("kind") == "epoch"]
        if sum(x["window_batches"] for x in training) != 1024 or sum(x["window_samples"] for x in training) != 8192 or len(epochs) != 1:
            raise ValueError("training exposure incomplete")
        for row in rows:
            for key in ("metrics", "train", "validation"):
                if any(isinstance(v, float) and not math.isfinite(v) for v in row.get(key, {}).values()):
                    raise ValueError("nonfinite completed metrics")
        audit = dict(complete=True, source=r["commit"], checkpoint=str(ck), sha256=sha(ck),
                     config_digest=cfg.digest(), epoch=1, global_step=13060, updates=1024,
                     batch_size=8, validation=epochs[0]["validation"])
        dump(out / "short-audit.json", audit)
        del payload
        done = out / "training-complete.json"
        dump(done, dict(status="training_and_offline_finished", returncode=0,
                        success_basis="existing wrapper summary plus actual checkpoint and complete epoch"))
        dump(out / "panel-ready.json", dict(checkpoint_mtime_ns=ck.stat().st_mtime_ns, sha256=audit["sha256"]))
        gpu = acquire([h["target_gpu"]], shared_panel=True, ordered=True)
        if gpu["index"] != "1":
            raise ValueError("GPU1 UUID/index mapping differs")
        dump(out / "panel-placement.json", dict(gpu=gpu,
             shared_gpu_allowed=True, checkpoint_sha256=audit["sha256"], unix_time=time.time()))
        panel = Path(r["root"]) / (r["name"] + "-closed-loop-replan8")
        execute("standard18", [r["python"], "-B", "-u", r["panel_runner"], "--repo", str(source),
            "--run", str(run), "--manifest", r["panel_manifest"], "--output", str(panel),
            "--training-status", str(done), "--source", r["commit"], "--step", "13060", "--epoch", "1",
            "--checkpoint-file", "latest.pt", "--gpu", gpu["uuid"], "--egl", "1", "--port", str(r["port"])], gpu)
        q = json.loads((panel / "summary.json").read_text())
        if not q["complete"] or q["trials"] != 18 or q["errors"]:
            raise ValueError("standard18 incomplete")
        inventory = []
        for row in q["records"]:
            npz = Path(row["result_path"]).with_name("trajectory.npz")
            inventory.append(dict(index=row["index"], path=str(npz), sha256=sha(npz), bytes=npz.stat().st_size))
        dump(out / "trajectory-inventory.json", dict(complete=True, records=inventory))
        dump(out / "panel-complete.json", dict(complete=True, successes=q["successes"], trials=18))
        held.close()
        held = None
        # Keep GPU1 for panels and GPU0 free for others; preserve audits on GPU2.
        gpu = acquire(h["audit_gpus"])
        final = out / "final-factual4"
        execute("final-factual4", [r["python"], "-B", "-u", r["factual_probe"], "--checkpoint", str(ck),
            "--plan", r["factual_plan"], "--masks", r["factual_masks"], "--output", str(final), "--deterministic"], gpu, True)
        for stage in r.get("post_panel_stages", []):
            execute(stage["name"], stage["command"], gpu, stage.get("deterministic", False))
            value = json.loads(Path(stage["result"]).read_text())
            kind = stage["contract"]
            passed = (isinstance(value, list) and len(value) == 18) if kind == "plan18" else (
                value.get("status") == "complete" if kind == "status_complete" else value.get("complete") is True and bool(value.get("records")))
            if not passed:
                raise ValueError(stage["name"] + " output incomplete")
        state("complete", successes=q["successes"], trials=18, checkpoint=str(ck),
              scope="existing short, GPU1 panel and original audits complete; review required")
    except BaseException as exc:
        state("failed", error=repr(exc))
        raise
    finally:
        if held is not None:
            held.close()


if __name__ == "__main__":
    main()
