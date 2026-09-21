"""One bounded cross-episode shuffle of the factual P2 optional residual.

Uses the original recovery source/checkpoint. Holds the primary refined W
cache fixed; this is not a whole-P2 or full-lifecycle world intervention.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch


def cross_episode_panel(refs, *, batches=8, batch_size=8):
    by_episode = defaultdict(list)
    for index, ref in enumerate(refs):
        by_episode[ref.episode_idx].append(index)
    episodes = sorted(by_episode)
    if len(episodes) < 3:
        raise ValueError("require at least three independent validation episodes")
    sequence = [episodes[index % len(episodes)] for index in range(batches * batch_size)]
    totals, used = Counter(sequence), Counter()
    selected = []
    for episode in sequence:
        index = round((used[episode] + 0.5) * (len(by_episode[episode]) - 1) / totals[episode])
        selected.append(by_episode[episode][index])
        used[episode] += 1
    if len(set(selected)) != len(selected):
        raise ValueError("panel would repeat windows")
    groups = [selected[start:start + batch_size] for start in range(0, len(selected), batch_size)]
    for group in groups:
        donor_ids = [refs[index].episode_idx for index in group]
        if any(donor_ids[index] == donor_ids[(index - 1) % batch_size] for index in range(batch_size)):
            raise ValueError("cyclic permutation must use a different episode for every sample")
    return groups


class OptionalResidualShuffle:
    """Swap only router outputs, keeping the protected carrier untouched."""

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.mode = "record"
        self.records = []
        self.cursor = 0

    def reset(self, mode):
        self.mode, self.cursor = mode, 0
        if mode == "record":
            self.records = []

    def hook(self, _module, args, result):
        carrier = args[0]
        delta, metrics = result
        if delta.ndim != 3 or delta.shape[0] % self.batch_size:
            raise ValueError("local refiner flattened shape does not preserve sample-major order")
        if self.mode == "record":
            self.records.append((carrier.detach().clone(), delta.detach().clone()))
            return result
        if self.cursor >= len(self.records):
            raise AssertionError("refiner call count changed")
        expected_carrier, expected_delta = self.records[self.cursor]
        self.cursor += 1
        if not torch.equal(carrier, expected_carrier) or not torch.equal(delta, expected_delta):
            raise AssertionError("upstream/selector changed before the intended shuffle boundary")
        if self.mode == "identity":
            return result
        value = delta.reshape(self.batch_size, -1, *delta.shape[1:])
        return value.roll(1, dims=0).reshape_as(delta), metrics

    def summary(self):
        count = sum(delta.numel() for _, delta in self.records)
        carrier_count = sum(carrier.numel() for carrier, _ in self.records)
        delta_sq = carrier_sq = change_sq = 0.0
        for carrier, delta in self.records:
            shuffled = delta.reshape(self.batch_size, -1, *delta.shape[1:]).roll(1, dims=0).reshape_as(delta)
            delta_sq += float(delta.float().square().sum())
            carrier_sq += float(carrier.float().square().sum())
            change_sq += float((shuffled.float() - delta.float()).square().sum())
        if not count:
            raise AssertionError("optional router was not executed")
        return {"router_calls": len(self.records), "carrier_rms": (carrier_sq / carrier_count) ** 0.5,
                "optional_rms": (delta_sq / count) ** 0.5, "shuffled_delta_rms": (change_sq / count) ** 0.5}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--readonly-utils", type=Path, required=True)
    args = p.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.repo_root.resolve()))
    os.chdir(args.repo_root)
    from clearvla.mainline.checkpoint import (
        active_source_snapshot,
        checkpoint_identity_from_mapping,
        git_commit,
    )
    from clearvla.mainline.config import config_from_mapping
    from clearvla.mainline.data.loading import load_mainline_data, to_training_batch
    from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
    from clearvla.mainline.runtime import sampling
    from clearvla.mainline.runtime.evaluation import ValidationAccumulator
    from clearvla.mainline.runtime.identity import dataset_identity, language_identity

    spec = importlib.util.spec_from_file_location("readonly_utils", args.readonly_utils)
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    free, total = torch.cuda.mem_get_info(device)
    if total - free > 2 * 1024**3:
        raise RuntimeError("selected GPU is occupied")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "context.json").exists():
        raise FileExistsError("use a new probe output directory")
    utils._writes_only_in(args.output_dir)
    with args.checkpoint.open("rb") as stream:
        sha = hashlib.sha256()
        while chunk := stream.read(8 * 1024**2):
            sha.update(chunk)
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu", weights_only=False)
    config = config_from_mapping(payload["config"])
    identity = checkpoint_identity_from_mapping(payload["identity"])
    if identity.git_commit != "0973f1920cb8467e3b5f048aaaa180e34e27c83f":
        raise ValueError("not the declared schema28 recovery checkpoint")
    if git_commit(args.repo_root) != identity.git_commit or active_source_snapshot(args.repo_root) != identity.source:
        raise ValueError("checkpoint and source differ")
    if config.digest(include_paths=False) != identity.config_digest:
        raise ValueError("config identity differs")
    bundle = load_mainline_data(config)
    if dataset_identity(bundle, config) != identity.dataset or language_identity(bundle, config) != identity.language:
        raise ValueError("data/normalizer/language identity differs")
    model = ClearVLAMainlinePolicy(config).to(device).eval().requires_grad_(False)
    model.load_state_dict(payload["model"], strict=True)
    before = utils._tensor_digest(model.state_dict())
    dataset = bundle.datasets["val"]
    panel = cross_episode_panel(dataset.base.refs)
    routers = [(name, module.delta_router) for name, module in model.named_modules()
               if type(module).__name__ == "_UtilityPrecisionLocalRefiner"]
    if not routers:
        raise RuntimeError("active checkpoint has no expected P2 optional router")
    intervention = OptionalResidualShuffle(8)
    context = {"schema": "recovery-p2-optional-cross-episode-shuffle-v1", "git": identity.git_commit,
               "checkpoint_sha256": sha.hexdigest(), "epoch": payload["epoch"], "step": payload["global_step"],
               "source_digest": identity.source.digest, "data_identity": identity.as_dict()["dataset"],
               "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "scope": "P2 factual optional routed output only; fixed primary refined W; not whole P2",
               "batches": 8, "samples": 64, "validation_samples": len(dataset),
               "routers": [name for name, _ in routers], "selected_indices": panel,
               "initial_noise_seeds": [37337 + i for i in range(8)], "schedule": "original uniform E5",
               "optimizer_loaded": False, "checkpoint_writes": False, "state_before_sha256": before}
    utils._json(args.output_dir / "context.json", context)
    del payload
    def accumulator():
        return ValidationAccumulator.from_action_normalizer(bundle.action_normalizer, device=device,
            gripper_event_threshold=config.objectives.gripper_event_threshold,
            arm_motion_threshold=config.objectives.arm_motion_threshold)
    aggregates = {name: accumulator() for name in ("primary", "shuffle")}
    rows = []
    started = time.perf_counter()
    with torch.no_grad():
        for index, selected in enumerate(panel):
            raw = torch.utils.data.default_collate([dataset[i] for i in selected])
            batch = to_training_batch(raw, goal=bundle.goal, config=config, device=device)
            intervention.reset("record")
            handles = [router.register_forward_hook(intervention.hook) for _, router in routers]
            rng_cpu, rng_gpu = torch.get_rng_state(), torch.cuda.get_rng_state(device)
            try:
                cache, _ = sampling.deployment_cache(model, batch.online, config, dtype=torch.bfloat16)
                if index == 0:
                    intervention.reset("identity")
                    torch.set_rng_state(rng_cpu)
                    torch.cuda.set_rng_state(rng_gpu, device)
                    identical, _ = sampling.deployment_cache(model, batch.online, config, dtype=torch.bfloat16)
                    assert intervention.cursor == len(intervention.records)
                    assert torch.equal(cache.factual_dock.protected_detail, identical.factual_dock.protected_detail)
                    del identical
                intervention.reset("shuffle")
                torch.set_rng_state(rng_cpu)
                torch.cuda.set_rng_state(rng_gpu, device)
                shuffled_cache, _ = sampling.deployment_cache(model, batch.online, config, dtype=torch.bfloat16)
                assert intervention.cursor == len(intervention.records)
                boundary = intervention.summary()
            finally:
                for handle in handles:
                    handle.remove()
            # Only transplant the changed factual dock. The primary refined
            # world, transition source, history and initial noise remain fixed.
            primary, refined_cache = sampling.sample_refined_cached_action_with_cache(
                model, cache, config, dtype=torch.bfloat16,
                generator=torch.Generator(device=device).manual_seed(37337 + index))
            target_cache = dataclasses.replace(refined_cache, factual_dock=shuffled_cache.factual_dock)
            if index == 0:
                control = sampling.sample_cached_action(model, refined_cache, config, dtype=torch.bfloat16,
                    initial_physical_noise=primary.initial_physical_noise)
                assert torch.equal(control.action, primary.action)
                utils._json(args.output_dir / "parity.json", {"identity_hook_bit_exact": True, "fixed_W_identity_bit_exact": True})
                del control
            changed = sampling.sample_cached_action(model, target_cache, config, dtype=torch.bfloat16,
                initial_physical_noise=primary.initial_physical_noise)
            if not torch.isfinite(changed.action).all():
                raise FloatingPointError("nonfinite shuffled action")
            boundary["factual_dock_delta_rms"] = float((cache.factual_dock.protected_detail.float()
                - shuffled_cache.factual_dock.protected_detail.float()).square().mean().sqrt())
            scale = torch.as_tensor(bundle.action_normalizer.scale, device=device).reshape(1, 1, -1)
            delta = (changed.action.float() - primary.action.float()) / scale
            row = {"batch": index, "indices": selected,
                   "episodes": [dataset.base.refs[i].episode_idx for i in selected],
                   "donor_episodes": [dataset.base.refs[selected[(i - 1) % 8]].episode_idx for i in range(8)],
                   "boundary": boundary, "action_delta_rmse_native": float(delta.square().mean().sqrt()),
                   "arm_delta_rmse_native": float(delta[..., :6].square().mean().sqrt()),
                   "gripper_delta_rmse_native": float(delta[..., 6:].square().mean().sqrt())}
            for name, result in (("primary", primary), ("shuffle", changed)):
                aggregates[name].update(result.action, batch, motion_logits=result.motion_logits)
                per = accumulator()
                per.update(result.action, batch, motion_logits=result.motion_logits)
                row[name] = per.means()
            rows.append(row)
            print(json.dumps({"batch_done": index + 1, "boundary": boundary,
                "action_delta_rmse_native": row["action_delta_rmse_native"], "elapsed": time.perf_counter() - started}), flush=True)
            del primary, changed, cache, refined_cache, target_cache, shuffled_cache, batch
    after = utils._tensor_digest(model.state_dict())
    if before != after:
        raise AssertionError("model state changed")
    utils._json(args.output_dir / "summary.json", {"context": context, "rows": rows,
        "aggregate": {key: acc.means() for key, acc in aggregates.items()},
        "state_unchanged": True, "seconds": time.perf_counter() - started})
    print("P2_OPTIONAL_SHUFFLE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
