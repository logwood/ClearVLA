"""Independent black-box regressions for the two integration review findings."""

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from clearvla.mainline.checkpoint import (
    ArtifactIdentity,
    DatasetIdentity,
    build_checkpoint_identity,
)
from clearvla.mainline.config import ExperimentConfig, config_from_mapping
from clearvla.mainline.data.normalizer import ArrayNormalizer
from clearvla.mainline.model.component_contracts import ComponentSelection
from clearvla.mainline.runtime.checkpoints import CHECKPOINT_SCHEMA
from clearvla.mainline.runtime.deployment import build_deployment_abi, canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


class _TinyModel(torch.nn.Linear):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__(2, 1)
        self.selection = ComponentSelection.from_config(config)

    def set_training_step(self, step: int) -> None:
        self.step = step


def _deployment_payload(tmp_path: Path):
    from tests.test_libero_deployment_contract import _config, _profile

    config = _config()
    language = tmp_path / "goal.pt"
    torch.save(
        {
            "last_hidden_state": torch.zeros(1, 2, 4096),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
        },
        language,
    )
    normalizer = ArrayNormalizer.fit_zscore(
        [np.stack((np.zeros(7), np.ones(7))).astype(np.float32)]
    )
    normalizer_hash = canonical_sha256(normalizer.to_dict())
    dataset = DatasetIdentity(
        raw_root="synthetic",
        hdf5_glob="*.hdf5",
        inventory_sha256="a" * 64,
        state_normalizer_sha256=normalizer_hash,
        action_normalizer_sha256=normalizer_hash,
        decoded_cache_identity="b" * 64,
        dino_cache_identity="c" * 64,
    )
    identity = build_checkpoint_identity(
        config,
        repo_root=ROOT,
        dataset=dataset,
        language=ArtifactIdentity.from_file("goal", language),
        commit="a" * 40,
    )
    abi = build_deployment_abi(
        config,
        identity,
        action_normalizer=normalizer,
        state_normalizer=normalizer,
        data_profile=_profile(),
        gripper_indices=(6,),
        goal_metadata={},
    )
    payload = dict(
        schema=CHECKPOINT_SCHEMA,
        config=config.as_dict(),
        identity=identity.as_dict(),
        component_selection=ComponentSelection.from_config(config).as_dict(),
        data_state=dict(
            deployment_abi=abi,
            action_normalizer=normalizer.to_dict(),
            state_normalizer=normalizer.to_dict(),
        ),
        model=_TinyModel(config).state_dict(),
        epoch=1,
        global_step=1,
    )
    return payload, language


@pytest.mark.parametrize("mutation", ["runtime_abi", "selection", "model_source"])
def test_deployment_rejects_incompatible_runtime_before_constructing_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    from clearvla.mainline.manifest import manifest_from_mapping
    from clearvla.simulation import checkpoint as runtime

    payload, language = _deployment_payload(tmp_path)
    if mutation == "runtime_abi":
        manifest = manifest_from_mapping(payload["identity"]["manifest"])
        manifest = replace(
            manifest, components=replace(manifest.components, runtime="incompatible_runtime_v999")
        )
        payload["identity"]["manifest"] = manifest.as_dict()
        payload["identity"]["manifest_digest"] = manifest.digest()
        payload["data_state"]["deployment_abi"]["architecture_manifest"] = manifest.as_dict()
    elif mutation == "selection":
        payload["component_selection"]["world"] = "incompatible_world_v999"
    else:
        source = payload["identity"]["source"]
        for row in source["files"]:
            if row[0] == "clearvla/mainline/model/policy.py":
                row[1] = "f" * 64
                break
        else:
            raise AssertionError("fixture source closure lost policy.py")
        source["digest"] = canonical_sha256(source["files"])
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(payload, checkpoint)

    def should_not_construct(config):
        raise AssertionError("incompatible identity reached model construction")

    monkeypatch.setattr(runtime, "ClearVLAMainlinePolicy", should_not_construct)
    with pytest.raises(ValueError):
        runtime.load_deployment_checkpoint(
            checkpoint, device=torch.device("cpu"), t5_condition=language
        )


def test_deployment_accepts_same_runtime_with_relocated_artifacts_and_data_only_source_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from clearvla.simulation import checkpoint as runtime

    payload, language = _deployment_payload(tmp_path)
    payload["identity"]["language"]["path"] = "/unavailable/old/goal.pt"
    payload["config"]["data"]["raw_hdf5_root"] = "/unavailable/old/dataset"
    source = payload["identity"]["source"]
    for row in source["files"]:
        if row[0] == "clearvla/mainline/data/loading.py":
            row[1] = "f" * 64
            break
    else:
        raise AssertionError("fixture source closure lost training data loader")
    source["digest"] = canonical_sha256(source["files"])
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(payload, checkpoint)
    monkeypatch.setattr(runtime, "ClearVLAMainlinePolicy", _TinyModel)
    bundle = runtime.load_deployment_checkpoint(
        checkpoint, device=torch.device("cpu"), t5_condition=language
    )
    assert bundle.epoch == bundle.global_step == 1
    assert not bundle.model.training


@pytest.mark.parametrize("stale", [False, True])
def test_formal_causal_loader_checks_actual_rows_before_any_cache_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stale: bool,
) -> None:
    from clearvla.benchmarks.config import build_benchmark_config
    from clearvla.benchmarks.libero import convert_libero
    from clearvla.benchmarks.libero_terminal import convert_libero_causal_prefix
    from clearvla.mainline.data import loading
    from tests.test_libero_terminal import _FakeEnv, _quat_to_axisangle, _write_bddl, _write_raw

    _, instruction = _write_raw(tmp_path / "raw")
    bddl = _write_bddl(tmp_path / "bddl", instruction)
    convert_libero(
        tmp_path / "raw", tmp_path / "legacy", suites=("libero_spatial",), evaluator_bddl_root=bddl
    )
    convert_libero_causal_prefix(
        tmp_path / "legacy",
        tmp_path / "prefix",
        raw_source=tmp_path / "raw",
        evaluator_bddl_root=bddl,
        env_factory=_FakeEnv,
        quat_to_axisangle=_quat_to_axisangle,
    )
    config = config_from_mapping(
        build_benchmark_config(
            ROOT / "configs/mainline/object_intent_dynamics_323.json",
            tmp_path / "config.json",
            hdf5_root=tmp_path / "prefix",
            cache_root=tmp_path / "cache",
            language_bank=tmp_path / "language.pt",
            run_root=tmp_path / "run",
            gripper_event_threshold=0.1,
        )
    )
    if stale:
        episode = sorted((tmp_path / "prefix").glob("*.hdf5"))[0]
        with h5py.File(episode, "r+") as stream:
            # Same shape and valid bounds, but the model would see post-action states.
            stream["state"][1:] = stream["normalizer_reference_state"][1:]
            stream.attrs["observation_alignment"] = "post_action_v1"

    class ReachedCache(RuntimeError):
        pass

    def cache_boundary(*args, **kwargs):
        raise ReachedCache("data admitted to cache construction")

    monkeypatch.setattr(loading, "DecodedImageStore", cache_boundary)
    if stale:
        with pytest.raises(ValueError, match="causal|shift|alignment"):
            loading.load_mainline_data(config)
    else:
        with pytest.raises(ReachedCache):
            loading.load_mainline_data(config)
