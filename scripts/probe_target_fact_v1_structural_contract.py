"""Exercise TargetFact's target/scene role separation on deterministic CPU cases.

This probe is intentionally structural.  It supplies non-zero synthetic W
effects so the scene-consequence path can be tested independently of the
fresh model's exact-zero W heads, and it keeps target and scene outputs named
through the consequence/P3 boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearvla.mainline import config as config_module  # noqa: E402
from clearvla.mainline.checkpoint import (  # noqa: E402
    ArtifactIdentity,
    DatasetIdentity,
    SourceSnapshot,
    build_checkpoint_identity,
)
from clearvla.mainline.model.compiler import (  # noqa: E402
    ObjectFutureEffectReader,
    ObjectPolicyPlanCompiler,
    ZeroPreservingObjectConsequence,
)
from clearvla.mainline.model.component_contracts import (  # noqa: E402
    ComponentSelection,
)
from clearvla.mainline.model.intent import CoarseActionIntent  # noqa: E402
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy  # noqa: E402
from clearvla.mainline.model.types import (  # noqa: E402
    ActionIntentDock,
    FutureObjectDynamics,
    PhysicalActionCondition,
    PolicyIntentDock,
)
from clearvla.mainline.runtime.checkpoints import (  # noqa: E402
    CHECKPOINT_SCHEMA,
    P2_SHARED_TARGET_PRIOR_V1_NEW_STATE_KEY,
    TARGET_ACTION_SYSTEM_V2_MIGRATION,
    TARGET_ACTION_SYSTEM_V2_NEW_STATE_KEYS,
    TARGET_ACTION_SYSTEM_V2_SOURCE_PATHS,
    TARGET_FACT_V1_NEW_STATE_KEYS,
    TARGET_FACT_V1_RETIRED_STATE_KEYS,
    load_checkpoint_for_initialization,
)
from clearvla.tools.mainline_equivalence import (  # noqa: E402
    build_reduced_equivalence_config,
)

MODE = "target_action_bottleneck_v1"


def _source_digest(rows: tuple[tuple[str, str], ...]) -> str:
    payload = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _max_delta(left: Tensor, right: Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().amax())


def _rms(value: Tensor | None) -> float:
    if value is None:
        return 0.0
    return float(value.detach().float().square().mean().sqrt())


def _coarse_contract(seed: int) -> dict[str, object]:
    torch.manual_seed(seed)
    batch, objects, hidden = 1, 4, 16
    coarse = CoarseActionIntent(
        hidden=hidden,
        action_dim=7,
        heads=4,
        horizon=24,
        action_condition_mode="sequence_prefix_v1",
        target_fact_mode=True,
    )
    memory = torch.randn(batch, objects, hidden)
    validity = torch.ones(batch, objects, 1, dtype=torch.float32)
    posterior = torch.tensor([[0.55, 0.25, 0.15, 0.05]], dtype=torch.float32)
    support = torch.ones(batch, objects, dtype=torch.bool)
    target_summary = torch.einsum("bk,bkh->bh", posterior, memory)
    base_dock = ActionIntentDock(
        public_interval_carrier=torch.randn(batch, 4, hidden),
        history_memory=torch.randn(batch, 5, hidden),
        public_object_memory=memory,
        public_object_validity=validity,
        target_summary=target_summary,
        scene_context=torch.zeros(batch, hidden),
        target_posterior=posterior,
        target_support=support,
    )
    first = coarse(base_dock, collect_diagnostics=True)
    operation_changed_dock = replace(
        base_dock,
        public_interval_carrier=base_dock.public_interval_carrier + 3.0,
        history_memory=base_dock.history_memory - 2.0,
    )
    operation_changed = coarse(operation_changed_dock, collect_diagnostics=True)
    target_changed_dock = replace(
        base_dock,
        target_summary=target_summary * -1.7,
    )
    target_changed = coarse(target_changed_dock, collect_diagnostics=True)
    raw_k_changed_dock = replace(
        base_dock,
        public_object_memory=memory * -7.0 + 11.0,
        target_posterior=torch.tensor(
            [[0.05, 0.15, 0.25, 0.55]], dtype=torch.float32
        ),
    )
    raw_k_changed = coarse(raw_k_changed_dock, collect_diagnostics=True)
    permutation = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    permuted_dock = replace(
        base_dock,
        public_object_memory=memory[:, permutation],
        public_object_validity=validity[:, permutation],
        target_posterior=posterior[:, permutation],
        target_support=support[:, permutation],
        target_summary=torch.einsum(
            "bk,bkh->bh", posterior[:, permutation], memory[:, permutation]
        ),
    )
    permuted = coarse(permuted_dock, collect_diagnostics=True)

    # The coarse A0 producer must consume the already-resolved physical target
    # fact, not reopen raw K or consume p as a second hidden target selector.
    memory_vjp = memory.detach().clone().requires_grad_(True)
    posterior_vjp = posterior.detach().clone().requires_grad_(True)
    target_vjp = target_summary.detach().clone().requires_grad_(True)
    vjp_dock = replace(
        base_dock,
        public_object_memory=memory_vjp,
        target_summary=target_vjp,
        target_posterior=posterior_vjp,
    )
    coarse(vjp_dock, collect_diagnostics=False).action_prediction.square().sum().backward()

    checks = {
        "coarse_registers_no_raw_k_reader": coarse.object_read is None,
        "coarse_emits_full_24_row_a0": tuple(first.action_prediction.shape) == (1, 24, 7),
        "raw_k_and_p_are_inert_after_target_fact_materialization": (
            _max_delta(first.action_prediction, raw_k_changed.action_prediction) == 0.0
        ),
        "coarse_action_uses_operation_context": (
            _max_delta(first.action_prediction, operation_changed.action_prediction) > 0.0
        ),
        "coarse_action_uses_target_physical_fact": (
            _max_delta(first.action_prediction, target_changed.action_prediction) > 0.0
        ),
        "target_fact_has_direct_a0_vjp": _rms(target_vjp.grad) > 0.0,
        "raw_k_has_no_direct_a0_vjp": memory_vjp.grad is None,
        "p_has_no_second_direct_a0_vjp": posterior_vjp.grad is None,
        "joint_k_permutation_preserves_coarse_action": (
            _max_delta(first.action_prediction, permuted.action_prediction) <= 1e-6
        ),
    }
    return {
        "metrics": {
            "raw_k_p_action_delta_max": _max_delta(
                first.action_prediction, raw_k_changed.action_prediction
            ),
            "coarse_operation_action_delta_max": _max_delta(
                first.action_prediction, operation_changed.action_prediction
            ),
            "coarse_target_fact_action_delta_max": _max_delta(
                first.action_prediction, target_changed.action_prediction
            ),
            "target_fact_gradient_rms": _rms(target_vjp.grad),
            "raw_k_gradient_rms": _rms(memory_vjp.grad),
            "direct_p_gradient_rms": _rms(posterior_vjp.grad),
        },
        "checks": checks,
    }


def _dynamics_fixture(seed: int) -> tuple[FutureObjectDynamics, PolicyIntentDock, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    batch, intervals, objects, cameras = 1, 4, 3, 2
    content, hidden, route, horizon, basis = 6, 8, 3, 2, 1
    semantic = torch.randn(
        batch, intervals, objects, content, generator=generator
    )
    transport = 0.2 * torch.randn(
        batch, intervals, objects, cameras, 2, generator=generator
    )
    current = torch.randn(batch, objects, content, generator=generator)
    camera_coordinates = torch.tanh(
        torch.randn(batch, objects, cameras, 2, generator=generator)
    )
    dynamics = FutureObjectDynamics(
        current_reference=current,
        successor_content=current[:, None] + semantic,
        semantic_delta=semantic,
        transport_mean=transport,
        transport_covariance=torch.zeros(
            batch, intervals, objects, cameras, 3, dtype=torch.float32
        ),
        chart_availability=torch.ones(batch, objects, 1, dtype=torch.float32),
        log_chart_availability=torch.zeros(batch, objects, 1, dtype=torch.float32),
        camera_coordinates=camera_coordinates,
        camera_chart_availability=torch.ones(
            batch, objects, cameras, 1, dtype=torch.float32
        ),
        log_camera_chart_availability=torch.zeros(
            batch, objects, cameras, 1, dtype=torch.float32
        ),
    )
    posterior = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    support = torch.ones(batch, objects, dtype=torch.bool)
    intent = PolicyIntentDock(
        interval_key=torch.randn(batch, intervals, hidden, generator=generator),
        temporal_control=torch.randn(batch, horizon, hidden, generator=generator),
        state_change_evidence=torch.randn(batch, hidden, generator=generator),
        target_object_address_logit=torch.zeros(
            batch, intervals, objects, dtype=torch.float32
        ),
        typed_common_value=torch.randn(
            batch, objects, 3, route, generator=generator
        ),
        typed_interval_residual_value=torch.randn(
            batch, intervals, objects, 3, route, generator=generator
        ),
        target_posterior=posterior,
        target_support=support,
        target_log_prior=torch.zeros_like(posterior),
        target_summary=torch.randn(batch, hidden, generator=generator),
    )
    action_query = torch.randn(
        batch, horizon, basis, hidden, generator=generator
    )
    return dynamics, intent, action_query


def _permute_intent(intent: PolicyIntentDock, permutation: Tensor) -> PolicyIntentDock:
    return replace(
        intent,
        target_object_address_logit=intent.target_object_address_logit[
            :, :, permutation
        ],
        typed_common_value=intent.typed_common_value[:, permutation],
        typed_interval_residual_value=intent.typed_interval_residual_value[
            :, :, permutation
        ],
        target_posterior=intent.target_posterior[:, permutation],  # type: ignore[index]
        target_support=intent.target_support[:, permutation],  # type: ignore[index]
        target_log_prior=intent.target_log_prior[:, permutation],  # type: ignore[index]
    )


def _p2_contract(seed: int) -> dict[str, object]:
    dynamics, intent, action_query = _dynamics_fixture(seed)
    generator = torch.Generator().manual_seed(seed + 17)
    interval_action = torch.randn(1, 4, 7, generator=generator)
    action_condition = PhysicalActionCondition.from_interval_action(
        interval_action,
        torch.zeros(1, 7),
    )
    action_proposal = torch.randn(1, 2, 7, generator=generator)
    reader = ObjectFutureEffectReader(
        hidden=8,
        content_dim=6,
        route_dim=3,
        spatial_intent_mode=MODE,
    )
    reader.train()
    selected, spatial_metrics = reader.spatial_select(
        action_query,
        dynamics,
        intent,
        action_condition=action_condition,
        collect_diagnostics=True,
    )
    expected_scene_semantic = 0.5 * (
        dynamics.semantic_delta[:, :, 1] + dynamics.semantic_delta[:, :, 2]
    )
    target_raw_error = _max_delta(
        selected.semantic_value[:, 0, 0], dynamics.semantic_delta[:, :, 0]
    )
    scene_raw_error = _max_delta(
        selected.semantic_scene_value[:, 0, 0], expected_scene_semantic
    )

    zero_bundle, _ = reader(
        action_query,
        dynamics,
        intent,
        action_condition=action_condition,
        collect_diagnostics=False,
    )
    reader.zero_grad(set_to_none=True)
    zero_bundle.scene.combined().sum().backward()
    scene_semantic_weight_grad = _rms(reader.semantic_scene_value.weight.grad)  # type: ignore[union-attr]
    scene_geometry_weight_grad = _rms(reader.transport_scene_value.weight.grad)  # type: ignore[union-attr]

    modified = replace(
        dynamics,
        successor_content=(
            dynamics.successor_content
            + torch.tensor([0.0, 2.0, -1.0])[None, None, :, None]
        ),
        semantic_delta=(
            dynamics.semantic_delta
            + torch.tensor([0.0, 2.0, -1.0])[None, None, :, None]
        ),
        transport_mean=(
            dynamics.transport_mean
            + torch.tensor([0.0, 0.4, -0.25])[None, None, :, None, None]
        ),
    )
    with torch.no_grad():
        reader.semantic_scene_value.weight.fill_(0.075)  # type: ignore[union-attr]
        reader.transport_scene_value.weight.fill_(0.125)  # type: ignore[union-attr]
    original_bundle, _ = reader(
        action_query,
        dynamics,
        intent,
        action_condition=action_condition,
        collect_diagnostics=False,
    )
    modified_bundle, _ = reader(
        action_query,
        modified,
        intent,
        action_condition=action_condition,
        collect_diagnostics=False,
    )

    permutation = torch.tensor([2, 0, 1], dtype=torch.long)
    permuted_bundle, _ = reader(
        action_query,
        dynamics.permute(permutation),
        _permute_intent(intent, permutation),
        action_condition=action_condition,
        collect_diagnostics=False,
    )

    unavailable_camera = dynamics.camera_chart_availability.clone()
    unavailable_camera[:, 0] = 0.0
    camera_missing = replace(
        dynamics,
        camera_chart_availability=unavailable_camera,
    )
    camera_missing_bundle, _ = reader(
        action_query,
        camera_missing,
        intent,
        action_condition=action_condition,
        collect_diagnostics=False,
    )

    consequence = ZeroPreservingObjectConsequence(hidden=8)
    factual = torch.randn_like(original_bundle.target.semantic)
    original_consequence, _ = consequence(
        factual_base=factual,
        effect=original_bundle,
        collect_diagnostics=False,
    )
    modified_consequence, _ = consequence(
        factual_base=factual,
        effect=modified_bundle,
        collect_diagnostics=False,
    )
    plan_compiler = ObjectPolicyPlanCompiler(
        hidden=8,
        horizon=2,
        basis=1,
        action_dim=7,
        target_action_bottleneck=True,
    )
    p1_residual = torch.randn_like(factual)
    original_plan, _ = plan_compiler(
        p1_policy_residual=p1_residual,
        consequence=original_consequence,
        intent=intent,
        action_query=action_query,
        action_proposal=action_proposal,
        collect_diagnostics=False,
    )
    modified_plan, _ = plan_compiler(
        p1_policy_residual=p1_residual,
        consequence=modified_consequence,
        intent=intent,
        action_query=action_query,
        action_proposal=action_proposal,
        collect_diagnostics=False,
    )

    supported_nan_intent = replace(
        intent,
        typed_common_value=intent.typed_common_value.clone(),
    )
    supported_nan_intent.typed_common_value[:, 0] = torch.nan
    supported_nan_rejected = False
    try:
        reader.spatial_select(
            action_query,
            dynamics,
            supported_nan_intent,
            action_condition=action_condition,
            collect_diagnostics=False,
        )
    except ValueError:
        supported_nan_rejected = True

    checks = {
        "one_hot_target_reads_only_target_semantics": target_raw_error <= 1e-6,
        "one_hot_scene_is_the_other_object_expectation": scene_raw_error <= 1e-6,
        "scene_projection_starts_exact_zero": (
            int(torch.count_nonzero(zero_bundle.scene.semantic).item()) == 0
            and int(torch.count_nonzero(zero_bundle.scene.geometry).item()) == 0
        ),
        "ordinary_scene_value_projection_receives_gradient": (
            scene_semantic_weight_grad > 0.0 and scene_geometry_weight_grad > 0.0
        ),
        "non_target_w_does_not_rewrite_target_effect": (
            _max_delta(
                original_bundle.target.semantic,
                modified_bundle.target.semantic,
            )
            <= 1e-6
            and _max_delta(
                original_bundle.target.geometry,
                modified_bundle.target.geometry,
            )
            <= 1e-6
        ),
        "non_target_w_changes_the_scene_effect": (
            _max_delta(
                original_bundle.scene.semantic,
                modified_bundle.scene.semantic,
            )
            > 1e-6
            and _max_delta(
                original_bundle.scene.geometry,
                modified_bundle.scene.geometry,
            )
            > 1e-6
        ),
        "target_and_scene_each_preserve_k_permutation": (
            _max_delta(original_bundle.target.semantic, permuted_bundle.target.semantic)
            <= 1e-6
            and _max_delta(
                original_bundle.target.geometry, permuted_bundle.target.geometry
            )
            <= 1e-6
            and _max_delta(original_bundle.scene.semantic, permuted_bundle.scene.semantic)
            <= 1e-6
            and _max_delta(original_bundle.scene.geometry, permuted_bundle.scene.geometry)
            <= 1e-6
        ),
        "scene_geometry_survives_missing_target_camera": (
            _rms(camera_missing_bundle.target.geometry) == 0.0
            and _rms(camera_missing_bundle.scene.geometry) > 0.0
        ),
        "scene_stays_separate_through_consequence_and_plan": (
            _max_delta(
                original_consequence.protected_consequence,
                modified_consequence.protected_consequence,
            )
            <= 1e-6
            and _max_delta(original_plan.protected_base, modified_plan.protected_base)
            <= 1e-6
            and _max_delta(original_plan.temporal, modified_plan.temporal) <= 1e-6
            and _max_delta(original_plan.state_change, modified_plan.state_change)
            <= 1e-6
            and original_plan.scene is not None
            and modified_plan.scene is not None
            and _max_delta(original_plan.scene, modified_plan.scene) > 1e-6
        ),
        "supported_typed_nan_is_rejected_without_diagnostics": supported_nan_rejected,
        "target_geometry_k_marginal_is_exact": (
            float(spatial_metrics["object_p2_target_geometry_k_marginal_error"])
            <= 1e-6
        ),
        "scene_geometry_k_marginal_is_exact": (
            float(spatial_metrics["object_p2_scene_geometry_k_marginal_error"])
            <= 1e-6
        ),
    }
    return {
        "metrics": {
            "target_raw_semantic_error_max": target_raw_error,
            "scene_raw_semantic_error_max": scene_raw_error,
            "scene_semantic_projection_gradient_rms": scene_semantic_weight_grad,
            "scene_geometry_projection_gradient_rms": scene_geometry_weight_grad,
            "non_target_to_target_semantic_delta_max": _max_delta(
                original_bundle.target.semantic, modified_bundle.target.semantic
            ),
            "non_target_to_target_geometry_delta_max": _max_delta(
                original_bundle.target.geometry, modified_bundle.target.geometry
            ),
            "non_target_to_scene_semantic_delta_max": _max_delta(
                original_bundle.scene.semantic, modified_bundle.scene.semantic
            ),
            "non_target_to_scene_geometry_delta_max": _max_delta(
                original_bundle.scene.geometry, modified_bundle.scene.geometry
            ),
        },
        "checks": checks,
    }


def _system_v2_checkpoint_migration_contract(
    config: config_module.ExperimentConfig,
) -> dict[str, object]:
    """Exercise the real initialization loader, not only state-key arithmetic."""

    digest = "a" * 64
    dataset = DatasetIdentity(
        raw_root="synthetic://target-action-system-v2",
        hdf5_glob="*.hdf5",
        inventory_sha256=digest,
        state_normalizer_sha256="b" * 64,
        action_normalizer_sha256="c" * 64,
        decoded_cache_identity="d" * 64,
        dino_cache_identity="e" * 64,
    )
    language = ArtifactIdentity(
        logical_name="synthetic-language",
        path="synthetic://language.pt",
        size_bytes=1,
        sha256="f" * 64,
    )
    current_identity = build_checkpoint_identity(
        config,
        repo_root=REPO_ROOT,
        dataset=dataset,
        language=language,
        commit="1" * 40,
    )
    source_rows = dict(current_identity.source.files)
    changed_path = "clearvla/mainline/model/transition.py"
    if changed_path not in source_rows:
        raise RuntimeError(f"active source snapshot omits {changed_path}")
    source_rows[changed_path] = (
        "2" * 64 if source_rows[changed_path] != "2" * 64 else "3" * 64
    )
    saved_rows = tuple(sorted(source_rows.items()))
    saved_source = SourceSnapshot(
        files=saved_rows,
        digest=_source_digest(saved_rows),
    )
    saved_source.validate()
    saved_identity = replace(current_identity, source=saved_source)
    saved_identity.validate()

    torch.manual_seed(74_001)
    source_model = ClearVLAMainlinePolicy(config)
    source_state = {
        name: value.detach().clone()
        for name, value in source_model.state_dict().items()
        if name not in TARGET_ACTION_SYSTEM_V2_NEW_STATE_KEYS
    }
    retained_name = next(
        name
        for name, value in source_state.items()
        if value.is_floating_point() and value.numel() > 0
    )
    source_state[retained_name] = source_state[retained_name] + 0.125

    torch.manual_seed(74_002)
    target_model = ClearVLAMainlinePolicy(config)
    target_before = {
        name: target_model.state_dict()[name].detach().clone()
        for name in TARGET_ACTION_SYSTEM_V2_NEW_STATE_KEYS
    }
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "identity": saved_identity.as_dict(),
        "config": config.as_dict(),
        "component_selection": ComponentSelection.from_config(config).as_dict(),
        "model": source_state,
        "epoch": 9,
        "global_step": 123,
        "best_metric": 0.75,
    }

    def rejected(
        checkpoint: Path,
        *,
        candidate: ClearVLAMainlinePolicy,
        expected: str,
    ) -> bool:
        try:
            load_checkpoint_for_initialization(
                checkpoint,
                model=candidate,
                config=config,
                identity=current_identity,
                model_contract_migration=TARGET_ACTION_SYSTEM_V2_MIGRATION,
            )
        except ValueError as error:
            return expected in str(error)
        return False

    with tempfile.TemporaryDirectory(
        prefix="clearvla-target-action-system-v2-"
    ) as directory:
        checkpoint = Path(directory) / "prior-target-action.pt"
        torch.save(payload, checkpoint)
        state = load_checkpoint_for_initialization(
            checkpoint,
            model=target_model,
            config=config,
            identity=current_identity,
            model_contract_migration=TARGET_ACTION_SYSTEM_V2_MIGRATION,
        )
        extra_missing_payload = dict(payload)
        extra_missing_payload["model"] = {
            name: value
            for name, value in source_state.items()
            if name != retained_name
        }
        extra_missing_checkpoint = Path(directory) / "extra-missing.pt"
        torch.save(extra_missing_payload, extra_missing_checkpoint)
        torch.manual_seed(74_003)
        extra_missing_rejected = rejected(
            extra_missing_checkpoint,
            candidate=ClearVLAMainlinePolicy(config),
            expected="must add exactly its narrow CT readout and lift",
        )

        disallowed_path = "clearvla/mainline/runtime/sampling.py"
        disallowed_rows = dict(saved_rows)
        if disallowed_path not in disallowed_rows:
            raise RuntimeError(f"active source snapshot omits {disallowed_path}")
        disallowed_rows[disallowed_path] = (
            "4" * 64
            if disallowed_rows[disallowed_path] != "4" * 64
            else "5" * 64
        )
        disallowed_rows_tuple = tuple(sorted(disallowed_rows.items()))
        disallowed_identity = replace(
            saved_identity,
            source=SourceSnapshot(
                files=disallowed_rows_tuple,
                digest=_source_digest(disallowed_rows_tuple),
            ),
        )
        disallowed_identity.validate()
        disallowed_payload = dict(payload)
        disallowed_payload["identity"] = disallowed_identity.as_dict()
        disallowed_checkpoint = Path(directory) / "disallowed-source.pt"
        torch.save(disallowed_payload, disallowed_checkpoint)
        torch.manual_seed(74_004)
        disallowed_source_rejected = rejected(
            disallowed_checkpoint,
            candidate=ClearVLAMainlinePolicy(config),
            expected="source drift escapes the allow-list",
        )

        torch.manual_seed(74_005)
        nonzero_head_model = ClearVLAMainlinePolicy(config)
        if nonzero_head_model.transition.physical_delta_head is None:
            raise RuntimeError("target-action model has no physical CT readout")
        with torch.no_grad():
            nonzero_head_model.transition.physical_delta_head.weight.fill_(1.0e-3)
        nonzero_head_rejected = rejected(
            checkpoint,
            candidate=nonzero_head_model,
            expected="requires exact-zero physical readout",
        )
    target_after = target_model.state_dict()
    checks = {
        "real_loader_accepts_prior_target_checkpoint": (
            state.model_contract_migration == TARGET_ACTION_SYSTEM_V2_MIGRATION
        ),
        "real_loader_reports_only_declared_source_drift": (
            state.changed_source_files == (changed_path,)
        ),
        "real_loader_restores_retained_state": torch.equal(
            target_after[retained_name], source_state[retained_name]
        ),
        "real_loader_preserves_current_new_owner_initialization": all(
            torch.equal(target_after[name], target_before[name])
            for name in TARGET_ACTION_SYSTEM_V2_NEW_STATE_KEYS
        ),
        "real_loader_restores_scalar_provenance": (
            state.epoch == 9
            and state.global_step == 123
            and state.best_metric == 0.75
        ),
        "real_loader_rejects_one_extra_missing_owner": extra_missing_rejected,
        "real_loader_rejects_source_outside_allowlist": disallowed_source_rejected,
        "real_loader_rejects_nonzero_new_physical_head": nonzero_head_rejected,
    }
    return {
        "retained_probe_parameter": retained_name,
        "changed_source_files": list(state.changed_source_files),
        "checks": checks,
    }


def _migration_contract() -> dict[str, object]:
    base = build_reduced_equivalence_config({"config": config_module})
    results: dict[str, object] = {}
    all_checks: dict[str, bool] = {}
    for action_mode in ("interval_mean_v1", "sequence_prefix_v1"):
        target_config = replace(
            base,
            top=replace(
                base.top,
                p2_spatial_intent_mode=MODE,
                world_action_condition_mode=action_mode,
            ),
        )
        target_state = ClearVLAMainlinePolicy(target_config).state_dict()
        for source_mode in ("post_pool_only", "shared_target_prior_v1"):
            source_config = replace(
                base,
                top=replace(
                    base.top,
                    p2_spatial_intent_mode=source_mode,
                    world_action_condition_mode=action_mode,
                ),
            )
            source_state = ClearVLAMainlinePolicy(source_config).state_dict()
            new = set(target_state) - set(source_state)
            retired = set(source_state) - set(target_state)
            expected_new = set(TARGET_FACT_V1_NEW_STATE_KEYS)
            if action_mode == "sequence_prefix_v1":
                expected_new.remove("intent.coarse_action.sequence_row_offset")
            expected_retired = set(TARGET_FACT_V1_RETIRED_STATE_KEYS)
            if source_mode == "post_pool_only":
                expected_retired.remove(P2_SHARED_TARGET_PRIOR_V1_NEW_STATE_KEY)
            checks = {
                "new_state_exact": new == expected_new,
                "retired_state_exact": retired == expected_retired,
                "coarse_raw_k_reader_is_retired": all(
                    name not in target_state
                    for name in (
                        "intent.coarse_action.object_read.attention.in_proj_weight",
                        "intent.coarse_action.object_read.attention.out_proj.weight",
                        "intent.coarse_action.object_read.ffn.1.weight",
                        "intent.coarse_action.object_read.ffn.3.weight",
                    )
                ),
                "new_scene_effect_projections_are_exact_zero": all(
                    int(torch.count_nonzero(target_state[name]).item()) == 0
                    for name in (
                        "policy_compiler.effect_reader.semantic_scene_value.weight",
                        "policy_compiler.effect_reader.transport_scene_value.weight",
                    )
                ),
                "new_target_selector_and_row_offsets_are_exact_zero": (
                    int(
                        torch.count_nonzero(
                            target_state["intent.organizer.target_score.weight"]
                        ).item()
                    )
                    == 0
                    and (
                        action_mode == "sequence_prefix_v1"
                        or int(
                            torch.count_nonzero(
                                target_state[
                                    "intent.coarse_action.sequence_row_offset"
                                ]
                            ).item()
                        )
                        == 0
                    )
                ),
                "sequence_source_preserves_existing_row_offset": (
                    action_mode != "sequence_prefix_v1"
                    or "intent.coarse_action.sequence_row_offset" not in new
                ),
            }
            key = f"{action_mode}:{source_mode}"
            all_checks.update(
                {f"{key}:{name}": value for name, value in checks.items()}
            )
            results[key] = {
                "new_count": len(new),
                "retired_count": len(retired),
                "checks": checks,
            }
    current_target = ClearVLAMainlinePolicy(
        replace(base, top=replace(base.top, p2_spatial_intent_mode=MODE))
    ).state_dict()
    prior_target = {
        name: value
        for name, value in current_target.items()
        if name not in TARGET_ACTION_SYSTEM_V2_NEW_STATE_KEYS
    }
    physical_ct_new = set(current_target) - set(prior_target)
    physical_ct_checks = {
        "new_state_exact": physical_ct_new
        == set(TARGET_ACTION_SYSTEM_V2_NEW_STATE_KEYS),
        "physical_readout_is_exact_zero": int(
            torch.count_nonzero(
                current_target["transition.physical_delta_head.weight"]
            ).item()
        )
        == 0,
        "evidence_lift_is_finite": bool(
            torch.isfinite(
                current_target["execution_bottom.transition_delta_lift.weight"]
            ).all()
        ),
        "optimizer_owner_is_in_narrow_source_allowlist": (
            "clearvla/mainline/training/optimizer.py"
            in TARGET_ACTION_SYSTEM_V2_SOURCE_PATHS
        ),
    }
    all_checks.update(
        {
            f"target_action_system_v2:{name}": value
            for name, value in physical_ct_checks.items()
        }
    )
    results["target_action_system_v2"] = {
        "new_count": len(physical_ct_new),
        "retired_count": 0,
        "checks": physical_ct_checks,
    }
    checkpoint_migration = _system_v2_checkpoint_migration_contract(
        replace(base, top=replace(base.top, p2_spatial_intent_mode=MODE))
    )
    checkpoint_checks = cast(dict[str, bool], checkpoint_migration["checks"])
    all_checks.update(
        {
            f"target_action_system_v2_checkpoint:{name}": value
            for name, value in checkpoint_checks.items()
        }
    )
    results["target_action_system_v2_checkpoint"] = checkpoint_migration
    return {"sources": results, "checks": all_checks}


def run(*, seed: int) -> dict[str, object]:
    coarse = _coarse_contract(seed)
    p2 = _p2_contract(seed + 1)
    migration = _migration_contract()
    coarse_checks = cast(dict[str, bool], coarse["checks"])
    p2_checks = cast(dict[str, bool], p2["checks"])
    migration_checks = cast(dict[str, bool], migration["checks"])
    checks = {
        **{f"coarse:{name}": value for name, value in coarse_checks.items()},
        **{f"p2:{name}": value for name, value in p2_checks.items()},
        **{
            f"migration:{name}": value
            for name, value in migration_checks.items()
        },
    }
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"TargetFact structural contract failed: {failed}")
    return {
        "schema": "clearvla-target-fact-structural-contract-v1",
        "seed": int(seed),
        "mode": MODE,
        "coarse": coarse,
        "p2": p2,
        "migration": migration,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=73_801)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(seed=args.seed)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
