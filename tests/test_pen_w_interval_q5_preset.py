from clearvla.mainline.config import load_config
from clearvla.mainline.manifest import ARCHITECTURE_MANIFEST
from clearvla.mainline.runtime.deployment import deployment_flow_schedule


def test_new_pen_candidate_keeps_q5_without_changing_training_contract():
    base = load_config("configs/mainline/object_intent_dynamics_323.json")
    selected = load_config("configs/mainline/object_intent_dynamics_323_pen_w_interval_q5.json")
    schedule = deployment_flow_schedule(selected)
    assert schedule.candidate_id == "Q5/Q5"
    assert schedule.fingerprint == "3080a10e487ebba47e873303ec05c8e9828efc033094bb194ac57f69b9a0a1e9"
    assert schedule.physical_nfe == 10
    assert schedule.endpoint_head_calls == 2
    assert ARCHITECTURE_MANIFEST.components.top.endswith("_typed_interval_qk_v1")
    assert selected.bottom == base.bottom
    assert selected.objectives == base.objectives
    assert selected.optimizer == base.optimizer
    assert base.runtime.deployment_flow_schedule is None
    left, right = base.as_dict(), selected.as_dict()
    left["data"].pop("output_dir")
    right["data"].pop("output_dir")
    right["runtime"].pop("deployment_flow_schedule")
    assert left == right
    assert not any("spine" in key for key in selected.as_dict()["bottom"])
