import torch

from clearvla.mainline.config import load_config
from clearvla.mainline.model.calvin_object_binding import CalvinObjectBindingBridge
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy


def test_masked_pointer_has_exact_null_fallback_and_masks_invalid_slots():
    logits = torch.tensor([[1.0, 5.0, -2.0]])
    null = torch.tensor([-1.0])
    valid = torch.tensor([[True, False, True]])
    pointer = CalvinObjectBindingBridge._masked_pointer(logits, null, valid)
    assert pointer.shape == (1, 4)
    assert pointer[0, 1].item() == 0.0
    assert torch.allclose(pointer.sum(-1), torch.ones(1))

    none = CalvinObjectBindingBridge._masked_pointer(logits, null, torch.zeros_like(valid))
    assert torch.equal(none, torch.tensor([[0.0, 0.0, 0.0, 1.0]]))


def test_binding_supervised_loss_accepts_soft_correspondence_and_coverage():
    pointer = torch.tensor([[0.6, 0.2, 0.1, 0.1]], requires_grad=True)
    target = torch.tensor([[0.2, 0.6, 0.1, 0.1]])
    loss = CalvinObjectBindingBridge.supervised_loss(pointer, target, torch.tensor([[1.0]]))
    assert loss.item() > 0.0
    loss.backward()
    assert pointer.grad is not None and torch.isfinite(pointer.grad).all()


def test_calvin_config_constructs_binding_child(tmp_path):
    config = load_config("configs/mainline/calvin_object_binding_formal_v1.json")
    policy = ClearVLAMainlinePolicy(config)
    assert policy.intent.calvin_object_binding is not None
