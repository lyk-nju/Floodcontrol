import torch

from models.vae_wan_1d import BodyVAE
from utils.conditions.vae import BODY_DIM, VAEInput
from utils.motion_representation import (
    derive_patched_local_root,
    pack_body_motion,
    rotate_root_body_yaw,
    unpack_body_motion,
)


def make_body(batch=2, frames=8):
    return pack_body_motion(
        torch.randn(batch, frames, 21, 3),
        torch.randn(batch, frames, 22, 6),
        torch.randn(batch, frames, 22, 3),
        torch.randint(0, 2, (batch, frames, 4)).float(),
    )


def make_root(batch=2, frames=8):
    root = torch.zeros(batch, frames, 5)
    root[..., 0] = torch.arange(frames).float() / 20
    root[..., 3] = 1
    return root


def test_body265_pack_unpack_roundtrip():
    body = make_body()
    assert body.shape[-1] == BODY_DIM == 265
    parts = unpack_body_motion(body)
    rebuilt = pack_body_motion(*parts.values())
    assert torch.equal(rebuilt, body)


def test_contract_rejects_non_strict_frame_length():
    inputs = VAEInput(
        body_motion=torch.zeros(1, 5, 265),
        root_motion=torch.zeros(1, 5, 5),
        frame_valid_mask=torch.ones(1, 5, dtype=torch.bool),
    )
    try:
        inputs.validate()
    except ValueError as error:
        assert "divisible by four" in str(error)
    else:
        raise AssertionError("non-strict frame length was accepted")


def test_backward_local_root_and_cold_start_validity():
    root = make_root(batch=1)
    local, valid = derive_patched_local_root(root, None)
    assert local.shape == (1, 2, 4, 4)
    assert not valid[0, 0, 0, :3].any()
    assert valid[0, 0, 0, 3]
    assert torch.allclose(local[0, 0, 1:, 1], torch.ones(3), atol=1e-6)


def test_local_velocity_is_invariant_to_global_yaw_rotation():
    root = make_root(batch=1)
    body = make_body(batch=1)
    before, valid = derive_patched_local_root(root, None)
    rotated_root, _ = rotate_root_body_yaw(root, body, torch.tensor([0.7]))
    after, rotated_valid = derive_patched_local_root(rotated_root, None)
    assert torch.equal(valid, rotated_valid)
    assert torch.allclose(before[valid], after[valid], atol=1e-5)


def test_contacts_are_not_zscore_normalized():
    model = BodyVAE(
        latent_dim=8, hidden_dim=8, encoder_layers=1, decoder_layers=1,
        allow_identity_statistics=True
    )
    body = make_body(batch=1)
    normalized = model.normalize_body(body)
    assert torch.equal(normalized[..., 261:], body[..., 261:])
