import pytest
import torch

from tests.vae_helpers import make_vae
from utils.training.vae.checkpoint import PHYSICAL_STATISTIC_BUFFERS, load_vae_checkpoint


MODEL_PARAMS = {
    "latent_dim": 4,
    "hidden_dim": 8,
    "encoder_layers": 1,
    "decoder_layers": 1,
    "kernel_size": 3,
    "dropout": 0.0,
}


def _checkpoint(model, path, *, ema=True, historical_latent_buffers=False):
    state = model.state_dict()
    if historical_latent_buffers:
        state = dict(state)
        state["latent_mean"] = torch.zeros(model.latent_dim)
        state["latent_std"] = torch.ones(model.latent_dim)
    payload = {"state_dict": state}
    if ema:
        payload["ema_state"] = {
            "shadow_params": [
                parameter.detach().clone() for parameter in model.parameters()
            ]
        }
    torch.save(payload, path)


def test_checkpoint_loader_constructs_frozen_ema_vae_with_physical_buffers(tmp_path):
    source = make_vae(**MODEL_PARAMS)
    checkpoint = tmp_path / "training.ckpt"
    _checkpoint(source, checkpoint, historical_latent_buffers=True)

    loaded = load_vae_checkpoint(checkpoint, model_params=MODEL_PARAMS)

    assert not loaded.training
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    for name in PHYSICAL_STATISTIC_BUFFERS:
        assert torch.equal(getattr(loaded, name), getattr(source, name))
    assert set(loaded.state_dict()) == set(source.state_dict())


def test_checkpoint_loader_requires_ema_by_default(tmp_path):
    model = make_vae(**MODEL_PARAMS)
    checkpoint = tmp_path / "raw.ckpt"
    _checkpoint(model, checkpoint, ema=False)
    with pytest.raises(ValueError, match="missing ema_state"):
        load_vae_checkpoint(checkpoint, model_params=MODEL_PARAMS)


def test_checkpoint_loader_rejects_external_statistics_paths(tmp_path):
    model = make_vae(**MODEL_PARAMS)
    checkpoint = tmp_path / "training.ckpt"
    _checkpoint(model, checkpoint)
    with pytest.raises(ValueError, match="architecture only"):
        load_vae_checkpoint(
            checkpoint,
            model_params={**MODEL_PARAMS, "motion_stats_path": "external.npz"},
        )
