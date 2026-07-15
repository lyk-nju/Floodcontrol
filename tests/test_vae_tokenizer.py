import pytest
import torch

from tests.vae_helpers import make_vae, write_statistics
from tools.compute_vae_latent_stats import LatentStatisticsAccumulator
from utils.training.vae.checkpoint import load_vae_checkpoint


def _checkpoint(model, path, *, ema=True):
    payload = {"state_dict": model.state_dict()}
    if ema:
        payload["ema_state"] = {
            "shadow_params": [parameter.detach().clone() for parameter in model.parameters()]
        }
    torch.save(payload, path)


def test_checkpoint_loader_uses_ema_and_preserves_latent_statistics(tmp_path):
    source = make_vae(
        latent_dim=4, hidden_dim=8, encoder_layers=1, decoder_layers=1
    )
    checkpoint = tmp_path / "training.ckpt"
    _checkpoint(source, checkpoint)
    motion, latent = write_statistics(
        tmp_path / "target", latent_dim=4, latent_mean=2.0, latent_std=3.0
    )
    target = type(source)(
        motion_stats_path=motion,
        latent_stats_path=latent,
        latent_dim=4,
        hidden_dim=8,
        encoder_layers=1,
        decoder_layers=1,
    )
    load_vae_checkpoint(target, checkpoint)
    assert torch.equal(target.latent_mean, torch.full((4,), 2.0))
    assert torch.equal(target.latent_std, torch.full((4,), 3.0))
    assert not target.training
    assert not any(parameter.requires_grad for parameter in target.parameters())
    # Loading is idempotent even after the first call freezes the model.
    load_vae_checkpoint(target, checkpoint)


def test_checkpoint_loader_requires_ema_by_default(tmp_path):
    model = make_vae(latent_dim=4, hidden_dim=8, encoder_layers=1, decoder_layers=1)
    checkpoint = tmp_path / "raw.ckpt"
    _checkpoint(model, checkpoint, ema=False)
    with pytest.raises(ValueError, match="missing ema_state"):
        load_vae_checkpoint(model, checkpoint)


def test_latent_statistics_fail_on_non_finite_values():
    accumulator = LatentStatisticsAccumulator(2)
    with pytest.raises(ValueError, match="sample/nonfinite"):
        accumulator.update(
            torch.tensor([[float("nan"), 0.0]]), sample_identity="sample/nonfinite"
        )
    accumulator.update(torch.tensor([[1.0, 2.0]]), sample_identity="sample/ok")
    accumulator.total[0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        accumulator.finish()
