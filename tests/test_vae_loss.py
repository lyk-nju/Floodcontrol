import torch

from models.vae_wan_1d import BodyVAE
from utils.conditions.vae import VAEInput
from utils.training.vae.losses import VAELoss


def test_vae_loss_blocks_contacts_and_kl_warmup():
    model = BodyVAE(
        latent_dim=128, hidden_dim=32, encoder_layers=1, decoder_layers=1,
        allow_identity_statistics=True
    )
    body = torch.randn(2, 8, 265)
    body[..., 261:] = torch.randint(0, 2, body[..., 261:].shape).float()
    root = torch.zeros(2, 8, 5)
    root[..., 3] = 1
    inputs = VAEInput(body, root, torch.ones(2, 8, dtype=torch.bool))
    prediction = model(inputs)
    loss_fn = VAELoss(
        body_cont_mean=model.body_cont_mean,
        body_cont_std=model.body_cont_std,
        beta_kl=1e-4,
        kl_warmup_steps=100,
    )
    losses = loss_fn(inputs, prediction, global_step=50)
    assert set(("position", "rotation", "velocity", "contact", "skating", "kl", "total")) <= set(losses)
    assert torch.isfinite(losses["total"])
    assert torch.allclose(losses["kl_beta"], torch.tensor(5e-5))
    losses["total"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_optional_geometry_losses_are_independent():
    model = BodyVAE(
        latent_dim=8, hidden_dim=16, encoder_layers=1, decoder_layers=1,
        allow_identity_statistics=True
    )
    body = torch.randn(1, 8, 265)
    body[..., 261:] = 0
    root = torch.zeros(1, 8, 5)
    root[..., 3] = 1
    inputs = VAEInput(body, root, torch.ones(1, 8, dtype=torch.bool))
    prediction = model(inputs)
    loss_fn = VAELoss(
        body_cont_mean=model.body_cont_mean,
        body_cont_std=model.body_cont_std,
        lambda_geodesic=0.1,
        lambda_velocity_consistency=0.1,
    )
    losses = loss_fn(inputs, prediction)
    assert losses["geodesic"] > 0
    assert losses["velocity_consistency"] >= 0


def test_fk_loss_requires_versioned_skeleton():
    try:
        VAELoss(
            body_cont_mean=torch.zeros(261),
            body_cont_std=torch.ones(261),
            lambda_fk=1.0,
        )
    except ValueError as error:
        assert "versioned skeleton" in str(error)
    else:
        raise AssertionError("FK loss accepted missing skeleton")
