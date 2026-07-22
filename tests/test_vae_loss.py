import torch

from models.vae_wan_1d import BodyVAE
from utils.conditions.vae import BodyPrediction, VAEPrediction, VAEInput, VAEPosterior
from tests.vae_helpers import make_vae
from utils.training.vae.losses import VAELoss


def test_vae_loss_blocks_contacts_and_kl_warmup():
    model = make_vae(
        latent_dim=128, hidden_dim=32, encoder_layers=1, decoder_layers=1,
    )
    body = torch.randn(2, 8, 259)
    body[..., 255:] = torch.randint(0, 2, body[..., 255:].shape).float()
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
    model = make_vae(
        latent_dim=8, hidden_dim=16, encoder_layers=1, decoder_layers=1,
    )
    body = torch.randn(1, 8, 259)
    body[..., 255:] = 0
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
            body_cont_mean=torch.zeros(255),
            body_cont_std=torch.ones(255),
            lambda_fk=1.0,
        )
    except ValueError as error:
        assert "versioned skeleton" in str(error)
    else:
        raise AssertionError("FK loss accepted missing skeleton")


def test_skating_uses_position_transitions_and_reaches_position_gradient():
    body = torch.zeros(1, 4, 259)
    body[..., 255] = 1.0
    root = torch.zeros(1, 4, 5)
    root[..., 3] = 1.0
    feature_valid = torch.ones_like(body, dtype=torch.bool)
    feature_valid[:, 0, :63] = False
    inputs = VAEInput(
        body,
        root,
        torch.ones(1, 4, dtype=torch.bool),
        body_feature_valid_mask=feature_valid,
    )
    continuous = torch.zeros(1, 4, 255, requires_grad=True)
    positions = continuous[..., :63].reshape(1, 4, 21, 3)
    with torch.no_grad():
        positions[:, :, 6, 0] = torch.arange(4) * 0.1
    prediction = VAEPrediction(
        BodyPrediction(continuous, torch.full((1, 4, 4), -100.0)),
        VAEPosterior(torch.zeros(1, 1, 2), torch.zeros(1, 1, 2)),
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1, 4, 4),
        torch.ones(1, 1, 4, 4, dtype=torch.bool),
    )
    losses = VAELoss(
        body_cont_mean=torch.zeros(255),
        body_cont_std=torch.ones(255),
    )(inputs, prediction)
    # Frame zero is excluded because a predicted preceding position is absent;
    # three transitions have one contacted foot moving at 2 m/s.
    assert torch.allclose(losses["skating"], torch.tensor(0.5))
    losses["skating"].backward()
    assert continuous.grad is not None
    assert continuous.grad[..., :63].abs().sum() > 0


def test_position_skating_ignores_independent_velocity_feature():
    body = torch.zeros(1, 4, 259)
    body[..., 255] = 1.0
    root = torch.zeros(1, 4, 5)
    root[..., 3] = 1.0
    inputs = VAEInput(
        body,
        root,
        torch.ones(1, 4, dtype=torch.bool),
        body_feature_valid_mask=torch.ones_like(body, dtype=torch.bool),
    )
    continuous = torch.zeros(1, 4, 255)
    continuous[..., 189:].reshape(1, 4, 22, 3)[:, :, 7, 0] = 100.0
    prediction = VAEPrediction(
        BodyPrediction(continuous, torch.zeros(1, 4, 4)),
        VAEPosterior(torch.zeros(1, 1, 2), torch.zeros(1, 1, 2)),
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1, 4, 4),
        torch.ones(1, 1, 4, 4, dtype=torch.bool),
    )
    losses = VAELoss(
        body_cont_mean=torch.zeros(255),
        body_cont_std=torch.ones(255),
    )(inputs, prediction)
    assert torch.equal(losses["skating"], torch.tensor(0.0))
