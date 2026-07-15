import pytest
import torch

from models.vae_wan_1d import BodyVAE
from tests.vae_helpers import make_vae, write_statistics


def make_model():
    return make_vae(
        latent_dim=128,
        hidden_dim=32,
        encoder_layers=2,
        decoder_layers=2,
    ).eval()


def make_local(batch=2, tokens=3):
    local = torch.randn(batch, tokens, 4, 4)
    return local, torch.ones_like(local, dtype=torch.bool)


def test_four_frame_shapes_and_deterministic_tokenize():
    model = make_model()
    body = torch.randn(2, 12, 265)
    mask = torch.ones(2, 12, dtype=torch.bool)
    posterior = model.encode(body, mask)
    assert posterior.mu.shape == (2, 3, 128)
    assert torch.equal(model.tokenize(body, mask), posterior.mu)
    local, valid = make_local()
    decoded = model.decode(posterior.mu, local, valid)
    assert decoded.continuous_body.shape == (2, 12, 261)
    assert decoded.contact_logits.shape == (2, 12, 4)


@pytest.mark.parametrize("kernel_size", [1, 2, 4])
def test_causal_vae_rejects_unsafe_kernel_sizes(kernel_size):
    with pytest.raises(ValueError, match="odd and at least three"):
        make_vae(
            latent_dim=8,
            hidden_dim=8,
            encoder_layers=1,
            decoder_layers=1,
            kernel_size=kernel_size,
        )


def test_encoder_window_matches_full_clip_after_complete_warmup():
    model = make_model()
    tokens = 16
    body = torch.randn(1, tokens * 4, 265)
    mask = torch.ones(1, tokens * 4, dtype=torch.bool)
    full = model.encode(body, mask)
    start, active_tokens = 10, 3
    context = model.encoder_context_tokens
    local = model.encode_window(
        body[:, (start - context) * 4 : (start + active_tokens) * 4],
        mask[:, (start - context) * 4 : (start + active_tokens) * 4],
        context_token_count=context,
    )
    assert torch.allclose(local.mu, full.mu[:, start : start + active_tokens], atol=1e-6)


def test_offline_and_explicit_state_decode_match():
    model = make_model()
    latent = torch.randn(2, 3, 128)
    local, valid = make_local()
    offline = model.decode(latent, local, valid)
    state = model.init_decoder_state(2, dtype=latent.dtype)
    initial = state.clone()
    chunks = []
    contacts = []
    for token in range(3):
        state, output = model.decode_step(
            latent[:, token : token + 1],
            local[:, token : token + 1],
            valid[:, token : token + 1],
            state,
        )
        chunks.append(output.continuous_body)
        contacts.append(output.contact_logits)
    assert torch.allclose(torch.cat(chunks, 1), offline.continuous_body, atol=1e-5)
    assert torch.allclose(torch.cat(contacts, 1), offline.contact_logits, atol=1e-5)
    assert all(torch.count_nonzero(cache) == 0 for pair in initial.caches for cache in pair)


def test_two_decoder_states_do_not_share_cache():
    model = make_model()
    first = model.init_decoder_state(1)
    second = model.init_decoder_state(1)
    latent = torch.randn(1, 1, 128)
    local, valid = make_local(batch=1, tokens=1)
    first, _ = model.decode_step(latent, local, valid, first)
    assert any(torch.count_nonzero(cache) > 0 for pair in first.caches for cache in pair)
    assert all(torch.count_nonzero(cache) == 0 for pair in second.caches for cache in pair)


def test_statistics_files_only_require_arrays(tmp_path):
    motion, latent = write_statistics(
        tmp_path, latent_dim=4, latent_mean=2.0, latent_std=3.0
    )
    model = BodyVAE(
        motion_stats_path=motion,
        latent_stats_path=latent,
        latent_dim=4,
        hidden_dim=8,
        encoder_layers=1,
        decoder_layers=1,
    )
    assert torch.equal(model.latent_mean, torch.full((4,), 2.0))
    assert torch.equal(model.latent_std, torch.full((4,), 3.0))


def test_tiny_deterministic_mu_path_can_overfit_two_clips():
    torch.manual_seed(7)
    model = make_vae(
        latent_dim=8,
        hidden_dim=16,
        encoder_layers=1,
        decoder_layers=1,
        with_latent_stats=False,
    ).train()
    body = torch.randn(2, 8, 265)
    local = torch.randn(2, 2, 4, 4)
    valid = torch.ones_like(local, dtype=torch.bool)
    mask = torch.ones(2, 8, dtype=torch.bool)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    def reconstruction():
        output = model.decode(model.encode(body, mask).mu, local, valid)
        return (output.continuous_body - body[..., :261]).square().mean()

    initial = reconstruction().detach()
    for _ in range(20):
        optimizer.zero_grad()
        loss = reconstruction()
        loss.backward()
        optimizer.step()
    assert reconstruction().detach() < initial
