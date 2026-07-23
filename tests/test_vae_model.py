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
    body = torch.randn(2, 12, 259)
    mask = torch.ones(2, 12, dtype=torch.bool)
    posterior = model.encode(body, mask)
    assert posterior.mu.shape == (2, 3, 128)
    assert torch.equal(model.tokenize(body, mask), posterior.mu)
    local, valid = make_local()
    decoded = model.decode(posterior.mu, local, valid)
    assert decoded.continuous_body.shape == (2, 12, 255)
    assert decoded.contact_logits.shape == (2, 12, 4)


def test_encoder_ignores_feature_invalid_values():
    model = make_model()
    body = torch.randn(1, 8, 259)
    changed = body.clone()
    frame_mask = torch.ones(1, 8, dtype=torch.bool)
    feature_mask = torch.ones_like(body, dtype=torch.bool)
    feature_mask[:, :4, 189:259] = False
    changed[:, :4, 189:259] = 1000.0

    original = model.encode(body, frame_mask, feature_mask)
    modified = model.encode(changed, frame_mask, feature_mask)
    assert torch.equal(original.mu, modified.mu)
    assert torch.equal(original.logvar, modified.logvar)
    assert torch.equal(
        model.tokenize(body, frame_mask, feature_mask),
        original.mu,
    )


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


@pytest.mark.parametrize(
    ("start", "context"),
    [
        pytest.param(0, 0, id="cold-start"),
        pytest.param(3, 3, id="partial-context"),
        pytest.param(12, 8, id="full-context"),
    ],
)
def test_tokenize_window_matches_full_clip_for_real_context(start, context):
    model = make_model()
    assert model.encoder_context_tokens == 8
    tokens = 20
    body = torch.randn(1, tokens * 4, 259)
    mask = torch.ones(1, tokens * 4, dtype=torch.bool)
    full = model.encode(body, mask)
    active_tokens = 3
    local = model.tokenize_window(
        body[:, (start - context) * 4 : (start + active_tokens) * 4],
        mask[:, (start - context) * 4 : (start + active_tokens) * 4],
        context_token_count=torch.tensor([context], dtype=torch.long),
    )
    expected = full.mu[:, start : start + active_tokens]
    assert torch.allclose(local, expected, atol=1e-6)


def test_tokenize_window_rejects_partial_encoder_token():
    model = make_model()
    body = torch.zeros(1, 4, 259)
    mask = torch.tensor([[True, True, False, False]])
    with pytest.raises(ValueError, match="constant within each four-frame token"):
        model.tokenize_window(
            body,
            mask,
            context_token_count=torch.zeros(1, dtype=torch.long),
        )


def test_tokenize_window_mixed_batch_gathers_active_tokens_and_zeros_padding(tmp_path):
    motion_stats = write_statistics(tmp_path)
    model = BodyVAE(
        motion_stats_path=motion_stats,
        latent_dim=128,
        hidden_dim=32,
        encoder_layers=2,
        decoder_layers=2,
    ).eval()
    specs = (
        # start, real context, active tokens
        (0, 0, 3),
        (3, 3, 2),
        (12, model.encoder_context_tokens, 4),
    )
    full_body = torch.randn(len(specs), 20 * 4, 259)
    full_mask = torch.ones(len(specs), 20 * 4, dtype=torch.bool)
    full_feature_mask = torch.ones_like(full_body, dtype=torch.bool)
    full_posterior = model.encode(full_body, full_mask, full_feature_mask)

    window_tokens = [context + active for _, context, active in specs]
    max_window_tokens = max(window_tokens)
    body_with_context = torch.zeros(len(specs), max_window_tokens * 4, 259)
    window_mask = torch.zeros(len(specs), max_window_tokens * 4, dtype=torch.bool)
    window_feature_mask = torch.zeros_like(body_with_context, dtype=torch.bool)
    for batch_index, (start, context, active) in enumerate(specs):
        source = full_body[
            batch_index,
            (start - context) * 4 : (start + active) * 4,
        ]
        body_with_context[batch_index, : source.shape[0]] = source
        window_mask[batch_index, : source.shape[0]] = True
        window_feature_mask[batch_index, : source.shape[0]] = True

    context_count = torch.tensor([item[1] for item in specs], dtype=torch.long)
    raw_mu = model.tokenize_window(
        body_with_context,
        window_mask,
        context_count,
        body_feature_valid_mask=window_feature_mask,
    )

    max_active = max(item[2] for item in specs)
    assert raw_mu.shape == (len(specs), max_active, 128)
    for batch_index, (start, _, active) in enumerate(specs):
        expected = full_posterior.mu[batch_index, start : start + active]
        assert torch.allclose(
            raw_mu[batch_index, :active], expected, atol=1e-6
        )
        if active < max_active:
            assert not raw_mu[batch_index, active:].any()


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


def test_statistics_files_only_require_physical_arrays(tmp_path):
    motion = write_statistics(tmp_path)
    model = BodyVAE(
        motion_stats_path=motion,
        latent_dim=4,
        hidden_dim=8,
        encoder_layers=1,
        decoder_layers=1,
    )
    assert torch.equal(model.body_cont_mean, torch.zeros(255))
    assert torch.equal(model.local_root_std, torch.ones(4))


def test_tiny_deterministic_mu_path_can_overfit_two_clips():
    torch.manual_seed(7)
    model = make_vae(
        latent_dim=8,
        hidden_dim=16,
        encoder_layers=1,
        decoder_layers=1,
    ).train()
    body = torch.randn(2, 8, 259)
    local = torch.randn(2, 2, 4, 4)
    valid = torch.ones_like(local, dtype=torch.bool)
    mask = torch.ones(2, 8, dtype=torch.bool)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    def reconstruction():
        output = model.decode(model.encode(body, mask).mu, local, valid)
        return (output.continuous_body - body[..., :255]).square().mean()

    initial = reconstruction().detach()
    for _ in range(20):
        optimizer.zero_grad()
        loss = reconstruction()
        loss.backward()
        optimizer.step()
    assert reconstruction().detach() < initial
