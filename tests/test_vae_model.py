import torch

from models.vae_wan_1d import BodyVAE


def make_model():
    return BodyVAE(
        latent_dim=128,
        hidden_dim=32,
        encoder_layers=2,
        decoder_layers=2,
        allow_identity_statistics=True,
    ).eval()


def make_local(batch=2, tokens=3):
    local = torch.randn(batch, tokens, 4, 4)
    valid = torch.ones_like(local, dtype=torch.bool)
    return local, valid


def test_four_frame_shapes_and_deterministic_tokenize():
    model = make_model()
    body = torch.randn(2, 12, 265)
    body[..., 261:] = torch.randint(0, 2, body[..., 261:].shape).float()
    mask = torch.ones(2, 12, dtype=torch.bool)
    posterior = model.encode(body, mask)
    assert posterior.mu.shape == (2, 3, 128)
    assert torch.equal(model.tokenize(body, mask), posterior.mu)
    local, valid = make_local()
    decoded = model.decode(posterior.mu, local, valid)
    assert decoded.continuous_body.shape == (2, 12, 261)
    assert decoded.contact_logits.shape == (2, 12, 4)


def test_encoder_is_token_causal():
    model = make_model()
    body = torch.randn(1, 12, 265)
    mask = torch.ones(1, 12, dtype=torch.bool)
    before = model.encode(body, mask).mu
    changed = body.clone()
    changed[:, 8:] += 100
    after = model.encode(changed, mask).mu
    assert torch.allclose(before[:, :2], after[:, :2], atol=1e-6)


def test_offline_and_stream_decode_parity_and_snapshot():
    model = make_model()
    latent = torch.randn(2, 3, 128)
    local, valid = make_local()
    offline = model.decode(latent, local, valid)
    state = model.init_decoder_state(2)
    chunks = []
    contacts = []
    for token in range(3):
        if token == 1:
            snapshot = model.snapshot_decoder_state(state)
        state, output = model.stream_decode_step(
            latent[:, token : token + 1], local[:, token : token + 1], state,
            valid[:, token : token + 1], normalized_latent=False
        )
        chunks.append(output.continuous_body)
        contacts.append(output.contact_logits)
    assert torch.allclose(torch.cat(chunks, 1), offline.continuous_body, atol=1e-5)
    assert torch.allclose(torch.cat(contacts, 1), offline.contact_logits, atol=1e-5)
    restored = model.restore_decoder_state(snapshot)
    _, replay = model.stream_decode_step(
        latent[:, 1:2], local[:, 1:2], restored, valid[:, 1:2], normalized_latent=False
    )
    assert torch.equal(replay.continuous_body, chunks[1])


def test_decoder_states_are_session_isolated():
    model = make_model()
    state_a = model.init_decoder_state(1)
    state_b = model.init_decoder_state(1)
    latent = torch.randn(1, 1, 128)
    local, valid = make_local(batch=1, tokens=1)
    next_a, _ = model.stream_decode_step(
        latent, local, state_a, valid, normalized_latent=False
    )
    assert state_b.token_index == 0
    assert all(torch.count_nonzero(cache) == 0 for pair in state_b.caches for cache in pair)
    assert next_a.token_index == 1


def test_tiny_deterministic_mu_path_can_overfit_two_clips():
    torch.manual_seed(7)
    model = BodyVAE(
        latent_dim=8, hidden_dim=16, encoder_layers=1, decoder_layers=1,
        allow_identity_statistics=True
    ).train()
    body = torch.randn(2, 8, 265)
    body[..., 261:] = 0
    local = torch.randn(2, 2, 4, 4)
    valid = torch.ones_like(local, dtype=torch.bool)
    frame_mask = torch.ones(2, 8, dtype=torch.bool)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    def reconstruction():
        mu = model.encode(body, frame_mask).mu
        output = model.decode(mu, local, valid)
        return (output.continuous_body - body[..., :261]).square().mean()

    initial = reconstruction().detach()
    for _ in range(20):
        optimizer.zero_grad()
        loss = reconstruction()
        loss.backward()
        optimizer.step()
    assert reconstruction().detach() < initial
