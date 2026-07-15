# VAE reconstruction evaluation

The VAE evaluation package exposes two separate tasks. Both use the first 10
sample IDs from the HumanML3D and BABEL validation TXT files, the frozen EMA
encoder/decoder, deterministic posterior means, and the source explicit root.
They differ only in how body tokens are scheduled before the causal decoder.

## 1. Direct stream task

Every deterministic body token is committed directly to one persistent
`VAEDecoderState`. This isolates VAE cache behavior and verifies full-sequence
offline decode against token-by-token decode.

```bash
python -m eval.vae.stream --config eval/vae/stream.yaml
```

Results are written below `eval/vae/output/stream/`.

## 2. Rolling-window task

The default active view has 10 history slots and 10 future slots. History is
right-aligned before the fixed commit boundary; future is left-aligned after
it. Each step commits only the first future token (four frames) and then moves
the window forward by one token. Missing history at cold start and missing
future at sequence end remain invalid padded slots. History is read-only and is
never replayed into the persistent decoder cache.

```bash
python -m eval.vae.rolling --config eval/vae/rolling.yaml
```

Results are written below `eval/vae/output/rolling/`. Reconstruction NPZ files
also contain the complete rolling trace: timeline position IDs, history/future
masks, window origin, history/future ranges and committed token indices.

## Smoke overrides

Both entrypoints accept the same overrides:

```bash
python -m eval.vae.rolling \
  --config eval/vae/rolling.yaml \
  --device cpu \
  --sample-count 1 \
  --skip-video \
  --output /tmp/vae_rolling_eval
```

Each task writes the same per-dataset layout:

```text
output/<task>/<dataset>/
├── video/original/<sample_id>.mp4
├── video/reconstruction/<sample_id>.mp4
├── motion/original/<sample_id>.npz
├── motion/reconstruction/<sample_id>.npz
├── metrics/<sample_id>.json
├── manifest.json
└── summary.json
```

The reconstruction NPZ retains deterministic `posterior_mu`, derived
local-root patches, validity masks, contact logits/probabilities and global
joints. Offline decoding is executed internally and must match the submitted
stream within the configured tolerance. The default `1e-4` tolerance accounts
for floating-point accumulation-order differences between full-sequence and
long token-by-token convolutions.
