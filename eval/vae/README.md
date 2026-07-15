# VAE reconstruction evaluation

The VAE evaluation package exposes two separate tasks. Both use the first 10
sample IDs from the HumanML3D and BABEL validation TXT files, the frozen EMA
encoder/decoder, deterministic posterior means, and the source explicit root.
They differ only in how much causal decoder history is retained.

## 1. Direct stream task

Every deterministic body token is committed directly to one persistent
`VAEDecoderState`. This isolates VAE cache behavior and verifies full-sequence
offline decode against token-by-token decode.

```bash
python -m eval.vae.stream --config eval/vae/stream.yaml
```

Results are written below `eval/vae/output/stream/`.

## 2. Rolling-window task

This task models a finite-history runtime rather than an LDF denoising window.
For each current token it creates a fresh `VAEDecoderState`, replays at most the
previous 10 deterministic posterior-mean tokens together with their GT
local-root patches, decodes the current token, and commits only its four output
frames. There are no future tokens because the VAE decoder is causal.

The same full causal encoder output is used by both tasks, so differences are
caused only by truncating decoder history. A persistent full-history stream is
saved as the reference. Each rolling window is also decoded offline from a
fresh boundary; agreement between that result and token-by-token replay checks
the cache implementation independently of the expected finite-history error.

```bash
python -m eval.vae.rolling --config eval/vae/rolling.yaml
```

Results are written below `eval/vae/output/rolling/`. Reconstruction NPZ files
also contain the complete rolling trace: timeline position IDs, history/current
masks, window boundaries and committed token indices. They additionally retain
the persistent-stream reference for direct numerical comparison.

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
joints. Direct stream must match full-sequence offline decode. Rolling replay
must match offline decode of each identical truncated window. The default
`1e-4` tolerance accounts for floating-point accumulation-order differences
between full-sequence and token-by-token convolutions.
