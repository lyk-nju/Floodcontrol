"""Lightning-side boundary between physical motion and the hybrid LDF state."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch.utilities import rank_zero_warn
from omegaconf import OmegaConf

from utils.conditions.ldf import HybridMotion
from utils.initialize import instantiate_target
from utils.motion_process import ROOT_DIM
from utils.token_frame import (
    FRAMES_PER_TOKEN,
    frame_count_to_token_count,
    frame_valid_to_token_valid,
    require_aligned_frame_count,
)
from utils.training.lightning_module import BasicLightningModule
from utils.training.vae.checkpoint import load_vae_checkpoint
from utils.training.ldf.steps import anchor_physical_batch
from utils.training.ldf.conditioning import (
    create_xz_condition,
    sample_constraint_keep_mask,
    sample_future_horizon_tokens,
    sample_xz_constraint_mask,
)
from utils.training.ldf.losses import compute_offpath_loss, compute_velocity_loss
from utils.training.ldf.metrics import compute_heading_metrics
from utils.training.ldf.solver import run_training_solver
from utils.training.ldf.window import (
    ColdStartObjective,
    sample_cold_start_objective,
    sample_rollout_steps,
    sample_window_plan,
)
from utils.training.ldf.text import TextEmbeddingLookup


_LDF_STATISTIC_NAMES = (
    "root_mean",
    "root_std",
    "local_root_mean",
    "local_root_std",
)
_VAE_STATISTIC_NAMES = (
    "body_cont_mean",
    "body_cont_std",
    "local_root_mean",
    "local_root_std",
    "latent_mean",
    "latent_std",
)
_MODEL_RESUME_PARAMETER_NAMES = (
    "hidden_dim",
    "ffn_dim",
    "freq_dim",
    "text_dim",
    "text_len",
    "num_heads",
    "root_num_layers",
    "body_num_layers",
    "chunk_size",
    "noise_steps",
    "fps",
    "time_embedding_scale",
    "prediction_type",
)
_MODEL_RESUME_PARAMETER_DEFAULTS = {
    "time_embedding_scale": 1.0,
    "prediction_type": "vel",
}


def _create_curriculum_generator(seed: int, global_step: int) -> torch.Generator:
    """Create one rank-independent RNG for the global K/replay decision."""

    mixed_seed = (
        int(seed) + int(global_step) * 1_000_003 + 0x5E1F_F0CE
    ) % (2**63 - 1)
    return torch.Generator(device="cpu").manual_seed(mixed_seed)


def _persistent_cold_validation_phases(
    *,
    noise_steps: int,
    active_tokens: int,
    rollout_commits: int,
) -> tuple[tuple[str, int], ...]:
    """Resolve stable observation points in one true-cold lifecycle."""

    noise_steps = int(noise_steps)
    active_tokens = int(active_tokens)
    rollout_commits = int(rollout_commits)
    if noise_steps <= 0 or active_tokens <= 0:
        raise ValueError("persistent cold validation geometry must be positive")
    if noise_steps % active_tokens:
        raise ValueError(
            "persistent cold validation requires divisible noise/active steps"
        )
    if rollout_commits < 2:
        raise ValueError(
            "persistent cold validation requires at least two rollout commits"
        )
    steps_per_commit = noise_steps // active_tokens
    three_visible_index = (min(3, active_tokens) - 1) * steps_per_commit
    return (
        ("after_update_01", 0),
        ("three_tokens_visible", three_visible_index),
        ("first_commit", noise_steps - 1),
        ("second_commit", noise_steps + steps_per_commit - 1),
    )


def _load_root_statistics(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(path, allow_pickle=False) as data:
        values = []
        for name in ("root_mean", "root_std"):
            if name not in data:
                raise ValueError(f"root statistics are missing {name!r}")
            value = torch.from_numpy(np.asarray(data[name])).float()
            if tuple(value.shape) != (ROOT_DIM,):
                raise ValueError(f"{name} must have shape [{ROOT_DIM}]")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must contain only finite values")
            if name == "root_std" and bool((value <= 0).any()):
                raise ValueError("root_std must be positive")
            values.append(value)
    return values[0], values[1]


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_parameter_content_id(module: torch.nn.Module) -> str:
    """Identify the learned tokenizer actually loaded for LDF use."""

    digest = hashlib.blake2b(digest_size=20)
    for name, parameter in module.named_parameters():
        value = parameter.detach().cpu().contiguous()
        encoded_name = name.encode("utf-8")
        encoded_dtype = str(value.dtype).encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "little"))
        digest.update(encoded_name)
        digest.update(len(encoded_dtype).to_bytes(8, "little"))
        digest.update(encoded_dtype)
        digest.update(len(value.shape).to_bytes(8, "little"))
        for dimension in value.shape:
            digest.update(int(dimension).to_bytes(8, "little"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _model_resume_signature(params, *, latent_dim: int) -> dict[str, object]:
    """Keep only architecture and mathematical-semantics parameters."""

    signature: dict[str, object] = {"latent_dim": int(latent_dim)}
    for name in _MODEL_RESUME_PARAMETER_NAMES:
        if name in params:
            signature[name] = _plain_config(params[name])
        elif name in _MODEL_RESUME_PARAMETER_DEFAULTS:
            signature[name] = _MODEL_RESUME_PARAMETER_DEFAULTS[name]
        else:
            raise RuntimeError(f"LDF model configuration is missing {name!r}")
    return signature


def _plain_config(value):
    if value is None:
        return None
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


class LDFLightningModule(BasicLightningModule):
    """Train hybrid root/body flow targets with frozen VAE and text features."""

    def __init__(self, cfg):
        vae = instantiate_target(
            target=cfg.vae.target,
            cfg=None,
            hfstyle=False,
            **cfg.vae.params,
        )
        load_vae_checkpoint(
            vae,
            cfg.vae.checkpoint_path,
            use_ema=True,
            freeze=True,
        )
        if not vae.latent_statistics_ready:
            raise RuntimeError("LDF requires VAE latent statistics")

        root_mean, root_std = _load_root_statistics(cfg.data.root_stats_path)
        model_params = dict(cfg.model.params)
        injected_names = {
            "latent_dim",
            "root_mean",
            "root_std",
            "local_root_mean",
            "local_root_std",
        }
        duplicated = injected_names.intersection(model_params)
        if duplicated:
            raise ValueError(
                "LDF statistics and latent_dim are derived from the configured VAE; "
                f"remove duplicated model params: {sorted(duplicated)}"
            )
        model = instantiate_target(
            target=cfg.model.target,
            cfg=None,
            hfstyle=False,
            latent_dim=vae.latent_dim,
            root_mean=root_mean,
            root_std=root_std,
            local_root_mean=vae.local_root_mean.detach().cpu(),
            local_root_std=vae.local_root_std.detach().cpu(),
            **model_params,
        )
        super().__init__(cfg, model=model)
        self.vae = vae
        self.vae.eval().requires_grad_(False)
        self.text_embeddings = TextEmbeddingLookup(
            cfg.data.text_embeddings_path,
            expected_dim=int(cfg.model.params.text_dim),
            expected_text_len=int(cfg.model.params.text_len),
        )
        if self.model.latent_dim != self.vae.latent_dim:
            raise RuntimeError("LDF and VAE latent dimensions do not match")
        if not torch.equal(
            self.model.local_root_mean.cpu(), self.vae.local_root_mean.cpu()
        ) or not torch.equal(
            self.model.local_root_std.cpu(), self.vae.local_root_std.cpu()
        ):
            raise RuntimeError("LDF and VAE local-root statistics do not match")
        self._vae_checkpoint_path = str(cfg.vae.checkpoint_path)
        self._vae_tokenizer_content_id = _module_parameter_content_id(self.vae)
        self._last_heading_metrics: dict[str, torch.Tensor] = {}

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "vae"):
            self.vae.eval()
        return self

    @torch.no_grad()
    def _encode_prompt_timeline(
        self,
        prompt_timeline: list[list[str]],
        *,
        apply_dropout: bool,
        generator: torch.Generator | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if not prompt_timeline or not prompt_timeline[0]:
            raise ValueError("prompt_timeline must be a non-empty [B][T] list")
        token_length = len(prompt_timeline[0])
        if any(len(row) != token_length for row in prompt_timeline):
            raise ValueError("all prompt timelines must have the same token length")
        timelines = [[str(text) for text in row] for row in prompt_timeline]
        training_config = self.cfg.get("training")
        dropout = float(
            0.0
            if training_config is None
            else training_config.get("text_dropout", 0.0)
        )
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("text_dropout must lie in [0,1]")
        if apply_dropout and dropout:
            dropped = (
                torch.rand(
                    len(timelines),
                    device=self.device,
                    generator=generator,
                )
                < dropout
            )
            for batch_index, should_drop in enumerate(dropped.tolist()):
                if should_drop:
                    timelines[batch_index] = [""] * token_length

        flattened = [text for row in timelines for text in row]
        unique = list(dict.fromkeys(flattened + [""]))
        encoded = self.text_embeddings.lookup(unique)
        by_text = {text: value.detach() for text, value in zip(unique, encoded)}
        contexts = [by_text[text] for text in flattened]
        null = [by_text[""] for _ in timelines]
        return contexts, null

    @torch.no_grad()
    def _create_clean_motion(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[HybridMotion, torch.Tensor]:
        """Convert one physical LDF data batch to normalized clean motion."""

        root = batch["root_motion"]
        frame_valid = batch["frame_valid_mask"]
        if root.ndim != 3 or root.shape[-1] != ROOT_DIM:
            raise ValueError("root_motion must be physical [B,F,5]")
        if tuple(frame_valid.shape) != tuple(root.shape[:2]):
            raise ValueError("frame_valid_mask must match root_motion [B,F]")
        if frame_valid.dtype != torch.bool:
            raise TypeError("frame_valid_mask must be bool")
        frames = require_aligned_frame_count(root.shape[1])
        tokens = frame_count_to_token_count(frames)
        token_valid = frame_valid_to_token_valid(frame_valid)

        latent = self.vae.tokenize_window(
            batch["body_with_context"],
            batch["body_with_context_frame_valid_mask"],
            batch["context_token_count"],
        )
        if tuple(latent.shape[:2]) != (root.shape[0], tokens):
            raise ValueError(
                "VAE active latent shape does not match the active root token axis"
            )
        if latent.shape[-1] != self.model.latent_dim:
            raise ValueError("VAE latent dimension does not match LDF")

        root = root.reshape(
            root.shape[0], tokens, FRAMES_PER_TOKEN, ROOT_DIM
        )
        root = self.model.normalize_root(root)
        root = torch.where(
            token_valid[..., None, None], root, torch.zeros_like(root)
        )
        latent = torch.where(
            token_valid[..., None], latent, torch.zeros_like(latent)
        )
        motion = HybridMotion(root.detach(), latent.detach())
        motion.validate()
        return motion, token_valid

    def _step(
        self,
        batch,
        is_training=True,
        *,
        generator: torch.Generator | None = None,
        rollout_steps_override: int | None = None,
        initial_history_tokens: int | torch.Tensor | None = None,
        cold_persistent_microstep_override: int | None = None,
    ):
        self._last_heading_metrics = {}
        if generator is None and is_training:
            generator = torch.Generator(device=self.device).manual_seed(
                int(self.cfg.get("seed", 0))
                + int(self.global_step) * 1_000_003
                + int(getattr(self, "global_rank", 0))
            )
        rollout_steps = 1 if rollout_steps_override is None else int(
            rollout_steps_override
        )
        self_forcing = self.cfg.get("self_forcing")
        persistent_cold_validation = cold_persistent_microstep_override is not None
        if persistent_cold_validation and is_training:
            raise ValueError(
                "cold_persistent_microstep_override is only valid in validation"
            )
        cold_start_replay = bool(
            (is_training and batch.get("cold_start_replay", False))
            or persistent_cold_validation
        )
        cold_objective = None
        if cold_start_replay:
            if (
                not persistent_cold_validation
                and rollout_steps_override not in (None, 1)
            ):
                raise ValueError("cold-start replay requires K=1")
            if self_forcing is None:
                raise ValueError("cold-start replay requires self_forcing config")
            cold_config = self_forcing.get("cold_start") or {}
            if persistent_cold_validation:
                cold_objective = ColdStartObjective(
                    persistent=True,
                    rollout_commits=int(cold_config.get("rollout_commits", 1)),
                    supervised_microstep=int(cold_persistent_microstep_override),
                )
            else:
                cold_objective = sample_cold_start_objective(
                    persistent_probability=float(
                        cold_config.get("persistent_probability", 0.0)
                    ),
                    rollout_commits=int(cold_config.get("rollout_commits", 1)),
                    noise_steps=int(self.model.noise_steps),
                    active_tokens=int(self.model.chunk_size),
                    generator=_create_curriculum_generator(
                        int(self.cfg.get("seed", 0)), int(self.global_step)
                    ),
                )
            rollout_steps = int(cold_objective.rollout_commits)
            initial_history_tokens = 0
        elif (
            rollout_steps_override is None
            and is_training
            and self_forcing is not None
        ):
            schedule = [tuple(row) for row in self_forcing.k_schedule]
            replay = {
                int(key): float(value)
                for key, value in dict(self_forcing.teacher_replay).items()
            }
            rollout_steps = sample_rollout_steps(
                int(self.global_step),
                # K selects the objective and therefore the synchronized
                # metric/compute path.  It must be one global-batch choice,
                # not a rank-local augmentation draw.  Motion noise, H and
                # condition sampling continue to use the rank-local RNG.
                generator=_create_curriculum_generator(
                    int(self.cfg.get("seed", 0)), int(self.global_step)
                ),
                schedule=schedule,
                teacher_replay=replay,
            )

        training_config = self.cfg.get("training") or {}
        window_config = training_config.get("window") or {}
        max_window_tokens = int(window_config.get("max_tokens", 0))
        generation_tokens = int(window_config.get("generation_tokens", 0))
        if max_window_tokens <= 0 or generation_tokens <= 0:
            raise ValueError("training.window must define positive max/generation tokens")
        if generation_tokens != self.model.chunk_size:
            raise ValueError(
                "training generation window must equal the model active chunk size"
            )
        cold_denoise_step = None
        cold_persistent_microstep = None
        cold_phase_offset = None
        if cold_start_replay:
            root_motion = batch["root_motion"]
            if cold_objective is None:
                raise RuntimeError("cold objective was not sampled")
            if cold_objective.persistent:
                if cold_objective.supervised_microstep is None:
                    raise RuntimeError(
                        "persistent cold objective has no supervised microstep"
                    )
                cold_persistent_microstep = int(
                    cold_objective.supervised_microstep
                )
            else:
                cold_denoise_step = torch.randint(
                    0,
                    int(self.model.noise_steps),
                    (int(root_motion.shape[0]),),
                    device=root_motion.device,
                    generator=generator,
                )
            # The ordinary phase_offset describes only a steady-state commit.
            # Cold replay owns its runtime denoise phase explicitly instead.
            cold_phase_offset = torch.zeros(
                root_motion.shape[0],
                device=root_motion.device,
                dtype=root_motion.dtype,
            )
        validate_contract = bool(self.cfg.get("debug", False))
        plan = sample_window_plan(
            batch,
            active_tokens=self.model.chunk_size,
            rollout_steps=rollout_steps,
            latent_dim=self.model.latent_dim,
            generator=generator,
            initial_history_tokens=initial_history_tokens,
            phase_offset=cold_phase_offset,
            # Once the explicit replay contract is configured, ordinary
            # batches must not obtain H=0 accidentally from uniform H sampling.
            allow_cold_start=(
                cold_start_replay
                or initial_history_tokens is not None
                or self_forcing is None
                or "cold_start_replay" not in self_forcing
            ),
        )
        if validate_contract:
            plan.validate()
        anchored_batch = anchor_physical_batch(
            batch, plan.translation_anchor_xz
        )
        clean_motion, token_valid = self._create_clean_motion(anchored_batch)
        contexts, null_contexts = self._encode_prompt_timeline(
            batch["prompt_timeline"],
            apply_dropout=is_training,
            generator=generator,
        )

        constraint_keep = sample_constraint_keep_mask(
            clean_motion.batch_size,
            dropout_probability=float(
                training_config.get("constraint_dropout", 0.0)
            ),
            device=clean_motion.root_motion.device,
            generator=generator,
            apply_dropout=is_training,
        )
        max_horizon_token = int(training_config.get("max_horizon_token", 0))
        future_horizon_tokens = sample_future_horizon_tokens(
            token_valid_mask=token_valid,
            initial_active_end=(
                plan.initial_history_tokens + plan.active_tokens
            ),
            rollout_steps=plan.rollout_steps,
            max_horizon_token=max_horizon_token,
            generator=generator,
        )
        sampling = training_config.get("constraint_sampling") or {}
        constraint_mask = sample_xz_constraint_mask(
            token_valid_mask=token_valid,
            initial_active_start=plan.initial_history_tokens,
            initial_active_end=(
                plan.initial_history_tokens + plan.active_tokens
            ),
            future_horizon_tokens=future_horizon_tokens,
            rollout_steps=plan.rollout_steps,
            dense_probability=float(sampling.get("dense_probability", 0.5)),
            waypoint_probability=float(
                sampling.get("waypoint_probability", 0.25)
            ),
            goal_probability=float(sampling.get("goal_probability", 0.25)),
            max_waypoint_count=int(sampling.get("max_waypoint_count", 4)),
            generator=generator,
        )
        constraint_mask &= constraint_keep[:, None, None, None]

        def condition_builder(view, condition_motion):
            return create_xz_condition(
                clean_root_motion=condition_motion.root_motion,
                token_valid_mask=token_valid,
                constraint_mask=constraint_mask,
                view=view,
                text_context=contexts,
                text_null_context=null_contexts,
                future_horizon_tokens=future_horizon_tokens,
            )

        result = run_training_solver(
            self.model,
            clean_motion,
            plan,
            previous_root_frame=anchored_batch.get("previous_root_frame"),
            previous_root_valid_mask=anchored_batch.get(
                "previous_root_valid_mask"
            ),
            condition_builder=condition_builder,
            cold_denoise_step=cold_denoise_step,
            cold_persistent_microstep=cold_persistent_microstep,
        )
        if validate_contract:
            result.final_step.inputs.validate()
        if self._should_observe_heading(is_training=is_training):
            self._last_heading_metrics = self._observe_heading(
                result,
                token_valid_mask=token_valid,
                target_body=batch["body_motion"],
            )
        weights = self.cfg.get("loss") or {}
        if result.is_rollout:
            return compute_offpath_loss(
                result.prediction,
                result.final_step,
                root_mean=self.model.root_mean,
                root_std=self.model.root_std,
                root_weight=float(weights.get("root_weight", 1.0)),
                body_weight=float(weights.get("body_weight", 1.0)),
                rollout_weight=float(weights.get("rollout_weight", 1.0)),
                offpath_beta_min=float(weights.get("offpath_beta_min", 0.1)),
                root_boundary_weight=float(
                    weights.get("root_boundary_weight", 0.0)
                ),
                root_heading_cosine_weight=float(
                    weights.get("root_heading_cosine_weight", 0.0)
                ),
                root_heading_vector_weight=float(
                    weights.get("root_heading_vector_weight", 0.0)
                ),
                root_heading_beta_min=float(
                    weights.get("root_heading_beta_min", 0.1)
                ),
                root_heading_cosine_min_norm=float(
                    weights.get("root_heading_cosine_min_norm", 0.05)
                ),
            )
        return compute_velocity_loss(
            result.prediction,
            result.final_step,
            root_mean=self.model.root_mean,
            root_std=self.model.root_std,
            root_weight=float(weights.get("root_weight", 1.0)),
            body_weight=float(weights.get("body_weight", 1.0)),
            root_heading_cosine_weight=float(
                weights.get("root_heading_cosine_weight", 0.0)
            ),
            root_heading_vector_weight=float(
                weights.get("root_heading_vector_weight", 0.0)
            ),
            root_heading_beta_min=float(weights.get("root_heading_beta_min", 0.1)),
            root_heading_cosine_min_norm=float(
                weights.get("root_heading_cosine_min_norm", 0.05)
            ),
        )

    def _should_observe_heading(self, *, is_training: bool) -> bool:
        """Decode heading metrics sparsely in training and fully in validation."""

        if not is_training:
            return True
        interval = max(1, int(self.cfg.trainer.get("log_every_n_steps", 1)))
        return int(self.global_step) % interval == 0

    @torch.no_grad()
    def _observe_heading(
        self,
        result,
        *,
        token_valid_mask: torch.Tensor,
        target_body: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Recover the current clean endpoint and measure physical headings."""

        step = result.final_step
        prediction = result.prediction
        beta = step.inputs.beta
        predicted_latent = (
            step.inputs.noisy_motion.latent_motion
            + beta[..., None].to(step.inputs.noisy_motion.latent_motion)
            * prediction.velocity.latent_motion
        )
        frame_valid = token_valid_mask.repeat_interleave(
            FRAMES_PER_TOKEN, dim=1
        )
        predicted_body = self.vae.detokenize(
            predicted_latent,
            prediction.local_root_motion,
            prediction.local_root_feature_valid,
            frame_valid,
        ).continuous_body
        predicted_root = self.model.denormalize_root(
            prediction.clean_root_motion
        ).flatten(1, 2)
        target_root = self.model.denormalize_root(
            step.clean_motion.root_motion
        ).flatten(1, 2)
        metric_frames = step.loss_mask.repeat_interleave(
            FRAMES_PER_TOKEN, dim=1
        )
        return compute_heading_metrics(
            predicted_root=predicted_root,
            target_root=target_root,
            predicted_body=predicted_body,
            target_body=target_body,
            frame_mask=metric_frames,
            frame_valid_mask=frame_valid,
            fps=float(self.model.fps),
        )

    def _log_heading_metrics(
        self,
        prefix: str,
        *,
        batch_size: int,
        on_step: bool,
        on_epoch: bool,
    ) -> None:
        for name, value in self._last_heading_metrics.items():
            self.log(
                f"{prefix}/{name}",
                value,
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=False,
                sync_dist=True,
                batch_size=batch_size,
            )

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        self._log_heading_metrics(
            "train_metric",
            batch_size=int(batch["body_motion"].shape[0]),
            on_step=True,
            on_epoch=False,
        )
        return loss

    def _resume_contract(self) -> dict[str, object]:
        return {
            "resume_contract_version": 2,
            "vae_tokenizer_content_id": self._vae_tokenizer_content_id,
            # This remains metadata rather than a hard resume condition. A
            # larger table or a different compatible text feature source is a
            # valid fine-tuning choice.
            "text_embedding_content_id": self.text_embeddings.content_id,
            "ldf_statistics": {
                name: getattr(self.model, name).detach().cpu().clone()
                for name in _LDF_STATISTIC_NAMES
            },
            "vae_statistics": {
                name: getattr(self.vae, name).detach().cpu().clone()
                for name in _VAE_STATISTIC_NAMES
            },
            "model_signature": _model_resume_signature(
                self.cfg.model.params,
                latent_dim=self.model.latent_dim,
            ),
        }

    def on_save_checkpoint(self, checkpoint) -> None:
        super().on_save_checkpoint(checkpoint)
        checkpoint["ldf_resume_contract"] = self._resume_contract()

    def on_load_checkpoint(self, checkpoint) -> None:
        state = checkpoint.get("state_dict", {})
        for name in _LDF_STATISTIC_NAMES:
            if name not in state:
                raise RuntimeError(f"LDF resume checkpoint is missing {name}")
            if not torch.equal(
                state[name].detach().cpu(), getattr(self.model, name).cpu()
            ):
                raise RuntimeError(f"LDF resume statistics mismatch for {name}")

        saved_contract = checkpoint.get("ldf_resume_contract")
        if saved_contract is None:
            saved_contract = checkpoint.get("ldf_training_contract")
        current_contract = self._resume_contract()
        if not isinstance(saved_contract, dict):
            raise RuntimeError(
                "LDF resume checkpoint has no resume contract; "
                "old checkpoints cannot be resumed by this training entrypoint"
            )
        legacy_path_contract = isinstance(saved_contract.get("paths"), dict)
        saved_tokenizer_id = saved_contract.get("vae_tokenizer_content_id")
        if saved_tokenizer_id is not None:
            if saved_tokenizer_id != current_contract["vae_tokenizer_content_id"]:
                raise RuntimeError("LDF resume VAE tokenizer content does not match")
        else:
            # Checkpoints written before the tokenizer-parameter identity was
            # introduced stored the hash of the complete VAE checkpoint. Keep
            # those checkpoints resumable without weakening their old check.
            legacy_checkpoint_id = saved_contract.get("vae_checkpoint_content_id")
            if isinstance(legacy_checkpoint_id, str):
                if legacy_checkpoint_id != _file_sha256(self._vae_checkpoint_path):
                    raise RuntimeError(
                        "LDF resume VAE checkpoint content does not match"
                    )
            elif legacy_path_contract:
                rank_zero_warn(
                    "LDF resume checkpoint predates VAE content identities; "
                    "continuing after strict LDF state and VAE statistics checks."
                )
            else:
                raise RuntimeError("LDF resume contract has no VAE identity")

        saved_text_id = saved_contract.get("text_embedding_content_id")
        if (
            isinstance(saved_text_id, str)
            and saved_text_id != current_contract["text_embedding_content_id"]
        ):
            rank_zero_warn(
                "LDF resume text embedding content changed; continuing because "
                "text tables are a compatible fine-tuning input when text_dim "
                "and text_len still match."
            )
        for group_name in ("ldf_statistics", "vae_statistics"):
            saved_statistics = saved_contract.get(group_name)
            if group_name == "ldf_statistics" and saved_statistics is None:
                if legacy_path_contract:
                    # The four LDF buffers were already checked directly from
                    # state_dict before any checkpoint value could overwrite
                    # the configured model.
                    continue
                saved_statistics = {}
            if not isinstance(saved_statistics, dict):
                raise RuntimeError(f"LDF resume contract has no {group_name}")
            for name, current in current_contract[group_name].items():
                saved = saved_statistics.get(name)
                if not torch.is_tensor(saved) or not torch.equal(saved.cpu(), current):
                    label = "LDF" if group_name == "ldf_statistics" else "VAE"
                    raise RuntimeError(
                        f"LDF resume {label} statistics mismatch for {name}"
                    )
        saved_model_signature = saved_contract.get("model_signature")
        if saved_model_signature is None:
            legacy_model = saved_contract.get("model")
            if isinstance(legacy_model, dict) and isinstance(
                legacy_model.get("params"), dict
            ):
                saved_model_signature = _model_resume_signature(
                    legacy_model["params"],
                    latent_dim=self.model.latent_dim,
                )
            elif legacy_path_contract:
                rank_zero_warn(
                    "LDF resume checkpoint predates model signatures; relying "
                    "on strict state_dict loading for this legacy checkpoint."
                )
            else:
                raise RuntimeError("LDF resume contract has no model signature")
        if (
            saved_model_signature is not None
            and saved_model_signature != current_contract["model_signature"]
        ):
            raise RuntimeError("LDF resume model structure contract does not match")
        super().on_load_checkpoint(checkpoint)
        if not torch.equal(
            self.model.local_root_mean.cpu(), self.vae.local_root_mean.cpu()
        ) or not torch.equal(
            self.model.local_root_std.cpu(), self.vae.local_root_std.cpu()
        ):
            raise RuntimeError("resumed LDF local-root statistics do not match the VAE")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        probe = str(batch.get("validation_probe", "teacher_cold"))
        validation = self.cfg.get("validation") or {}
        base_seed = int(validation.get("seed", self.cfg.get("seed", 0)))
        generator = torch.Generator(device=self.device).manual_seed(
            base_seed + int(dataloader_idx) * 1_000_003 + int(batch_idx)
        )
        persistent_microstep = None
        metric_probe = probe
        if probe == "teacher_cold":
            rollout_steps = 1
            history_tokens: int | torch.Tensor = torch.zeros_like(
                batch["span_token_count"]
            )
        elif probe == "persistent_cold":
            # Zero-based solver indices.  These correspond to the state after
            # update 1, the first phase with roughly three visible tokens, the
            # first commit, and the second commit respectively.  Cycling one
            # phase per batch avoids multiplying persistent validation cost by
            # four while keeping every phase deterministic across ranks.
            rollout_steps = int(
                (self.cfg.self_forcing.get("cold_start") or {}).get(
                    "rollout_commits", 1
                )
            )
            persistent_phases = _persistent_cold_validation_phases(
                noise_steps=int(self.model.noise_steps),
                active_tokens=int(self.model.chunk_size),
                rollout_commits=rollout_steps,
            )
            phase_name, persistent_microstep = persistent_phases[
                int(batch_idx) % len(persistent_phases)
            ]
            metric_probe = f"{probe}/{phase_name}"
            history_tokens = torch.zeros_like(batch["span_token_count"])
        elif probe == "teacher_continuation":
            rollout_steps = 1
            history_tokens = self._validation_history_tokens(
                batch,
                rollout_steps=rollout_steps,
                fallback=1,
            )
        elif probe == "self_forcing":
            rollout_steps = max(
                int(row[1]) for row in self.cfg.self_forcing.k_schedule
            )
            history_tokens = self._validation_history_tokens(
                batch,
                rollout_steps=rollout_steps,
                fallback=1,
            )
        else:
            raise ValueError(f"unknown validation probe {probe!r}")
        loss_dict = self._step(
            batch,
            is_training=False,
            generator=generator,
            rollout_steps_override=rollout_steps,
            initial_history_tokens=history_tokens,
            cold_persistent_microstep_override=persistent_microstep,
        )
        batch_size = int(batch["body_motion"].shape[0])
        for key, value in loss_dict.items():
            self.log(
                f"val_loss/{metric_probe}/{key}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
                add_dataloader_idx=False,
            )
        self._log_heading_metrics(
            f"val_metric/{metric_probe}",
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
        return loss_dict

    def _validation_history_tokens(
        self,
        batch,
        *,
        rollout_steps: int,
        fallback: int,
    ) -> torch.Tensor:
        """Resolve deterministic early/middle/late H probes per sample."""

        counts = batch["span_token_count"].to(dtype=torch.long)
        maximum = counts - self.model.chunk_size - (int(rollout_steps) - 1)
        if bool((maximum < 1).any()):
            raise ValueError("continuation validation parent is too short")
        positions = batch.get("validation_position")
        if positions is None:
            requested = torch.full_like(maximum, int(fallback))
            return torch.minimum(requested.clamp_min(1), maximum)
        history = torch.empty_like(maximum)
        for index, position in enumerate(positions):
            if position == "early":
                history[index] = 1
            elif position == "middle":
                history[index] = max(1, int(maximum[index].item()) // 2)
            elif position == "late":
                history[index] = maximum[index]
            else:
                raise ValueError(f"unknown validation position {position!r}")
        return history

__all__ = ["LDFLightningModule"]
