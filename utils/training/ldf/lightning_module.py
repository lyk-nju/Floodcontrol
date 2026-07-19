"""Lightning-side boundary between physical motion and the hybrid LDF state."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
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
from utils.training.ldf.solver import run_training_solver
from utils.training.ldf.window import (
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


def _create_curriculum_generator(seed: int, global_step: int) -> torch.Generator:
    """Create one rank-independent RNG for the global K/replay decision."""

    mixed_seed = (
        int(seed) + int(global_step) * 1_000_003 + 0x5E1F_F0CE
    ) % (2**63 - 1)
    return torch.Generator(device="cpu").manual_seed(mixed_seed)


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
        self._vae_checkpoint_content_id = _file_sha256(cfg.vae.checkpoint_path)

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
    ):
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
        cold_start_replay = bool(
            is_training and batch.get("cold_start_replay", False)
        )
        if cold_start_replay:
            if rollout_steps_override not in (None, 1):
                raise ValueError("cold-start replay requires K=1")
            rollout_steps = 1
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
        cold_phase_offset = None
        if cold_start_replay:
            root_motion = batch["root_motion"]
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
        )
        if validate_contract:
            result.final_step.inputs.validate()
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
            )
        return compute_velocity_loss(
            result.prediction,
            result.final_step,
            root_weight=float(weights.get("root_weight", 1.0)),
            body_weight=float(weights.get("body_weight", 1.0)),
        )

    def _training_contract(self) -> dict[str, object]:
        return {
            "vae_checkpoint_content_id": self._vae_checkpoint_content_id,
            "text_embedding_content_id": self.text_embeddings.content_id,
            "ldf_statistics": {
                name: getattr(self.model, name).detach().cpu().clone()
                for name in _LDF_STATISTIC_NAMES
            },
            "vae_statistics": {
                name: getattr(self.vae, name).detach().cpu().clone()
                for name in _VAE_STATISTIC_NAMES
            },
            "model": {
                "target": str(self.cfg.model.target),
                "params": _plain_config(self.cfg.model.params),
            },
            "training": {
                "window": _plain_config(self.cfg.training.window),
                "max_horizon_token": int(self.cfg.training.max_horizon_token),
                "self_forcing": _plain_config(self.cfg.self_forcing),
                "root_statistics": _plain_config(
                    self.cfg.get("root_statistics")
                ),
            },
        }

    def on_save_checkpoint(self, checkpoint) -> None:
        super().on_save_checkpoint(checkpoint)
        checkpoint["ldf_training_contract"] = self._training_contract()

    def on_load_checkpoint(self, checkpoint) -> None:
        state = checkpoint.get("state_dict", {})
        for name in _LDF_STATISTIC_NAMES:
            if name not in state:
                raise RuntimeError(f"LDF resume checkpoint is missing {name}")
            if not torch.equal(
                state[name].detach().cpu(), getattr(self.model, name).cpu()
            ):
                raise RuntimeError(f"LDF resume statistics mismatch for {name}")

        saved_contract = checkpoint.get("ldf_training_contract")
        current_contract = self._training_contract()
        if not isinstance(saved_contract, dict):
            raise RuntimeError(
                "LDF resume checkpoint has no frozen training contract; "
                "old checkpoints cannot be resumed by this training entrypoint"
            )
        for identity_name in (
            "vae_checkpoint_content_id",
            "text_embedding_content_id",
        ):
            if saved_contract.get(identity_name) != current_contract[identity_name]:
                label = identity_name.removesuffix("_content_id").replace("_", " ")
                raise RuntimeError(f"LDF resume {label} content does not match")
        for group_name in ("ldf_statistics", "vae_statistics"):
            saved_statistics = saved_contract.get(group_name, {})
            for name, current in current_contract[group_name].items():
                saved = saved_statistics.get(name)
                if not torch.is_tensor(saved) or not torch.equal(saved.cpu(), current):
                    label = "LDF" if group_name == "ldf_statistics" else "VAE"
                    raise RuntimeError(
                        f"LDF resume {label} statistics mismatch for {name}"
                    )
        if saved_contract.get("model") != current_contract["model"]:
            raise RuntimeError("LDF resume model structure contract does not match")
        if saved_contract.get("training") != current_contract["training"]:
            raise RuntimeError("LDF resume 50-token training contract does not match")
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
        if probe == "teacher_cold":
            rollout_steps = 1
            history_tokens: int | torch.Tensor = torch.zeros_like(
                batch["span_token_count"]
            )
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
        )
        batch_size = int(batch["body_motion"].shape[0])
        for key, value in loss_dict.items():
            self.log(
                f"val_loss/{probe}/{key}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
                add_dataloader_idx=False,
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
