"""Lightning-side boundary between physical motion and the hybrid LDF state."""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import torch

from utils.conditions.ldf import HybridMotion, LDFCondition
from utils.initialize import instantiate_target
from utils.motion_process import ROOT_DIM
from utils.token_frame import (
    FRAMES_PER_TOKEN,
    frame_count_to_token_count,
    frame_valid_to_token_valid,
    prefix_valid_token_count,
    require_aligned_frame_count,
)
from utils.training.lightning_module import BasicLightningModule
from utils.training.vae.checkpoint import load_vae_checkpoint
from utils.training.ldf.batch import anchor_physical_batch, compute_velocity_loss
from utils.training.ldf.self_forcing import (
    SelfForcingState,
    run_self_forcing_rollout,
    sample_rollout_steps,
    sample_window_plan,
    self_forcing_phase_progress,
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

        root_mean, root_std = _load_root_statistics(cfg.root_stats_path)
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
            cfg.text_embeddings_path,
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
            else training_config.get("text_dropout_probability", 0.0)
        )
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("text_dropout_probability must lie in [0,1]")
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
        frame_patches = frame_valid.reshape(
            root.shape[0], tokens, FRAMES_PER_TOKEN
        )
        if not bool((frame_patches == frame_patches[..., :1]).all()):
            raise ValueError(
                "active frame validity must be constant within each four-frame token"
            )
        token_valid = frame_valid_to_token_valid(frame_valid)
        root_valid_token_count = prefix_valid_token_count(token_valid)

        latent = self.vae.tokenize_window(
            batch["body_with_context"],
            batch["body_with_context_frame_valid_mask"],
            batch["context_token_count"],
        )
        encoder_token_valid = frame_valid_to_token_valid(
            batch["body_with_context_frame_valid_mask"]
        )
        encoder_valid_token_count = prefix_valid_token_count(encoder_token_valid)
        active_token_count = encoder_valid_token_count - batch[
            "context_token_count"
        ].to(device=encoder_valid_token_count.device)
        if not torch.equal(
            active_token_count,
            root_valid_token_count.to(device=active_token_count.device),
        ):
            raise ValueError(
                "VAE active token counts do not match active root token counts"
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
        initial_history_tokens: int | None = None,
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
        if (
            rollout_steps_override is None
            and is_training
            and self_forcing is not None
            and bool(self_forcing.get("enabled", False))
        ):
            progress = self_forcing_phase_progress(
                int(self.global_step),
                phase_start_step=int(self_forcing.phase_start_step),
                phase_steps=int(self_forcing.phase_steps),
            )
            schedule = [tuple(row) for row in self_forcing.k_schedule]
            replay = {
                int(key): float(value)
                for key, value in dict(self_forcing.teacher_replay).items()
            }
            rollout_steps = sample_rollout_steps(
                progress,
                generator=generator,
                schedule=schedule,
                teacher_replay=replay,
            )

        plan = sample_window_plan(
            batch,
            active_tokens=self.model.chunk_size,
            rollout_steps=rollout_steps,
            latent_dim=self.model.latent_dim,
            generator=generator,
            initial_history_tokens=initial_history_tokens,
        )
        anchored_batch = anchor_physical_batch(
            batch, plan.translation_anchor_xz
        )
        clean_motion, _ = self._create_clean_motion(anchored_batch)
        contexts, null_contexts = self._encode_prompt_timeline(
            batch["prompt_timeline"],
            apply_dropout=is_training,
            generator=generator,
        )

        def condition_builder(_view):
            return LDFCondition(
                text_context=contexts,
                text_null_context=null_contexts,
            )

        result = run_self_forcing_rollout(
            self.model,
            SelfForcingState(clean_motion),
            plan,
            previous_root_frame=anchored_batch.get("previous_root_frame"),
            previous_root_valid_mask=anchored_batch.get(
                "previous_root_valid_mask"
            ),
            condition_builder=condition_builder,
        )
        weights = self.cfg.get("loss") or {}
        return compute_velocity_loss(
            result.prediction,
            result.final_step,
            root_weight=float(weights.get("root_weight", 1.0)),
            body_weight=float(weights.get("body_weight", 1.0)),
        )

    def _training_contract(self) -> dict[str, object]:
        paths = {
            "vae_checkpoint_path": self.cfg.vae.checkpoint_path,
            "motion_stats_path": self.cfg.vae.params.motion_stats_path,
            "latent_stats_path": self.cfg.vae.params.latent_stats_path,
            "root_stats_path": self.cfg.root_stats_path,
            "text_embeddings_path": self.cfg.text_embeddings_path,
        }
        return {
            "paths": {
                name: str(Path(str(path)).expanduser().resolve())
                for name, path in paths.items()
            },
            "vae_statistics": {
                name: getattr(self.vae, name).detach().cpu().clone()
                for name in _VAE_STATISTIC_NAMES
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
        if saved_contract is None:
            warnings.warn(
                "legacy LDF checkpoint has no training contract; only model statistics "
                "were validated",
                stacklevel=2,
            )
        else:
            if saved_contract.get("paths") != current_contract["paths"]:
                raise RuntimeError("LDF resume VAE/statistics/text paths do not match")
            saved_statistics = saved_contract.get("vae_statistics", {})
            for name, current in current_contract["vae_statistics"].items():
                saved = saved_statistics.get(name)
                if not torch.is_tensor(saved) or not torch.equal(
                    saved.cpu(), current
                ):
                    raise RuntimeError(f"LDF resume VAE statistics mismatch for {name}")
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
            rollout_steps, history_tokens = 1, 0
        elif probe == "teacher_continuation":
            rollout_steps = 1
            history_tokens = int(validation.get("continuation_history_tokens", 5))
        elif probe == "self_forcing":
            rollout_steps = int(validation.get("self_forcing_steps", 5))
            history_tokens = int(validation.get("self_forcing_history_tokens", 1))
        else:
            raise ValueError(f"unknown validation probe {probe!r}")

        with self.ema.average_parameters(self._trainable_parameters):
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


__all__ = ["LDFLightningModule"]
