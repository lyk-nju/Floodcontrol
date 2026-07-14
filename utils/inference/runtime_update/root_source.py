"""Compatibility-facing adapters into authoritative root-source contracts."""

from __future__ import annotations

from typing import Any

import torch

from utils.inference.stream_runtime.contracts import RootSourceProposal


def world_traj7_to_proposal(
    world_traj7: torch.Tensor,
    *,
    source_id: str,
    version: int,
    metadata: dict[str, Any] | None = None,
    frame_mask: torch.Tensor | None = None,
    strip_anchor: bool = True,
) -> RootSourceProposal:
    """Adapt authored world 7D into the future-only runtime DTO."""
    route = torch.as_tensor(world_traj7).detach().cpu().float()
    if route.ndim != 2 or int(route.shape[-1]) != 7:
        raise ValueError(f"world_traj7 must have shape [T,7], got {tuple(route.shape)}")
    start = 1 if strip_anchor else 0
    future = route[start:]
    if int(future.shape[0]) == 0:
        raise ValueError("root source must contain at least one future frame")
    if frame_mask is None:
        mask = torch.ones(future.shape[0], dtype=torch.bool)
    else:
        full_mask = torch.as_tensor(frame_mask).detach().cpu().bool().reshape(-1)
        if int(full_mask.shape[0]) != int(route.shape[0]):
            raise ValueError("frame_mask must match world_traj7 length")
        mask = full_mask[start:]
    return RootSourceProposal(
        future_traj7=future,
        future_frame_mask=mask,
        source_id=str(source_id),
        version=int(version),
        metadata=dict(metadata or {}),
    )


def condition_scenario_to_proposal(
    scenario,
    *,
    source_kind: str,
    version: int = 0,
) -> RootSourceProposal:
    """Adapt dataset/synthetic experiment scenarios without leaking eval DTOs."""
    metadata = dict(getattr(scenario, "metadata", {}) or {})
    metadata.update(
        {
            "source_kind": str(source_kind),
            "scenario_name": str(scenario.name),
            "update_frames": tuple(int(frame) for frame in scenario.update_frames),
            "base_sample_name": getattr(scenario, "base_sample_name", None),
            "caption_index": getattr(scenario, "caption_index", None),
            "visual_mask": getattr(scenario, "visual_mask", None),
            "anchor_frame_7d": scenario.condition_traj7[0].detach().cpu(),
        }
    )
    return world_traj7_to_proposal(
        scenario.condition_traj7,
        source_id=str(scenario.name),
        version=int(version),
        metadata=metadata,
        strip_anchor=True,
    )


def proposal_to_world_traj7(proposal: RootSourceProposal) -> torch.Tensor:
    """Materialize the authored anchor plus future for legacy diagnostics."""
    anchor = proposal.metadata.get("anchor_frame_7d")
    if anchor is None:
        raise ValueError("proposal metadata does not contain anchor_frame_7d")
    anchor_tensor = torch.as_tensor(anchor).detach().cpu().float().reshape(1, 7)
    return torch.cat([anchor_tensor, proposal.future_traj7], dim=0)


__all__ = [
    "RootSourceProposal",
    "condition_scenario_to_proposal",
    "proposal_to_world_traj7",
    "world_traj7_to_proposal",
]
