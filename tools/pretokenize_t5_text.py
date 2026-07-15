"""
Offline: run UMT5-XXL encoder on every unique caption under cfg.data (HumanML3D/Babel-style texts/*.txt).

调用方必须显式传入包含 ``model.params`` 与 ``data`` 的训练配置。新版four-frame
训练配置尚未接线，因此本工具不再回退到已经删除的legacy ``configs/ldf.yaml``。

Saves a single .pt file: { "embeddings": { caption_str: FloatTensor[L, 4096] on CPU }, "text_dim": 4096, ... }

Future training integration may pass these embeddings through ``LDFCondition``.

多卡（每 GPU 各加载一份 T5，分片编码，rank0 合并为一个 .pt）::

    torchrun --standalone --nproc_per_node=4 tools/pretokenize_t5_text.py --config <future-training-config>
"""

from __future__ import annotations

import argparse
import os
import sys
import torch
import torch.distributed as dist

from typing import Dict, Iterable, List, Optional, Set
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import OmegaConf
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.tools.t5 import T5EncoderModel
from utils.initialize import load_config
from utils.training.ldf.text import create_text_embedding_content_id


def _create_payload(
    embeddings: Dict[str, torch.Tensor],
    *,
    text_len: int,
    checkpoint_path: str | None,
    tokenizer_path: str | None,
) -> dict[str, object]:
    payload = {
        "embeddings": embeddings,
        "text_dim": 4096,
        "text_len": int(text_len),
        "checkpoint_path": str(checkpoint_path),
        "tokenizer_path": str(tokenizer_path),
    }
    payload["content_id"] = create_text_embedding_content_id(
        embeddings,
        text_dim=4096,
        text_len=int(text_len),
        checkpoint_path=str(checkpoint_path),
        tokenizer_path=str(tokenizer_path),
    )
    return payload

def _resolve_config_path(path: str) -> str:
    """支持从项目根或当前工作目录解析配置路径。"""
    if os.path.isabs(path) and os.path.isfile(path):
        return path
    if os.path.isfile(path):
        return os.path.abspath(path)
    alt = os.path.join(REPO_ROOT, path)
    if os.path.isfile(alt):
        return alt
    raise FileNotFoundError(
        f"找不到配置文件: {path!r}（已尝试 cwd 与仓库根目录 {REPO_ROOT!r}）"
    )


def _parse_overrides(override_list: Optional[List[str]]) -> Optional[Dict[str, str]]:
    if not override_list:
        return None
    out: Dict[str, str] = {}
    for item in override_list:
        key, sep, value = item.partition("=")
        if sep != "=":
            raise ValueError(f'--override 应为 key=value，收到: {item!r}')
        out[key.strip()] = value.strip()
    return out


def _meta_paths_from_data_block(data_cfg) -> List[str]:
    out: List[str] = []
    for key in ("train_meta_paths", "val_meta_paths", "test_meta_paths"):
        if key not in data_cfg or data_cfg[key] is None:
            continue
        out.extend(OmegaConf.to_container(data_cfg[key], resolve=True))
    return out


def collect_unique_captions(cfg) -> Set[str]:
    """Match datasets/humanml3d.py load_text: caption is the substring before first '#'."""
    captions: Set[str] = set()
    data = cfg.data
    blocks = []
    if getattr(data, "datasets", None) is not None and len(data.datasets) > 0:
        for ds in data.datasets:
            blocks.append(ds)
    else:
        blocks.append(data)

    for block in blocks:
        text_path = block.get("text_path", None)
        if text_path is None:
            rank_zero_info("Skipping a data block with text_path=None.")
            continue
        for meta_file in _meta_paths_from_data_block(block):
            meta_file = os.path.normpath(meta_file)
            if not os.path.isfile(meta_file):
                rank_zero_info(f"Meta file missing, skip: {meta_file}")
                continue
            root = os.path.dirname(meta_file)
            with open(meta_file, "r") as f:
                names = [ln.strip() for ln in f if ln.strip()]
            for name in tqdm(names, desc=f"scan {os.path.basename(meta_file)}", leave=False):
                txt_path = os.path.join(root, text_path, name + ".txt")
                if not os.path.isfile(txt_path):
                    continue
                with open(txt_path, "r") as tf:
                    for line in tf:
                        line = line.strip()
                        if not line:
                            continue
                        cap = line.split("#", 1)[0].strip()
                        captions.add(cap)
    captions.add("")
    return captions


def batched(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _dist_world() -> tuple[int, int, int]:
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return ws, rank, local_rank


def _log0(msg: str, rank: int) -> None:
    if rank == 0:
        rank_zero_info(msg)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description=(
            "从显式配置文件读取 data / model.params，"
            "扫描 caption 并离线导出 T5 特征。"
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="包含data与model.params的训练YAML",
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="+",
        default=None,
        help="覆盖配置项，与 train_ldf 相同，例如: model.params.text_len=256",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .pt path (default: <dirname(first train meta)>/t5_text_embeddings.pt)",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help=(
            "If --output already exists, load it and encode only captions missing "
            "from the existing embeddings, then save the merged table."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="单进程时: cuda | cuda:0 | cpu（默认 cuda）。多卡 torchrun 时忽略，改用 LOCAL_RANK。",
    )
    args = parser.parse_args()

    world_size, rank, local_rank = _dist_world()
    distributed = world_size > 1

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("多卡 pretokenize 需要 CUDA；请用单进程或确保 torchrun + GPU。")
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        if args.device:
            device = torch.device(args.device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = _resolve_config_path(args.config)
    override_args = _parse_overrides(args.override)
    _log0(f"pretokenize_t5_text: 加载配置 {config_path}", rank)
    if override_args:
        _log0(f"pretokenize_t5_text: --override {override_args}", rank)
    if distributed:
        _log0(
            f"pretokenize_t5_text: 分布式 WORLD_SIZE={world_size}（torchrun），本进程 RANK={rank} LOCAL_RANK={local_rank}",
            rank,
        )

    cfg = load_config(config_path, override_args)
    oc = cfg.config

    checkpoint_path = OmegaConf.select(oc, "text_encoder.checkpoint_path")
    tokenizer_path = OmegaConf.select(oc, "text_encoder.tokenizer_path")
    text_len = OmegaConf.select(oc, "text_encoder.text_len")
    if checkpoint_path is None:
        checkpoint_path = OmegaConf.select(oc, "model.params.checkpoint_path")
    if tokenizer_path is None:
        tokenizer_path = OmegaConf.select(oc, "model.params.tokenizer_path")
    if text_len is None:
        text_len = OmegaConf.select(oc, "model.params.text_len", default=512)
    if checkpoint_path is None or not os.path.isfile(str(checkpoint_path)):
        raise FileNotFoundError(
            f"T5 checkpoint not found at {checkpoint_path!r}; set "
            "text_encoder.checkpoint_path"
        )
    if tokenizer_path is None or not os.path.isdir(str(tokenizer_path)):
        raise FileNotFoundError(
            f"T5 tokenizer not found at {tokenizer_path!r}; set "
            "text_encoder.tokenizer_path"
        )
    model_text_len = int(OmegaConf.select(oc, "model.params.text_len", default=text_len))
    if int(text_len) != model_text_len:
        raise ValueError("text_encoder.text_len and model.params.text_len must match")
    _log0(
        f"从配置读取 T5: checkpoint_path={checkpoint_path!r}, "
        f"tokenizer_path={tokenizer_path!r}, text_len={text_len}",
        rank,
    )

    if distributed:
        if rank == 0:
            _bucket: List[Optional[List[str]]] = [sorted(collect_unique_captions(oc))]
        else:
            _bucket = [None]
        dist.broadcast_object_list(_bucket, src=0)
        ordered = _bucket[0]
        assert ordered is not None
        _log0(f"Unique captions (incl. empty): {len(ordered)}", rank)
    else:
        captions = collect_unique_captions(oc)
        ordered = sorted(captions)
        rank_zero_info(f"Unique captions (incl. empty): {len(ordered)}")

    out_path = args.output
    if out_path is None:
        first = None
        if getattr(oc.data, "datasets", None) is not None and len(oc.data.datasets) > 0:
            blocks = list(oc.data.datasets)
        else:
            blocks = [oc.data]
        for block in blocks:
            for p in _meta_paths_from_data_block(block):
                if os.path.isfile(p):
                    first = p
                    break
            if first:
                break
        if first is None:
            raise RuntimeError("Could not resolve default --output: no existing meta file found.")
        out_path = os.path.join(os.path.dirname(first), "t5_text_embeddings.pt")

    if rank == 0:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if distributed:
        dist.barrier()

    existing_embeddings: Dict[str, torch.Tensor] = {}
    existing_keys: Set[str] = set()
    if args.reuse_existing:
        if distributed:
            bucket: List[Optional[Set[str]]] = [None]
            if rank == 0 and os.path.isfile(out_path):
                existing_blob = torch.load(out_path, map_location="cpu", weights_only=False)
                existing_embeddings = existing_blob.get("embeddings", {})
                existing_keys = set(existing_embeddings.keys())
                bucket[0] = existing_keys
                _log0(
                    f"Loaded existing embeddings: {len(existing_keys)} entries from {out_path}",
                    rank,
                )
            dist.broadcast_object_list(bucket, src=0)
            existing_keys = bucket[0] or set()
        elif os.path.isfile(out_path):
            existing_blob = torch.load(out_path, map_location="cpu", weights_only=False)
            existing_embeddings = existing_blob.get("embeddings", {})
            existing_keys = set(existing_embeddings.keys())
            rank_zero_info(
                f"Loaded existing embeddings: {len(existing_keys)} entries from {out_path}"
            )

        if existing_keys:
            before = len(ordered)
            ordered = [
                caption
                for caption in ordered
                if caption not in existing_keys and caption.strip() not in existing_keys
            ]
            _log0(
                f"Reusing existing embeddings: {before - len(ordered)} covered, "
                f"{len(ordered)} missing captions to encode.",
                rank,
            )

    n = len(ordered)
    start = rank * n // world_size
    end = (rank + 1) * n // world_size
    local_ordered = ordered[start:end]

    encoder = T5EncoderModel(
        text_len=text_len,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        shard_fn=None,
    )
    encoder.model.to(device)
    encoder.model.eval()

    emb_dict: Dict[str, torch.Tensor] = {}
    batches = list(batched(local_ordered, args.batch_size))
    pbar = tqdm(
        batches,
        desc=f"T5 encode rank {rank}/{world_size}",
        disable=(distributed and rank != 0),
    )
    for batch in pbar:
        encoded = encoder(batch, device)
        for s, t in zip(batch, encoded):
            if t.ndim != 2 or t.shape[-1] != 4096:
                raise ValueError(f"UMT5 embedding for {s!r} must be [L,4096]")
            if not bool(torch.isfinite(t).all()):
                raise ValueError(f"UMT5 embedding for {s!r} contains non-finite values")
            emb_dict[s] = t.detach().cpu()

    encoder.model.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if distributed:
        if rank == 0:
            gather_list: List[Optional[Dict[str, torch.Tensor]]] = [None] * world_size
        else:
            gather_list = None
        dist.gather_object(emb_dict, gather_list, dst=0)
        if rank == 0:
            assert gather_list is not None
            merged: Dict[str, torch.Tensor] = dict(existing_embeddings)
            for part in gather_list:
                assert part is not None
                merged.update(part)
            payload = _create_payload(
                merged,
                text_len=text_len,
                checkpoint_path=checkpoint_path,
                tokenizer_path=tokenizer_path,
            )
            torch.save(payload, out_path)
            rank_zero_info(
                f"Saved {len(merged)} entries to {out_path} "
                f"(content_id={payload['content_id']})"
            )
        dist.barrier()
        dist.destroy_process_group()
    else:
        merged = dict(existing_embeddings)
        merged.update(emb_dict)
        payload = _create_payload(
            merged,
            text_len=text_len,
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
        )
        torch.save(payload, out_path)
        rank_zero_info(
            f"Saved {len(merged)} entries to {out_path} "
            f"(content_id={payload['content_id']})"
        )


if __name__ == "__main__":
    main()
