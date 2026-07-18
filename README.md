# Floodcontrol

Floodcontrol 是基于 FloodDiffusion 在线 diffusion-forcing 范式重建的动作生成仓库。新版采用 ARDY 风格的结构化生成状态：

```text
root_motion + latent_motion
        -> Root Transformer
        -> clean local_root_motion
        -> Body Transformer
```

当前仓库已经完成新版模型核心里程碑：

- strict `4 frames / token` 时间协议；
- body-only causal VAE核心、body265 codec、分组loss和仅含cache的显式`VAEDecoderState`已经落地；
- `HybridMotion`、typed condition、Root/Body non-causal LDF、constraint CFG以及dense trajectory / sparse waypoint / future goal XZ训练bridge已经落地；
- `generate/stream_generate_step`已经使用显式Hybrid流式状态；多token世界坐标生成统一由`InferenceSession.generate()`负责；
- HumanML3D/BABEL的独立body265 artifact、联合statistics与同构multi-dataset训练入口；
- 分主题架构文档和契约测试。

旧的附加控制网络、外置轨迹编码器、外置root planner以及263D/trajectory7 runtime已经从新版仓库物理删除。当前里程碑保证hybrid LDF与BodyVAE模型核心、唯一`humanml265`离线转换器、root5/body265运行时motion API、合成张量流式decode和LDF heading bridge。HumanML3D/BABEL Dataset只返回统一完整sample，VAE与LDF各自在training data层构造任务视图。第一版HumanML-only VAE已完成300k steps；训练、评测与LDF通过同一个公共函数加载EMA encoder+decoder。LDF训练入口已接入冻结EMA VAE在线确定性编码、冻结UMT5、逐token prompt timeline、固定噪声active band、persistent dense/sparse/goal XZ计划、最多20-token future XZ lookahead、相互独立的text/constraint dropout、root/body flow-v loss、EMA与可选detached self-forcing。HumanML caption在整段token上重复，BABEL区间被编译为token prompt timeline；每个motion token只cross-attend自身prompt。Web已经使用同一个`InferenceSession`完成commit-time decoder、四帧chunk、会话锁和route/text更新接线，只等待正式LDF checkpoint loader。

## 当前可验证范围

```bash
python -m pytest tests -q
```

迁移服务器时可以从FloodDiffusion同构的`HumanML3D`与`BABEL_streamed`源目录
一体化重建motion、statistics、UMT5和VAE latent statistics：

```bash
python -m tools.prepare_training_assets all \
  --raw-data-root /path/to/raw_data \
  --deps-root /path/to/deps \
  --vae-checkpoint /path/to/body_vae/last.ckpt \
  --workers 16 \
  --t5-devices 0,1,2 \
  --latent-device cuda:0
```

VAE checkpoint尚未训练完成时先运行`pre-vae`，训练完成后再运行`post-vae`；
完整输入、输出和恢复协议见
[`02_DATA_PIPELINE.md`](docs/rearchitecture/02_DATA_PIPELINE.md)。

测试覆盖typed condition、active/future XZ、Root/Body forward sensitivity、constraint CFG、逐token文本隔离、四帧VAE contract、HumanML恢复、随机yaw一致性、因果性、offline/stream parity、三角Hybrid stream和snapshot恢复。`train_vae.py`与`train_ldf.py`都会直接检查所需statistics、checkpoint和数据；正式LDF训练窗口固定为40 tokens/160 frames，canonical root statistics必须按这套窗口与逐样本anchor分布生成，不再使用额外的`training_ready`状态门闩。Web的四帧chunk runtime已经接入`InferenceSession`，模型加载在正式LDF checkpoint冻结前明确抛出`BLOCKED_ON_LDF_CHECKPOINT`。

LDF训练支持单卡和Lightning DDP。多卡时普通validation、dense-XZ轨迹/视频与完整HumanML T2M评测都会按sample分片；只有rank 0写全局summary和W&B。`configs/ldf.yaml`固定为远端8卡/resume配置，`configs/ldf_s5.yaml`单独保存本机S5配置；两者都显式拥有自己的路径，不依赖同一份`paths_default.yaml`切换服务器语义。

```bash
# 远端服务器
python train_ldf.py --config configs/ldf.yaml

# 本机S5
python train_ldf.py --config configs/ldf_s5.yaml
```

设计入口见 [`docs/rearchitecture/README.md`](docs/rearchitecture/README.md)。已经完成的代码与协议修改记录在 [`docs/DEVELOPMENT_LOG.md`](docs/DEVELOPMENT_LOG.md)，仓库级 agent 维护规则见 [`AGENTS.md`](AGENTS.md)。
