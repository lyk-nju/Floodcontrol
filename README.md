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
- body-only causal VAE核心、body265 codec、分组loss和显式`VAEDecoderState`已经落地；
- `HybridMotion`、typed condition、Root/Body non-causal LDF和constraint CFG已经落地；
- `generate/stream_generate/stream_generate_step`已经使用显式Hybrid流式状态；
- HumanML3D/BABEL的独立body265 artifact、联合statistics与同构multi-dataset训练入口；
- 分主题架构文档和契约测试。

旧的附加控制网络、外置轨迹编码器和外置root planner已经从新版仓库物理删除。当前里程碑保证hybrid LDF与BodyVAE模型核心、HumanML263到body265的显式转换、合成张量流式decode和LDF heading bridge。第一版明确使用HumanML-style预处理提供的IK-derived rotations，并在artifact中标记来源；本地`HumanML3D_motion`、`BABEL_motion`以及两者联合train statistics已经构建完成，真实LDF训练与Web生成仍需正式VAE checkpoint、latent artifacts和commit-time decoder事务接线。

## 当前可验证范围

```bash
python -m pytest tests -q
```

测试覆盖typed condition、Root/Body forward、constraint CFG、四帧VAE contract、HumanML恢复、随机yaw一致性、因果性、offline/stream parity、三角Hybrid stream和snapshot恢复。`train_vae.py`在缺少motion artifact split TXT/statistics时fail-fast；`train_ldf.py`与Web模型加载继续明确抛出`BLOCKED_ON_BODY_VAE`，直到正式VAE/latent artifact接线完成。

本地数据、依赖和输出目录默认分别为`./data`、`./deps`和`./outputs`，也可以通过`FLOODCONTROL_DATA`、`FLOODCONTROL_DEPS`和`FLOODCONTROL_OUTPUTS`覆盖。

设计入口见 [`docs/rearchitecture/README.md`](docs/rearchitecture/README.md)。已经完成的代码与协议修改记录在 [`docs/DEVELOPMENT_LOG.md`](docs/DEVELOPMENT_LOG.md)，仓库级 agent 维护规则见 [`AGENTS.md`](AGENTS.md)。
