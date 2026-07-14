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
- causal 1D VAE 的过渡实现仍保留，但尚未接入新版LDF；
- `HybridMotion`、typed condition、Root/Body non-causal LDF和constraint CFG已经落地；
- `generate/stream_generate/stream_generate_step`已经使用显式Hybrid流式状态；
- HumanML3D/BABEL 基础读取与 motion recovery；
- 分主题架构文档和契约测试。

旧的附加控制网络、外置轨迹编码器和外置root planner已经从新版仓库物理删除。当前里程碑只保证hybrid LDF模型核心与合成张量流式测试；真实训练和Web生成将在strict-4 body VAE、数据协议与commit-time decoder事务接入后恢复。完整旧实现仍可在`../FloodNet`中查阅。

设计入口见 [`docs/rearchitecture/README.md`](docs/rearchitecture/README.md)。已经完成的代码与协议修改记录在 [`docs/DEVELOPMENT_LOG.md`](docs/DEVELOPMENT_LOG.md)，仓库级 agent 维护规则见 [`AGENTS.md`](AGENTS.md)。
