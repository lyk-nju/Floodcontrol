# Floodcontrol

Floodcontrol 是基于 FloodDiffusion 在线 diffusion-forcing 范式重建的动作生成仓库。新版采用 ARDY 风格的结构化生成状态：

```text
root_motion + latent_motion
        -> Root Transformer
        -> clean local_root_motion
        -> Body Transformer
```

当前正式合同是：

- strict `4 frames / token` 时间协议；
- body-only causal VAE核心、Body259 codec、分组loss和仅含cache的显式`VAEDecoderState`已经落地；
- `HybridMotion`、typed condition、Root/Body non-causal LDF、constraint CFG以及dense trajectory / sparse waypoint / future goal XZ训练bridge已经落地；
- `generate/stream_generate_step`已经使用显式Hybrid流式状态；多token世界坐标生成统一由`InferenceSession.generate()`负责；
- HumanML3D/BABEL的独立Body259 artifact、HumanML-only VAE physical statistics与同构multi-dataset训练入口；
- `HybridMotion.root_motion`直接保存physical root5，`latent_motion`直接保存deterministic raw VAE `mu`；
- Root Transformer预测physical `x0`，Body Transformer预测raw-latent velocity；两者通过统一clean endpoint与solver velocity进入同一流式求解器；
- 分主题架构文档和契约测试。

旧的附加控制网络、外置轨迹编码器、外置root planner以及263D/trajectory7 runtime已经从新版仓库物理删除。HumanML3D/BABEL Dataset只返回完整sample，VAE与LDF各自在training data层构造任务视图。新版Body259对世界XYZ平移和全局yaw严格不变，Root5独占世界XYZ与绝对heading；旧Body265 artifacts、VAE和LDF checkpoint均保留但不兼容。新VAE训练完成后，LDF、评测和runtime通过公共loader恢复EMA网络与四个physical body/local-root buffers。LDF在线编码raw deterministic `mu`，Root状态不归一化，也不生成root/latent statistics。HumanML caption在整段token上重复，BABEL区间被编译为token prompt timeline；每个motion token只cross-attend自身prompt。

## 当前可验证范围

```bash
python -m pytest tests -q
```

迁移服务器时可以从FloodDiffusion同构的`HumanML3D`与`BABEL_streamed`源目录
一体化重建motion、VAE physical statistics和UMT5表：

```bash
python -m tools.prepare_training_assets pre-vae \
  --raw-data-root /path/to/raw_data \
  --deps-root /path/to/deps \
  --workers 16 \
  --t5-devices 0,1,2

python -m tools.prepare_training_assets verify \
  --raw-data-root /path/to/raw_data \
  --deps-root /path/to/deps \
  --vae-checkpoint /path/to/body_vae/last.ckpt
```

`pre-vae`不需要VAE checkpoint；`verify`读取一个自包含EMA VAE checkpoint并检查LDF启动资产。完整输入、输出和恢复协议见
[`02_DATA_PIPELINE.md`](docs/rearchitecture/02_DATA_PIPELINE.md)。

测试覆盖typed condition、active/future XZ、四种Root/Body prediction组合、CFG raw-output组合顺序、逐token文本隔离、四帧VAE raw-mu contract、因果性、offline/stream parity、persistent cold、三角Hybrid stream和snapshot恢复。正式LDF训练窗口固定为50 tokens/200 frames、active chunk为5；训练和T2M均使用joint CFG，默认scale为3。Web的四帧chunk runtime已经接入`InferenceSession`，模型加载在正式LDF checkpoint冻结前明确抛出`BLOCKED_ON_LDF_CHECKPOINT`。

LDF训练支持单卡和Lightning DDP。多卡时普通validation、dense-XZ轨迹/视频与完整HumanML T2M评测都会按sample分片；只有rank 0写全局summary和W&B。`configs/ldf.yaml`固定为远端8卡/resume配置，`configs/ldf_s5.yaml`单独保存本机S5配置；两者都显式拥有自己的路径，不依赖同一份`paths_default.yaml`切换服务器语义。

```bash
# 远端服务器
python train_ldf.py --config configs/ldf.yaml

# 本机S5
python train_ldf.py --config configs/ldf_s5.yaml
```

设计入口见 [`docs/rearchitecture/README.md`](docs/rearchitecture/README.md)。已经完成的代码与协议修改记录在 [`docs/DEVELOPMENT_LOG.md`](docs/DEVELOPMENT_LOG.md)，仓库级 agent 维护规则见 [`AGENTS.md`](AGENTS.md)。
