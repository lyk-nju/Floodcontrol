# 05 训练配置与实验矩阵

状态：`TEACHER_BASELINE_CONFIGURED / ROOT_STATS_REFRESH_REQUIRED`

本文只有在模型、数据、runtime 和训练方法文档的相关条目冻结后才开始填写。它记录可执行配置，不反向定义语义协议。

## 本文将记录什么

- tokenizer/VAE：架构规模、latent dim、KL/FSQ、optimizer、LR、batch、clip length、loss weights、训练步数和 checkpoint/eval 频率。
- LDF：root/body Transformer 规模、schedule、denoise steps、history/generation/future 长度、CFG/dropout、optimizer、EMA 和训练阶段。
- self-forcing：rollout 长度、采样概率曲线、detach/TBPTT、显存预算。
- 数据：cache version、stats identity、sampler 参数和 worker/prefetch 配置。
- 实验：单变量消融、seed、GPU/时间预算、验收指标和停止条件。

## 配置值的标签

- `DERIVED`：由冻结协议计算得到，例如 `frames_per_token=4` 后的 shape。
- `BASELINE`：来自 FloodDiffusion 或 ARDY 的可复现实验起点。
- `TUNABLE`：需要消融，不得写成协议事实。
- `RESOURCE`：由显存、吞吐或硬件约束决定。

## LDF active-band kernel（当前实现）

```yaml
data:
  min_frames: 40
  max_frames: 200
  cold_start_probability: 0.1

model.params:
  chunk_size: 5

self_forcing:
  enabled: false
  k_schedule: [[0.0, 2], [0.4, 3], [0.7, 5]]
  teacher_replay: {2: 0.2, 3: 0.1, 5: 0.1}

text_encoder:
  text_len: 128

training:
  text_dropout_probability: 0.1
```

`min/max_frames`定义batch共享source span S，`length_bucket_frames`控制训练batch的长度分桶，`chunk_size`是active band唯一来源，不在training配置中重复。首轮teacher baseline明确设置`self_forcing.enabled: false`；K schedule与teacher replay只在后续显式开启时生效，进度由`phase_start_step/phase_steps`单独定义。text长度128沿用FloodDiffusion论文设置，dropout和root/body loss权重仍是`TUNABLE`。fixed absolute-token noise、固定初始H anchor、逐token prompt可见性和只替换最左active token是训练协议。

`configs/ldf.yaml`是HumanML3D teacher baseline；`configs/ldf_multi.yaml`从头训练同一LDF主干并拼接HumanML3D+BABEL，使用相同VAE/T5合同，但要求先计算混合训练集自己的root statistics和预编码文本表。两个入口都使用单卡Lightning、LDF EMA，并从独立checkpoint加载冻结VAE；VAE不进入LDF optimizer、EMA或checkpoint，UMT5不进入训练进程。`text_embeddings_path`必须由`tools/pretokenize_t5_text.py`生成且包含空文本及训练/验证全部caption。

root statistics必须通过同一配置的数据mixture与span sampler生成：

```bash
python -m tools.compute_ldf_root_stats --config configs/ldf.yaml
python -m tools.compute_ldf_root_stats --config configs/ldf_multi.yaml
```

工具默认写入各配置的`root_stats_path`，并复用min/max span、chunk size、cold-start概率和短样本过滤。确认统计产物后，人工把对应配置的`status`从`root_statistics_required`切到`training_ready`；`train_ldf.py`不会仅凭同名旧文件存在就静默启动。

## 开始填写前的门槛

- 模型 I/O shape 全部确定。
- full-clip/cache/window 数据产品确定。
- runtime history/active/frontier 最大长度确定。
- scheduler 方程和训练阶段确定。
- 每个训练任务的验收指标确定。

## 首批配置文档建议

协议冻结后，不把所有实验继续堆在本文；按任务生成独立配置说明：

```text
configs/tokenizer_v1.yaml        + tokenizer_v1.md
configs/ldf_teacher_v1.yaml      + ldf_teacher_v1.md
configs/ldf_self_forcing_v1.yaml + ldf_self_forcing_v1.md
configs/eval_protocol_v1.yaml    + eval_protocol_v1.md
```
