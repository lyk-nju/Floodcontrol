# 05 训练配置与实验矩阵

状态：`TRAJECTORY_CONDITIONED_TEACHER_BASELINE_READY`

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
  min_frames: 20
  max_frames: 200

model.params:
  chunk_size: 5

self_forcing:
  enabled: true
  phase_start_step: 100000
  phase_steps: 200000
  k_schedule: [[0.0, 2], [0.5, 5]]
  teacher_replay: {2: 0.0, 5: 0.0}

text_encoder:
  text_len: 128

training:
  text_dropout: 0.1
  constraint_dropout: 0.1
  window:
    max_tokens: 50
    generation_tokens: 5
    sampling: random_generation_start
  max_horizon_token: 45
  constraint_sampling:
    dense_probability: 0.5
    waypoint_probability: 0.25
    goal_probability: 0.25
    max_waypoint_count: 4

lr_scheduler:
  target: diffusers.optimization.get_cosine_schedule_with_warmup
  params:
    num_warmup_steps: 5000
    num_training_steps: ${trainer.max_steps}

validation:
  generation:
    enabled: true
    steps: 5000
    modes: [stream]
    num_runs: 3
    video_samples: 4
    max_horizon_token: 10
    rolling:
      window_tokens: 50
  dense_xz:
    enabled: true
    probe: dense_xz
    segment_frames: 20
  t2m:
    enabled: true
    steps: 10000
```

`max_frames=200`与`window.max_tokens=50`共同定义每个sample最多10秒的parent窗口；短动作保留自然长度，长动作随机裁50 tokens，batch仅右padding。`chunk_size`与`window.generation_tokens`必须同为5。每个sample独立采样generation start，由此得到`H_i`和`F_i`，唯一预算是`H_i+5+F_i<=50`；不再存在独立history上限。`H=0`严格表示true cold start，只允许parent从真实序列第0 token开始，并要求VAE context为0、previous-root无效；中间parent的合法范围从`H=1`开始。`max_horizon_token=45`是future XZ的最大时间范围，实际可见范围为`min(45,F_i)`，并不表示每次都提供45个观测。首轮teacher baseline使用K=1；self-forcing K步会从同一预算预留`K-1`个rollout token，并自动过滤短于`5+K-1` tokens的训练sample。LDF在300k训练内使用5k-step linear warmup后的cosine decay。约束按50% dense trajectory、25% 1–4个frame-level waypoints（每个waypoint是一帧XZ观测）、25%单一future goal采样；constraint dropout与text dropout独立覆盖四种CFG条件组合。

`phase_start_step`是硬启动边界而不只是progress原点：此前固定K=1，到达边界后才采样schedule与teacher replay。当前300k recipe对应`0–100k: K=1`、`100k–200k: K=2`、`200k–300k: K=5`，并将replay概率设为0以保持严格分段。启动入口会一次性校验phase非负/正长度、threshold位于`[0,1]`且严格递增、K严格递增且不小于2、replay keys与schedule K完全一致且概率合法，以及最大K的窗口预算。该进度使用checkpoint中的absolute `global_step`；若从非零step完整resume，分段不会相对新进程重新计时。

WandB的key/entity仍由共享环境配置提供，但实验级project由各训练配置所有：`vae*.yaml`显式使用`VAE_Flood`，`ldf*.yaml`显式使用`Floodcontrol`，避免共享路径配置改变某一类实验的冻结recipe。

`configs/ldf.yaml`是HumanML3D teacher baseline；`configs/ldf_multi.yaml`拼接HumanML3D+BABEL，并与baseline共享HumanML3D canonical root statistics、VAE physical/latent statistics和模型合同。这样无论multi模型从头训练还是从HumanML checkpoint继续训练，normalized root语义都不会随数据mixture变化；multi配置只切换Dataset与包含BABEL caption的联合T5表。两个入口默认使用单卡Lightning，也支持将`trainer.devices`设为多卡并使用`strategy: ddp`；普通loss validation、dense-XZ/video和完整T2M均走同一DDP作业，不需要关闭训练期生成评测。LDF使用EMA并从独立checkpoint加载冻结VAE；VAE不进入LDF optimizer、EMA或checkpoint，UMT5不进入训练进程。root statistics和离线T5表都属于数据产物，因此路径统一放在`data.root_stats_path`和`data.text_embeddings_path`；`data.text_meta_paths`只指向处理后数据集级`all.txt`，后者覆盖全部正式split，而不依赖当前训练或评测配置。`tools/pretokenize_t5_text.py`据此生成包含空文本、完整数据集caption和离线内容`content_id`的表。resume同时校验路径、数值statistics与文本内容身份。

训练DataLoader内部固定使用20-frame（1秒）长度bucket减少batch右侧padding。这是20 FPS项目合同下的加载实现细节，不改变crop或窗口语义，因此不暴露为正式YAML实验参数。DDP时sampler先构造`per_device_batch_size * world_size`的全局同长度batch，再为每个rank切分自己的per-device batch；不足一个全局batch的bucket会确定性重复少量样本，使所有rank拥有完全相同的step数。validation使用无重复的strided rank shard。由于这两条sampler都由项目显式拥有，Trainer固定`use_distributed_sampler: false`，禁止Lightning再次注入一层DistributedSampler。

每次LDF fit启动都会打印per-GPU显存预算。启动报告分别列出当前参数与buffer、梯度、Adam/AdamW moment、EMA shadow及可能的DDP通信bucket，并用设备总显存减去这些固定项，显示留给activation、CUDA context、allocator fragmentation和kernel workspace的余量。该值是保守的固定显存预算，不伪装成无法静态确定的硬上限。sanity check结束后会重置CUDA peak；首个完整训练step（包括backward和首次optimizer state分配）打印所有rank中的最大`allocated/reserved`实测值，后续validation或generation刷新峰值时继续报告，fit结束时输出整轮累计峰值。

正式YAML不再暴露`contract_validation_every_n_steps`。`debug: false`时只执行structure validation；需要定位动态合同问题时设置`debug: true`，代码会在debug路径执行完整plan/input检查并承担相应GPU同步开销。

canonical root statistics只通过HumanML配置与正式span sampler生成一次：

```bash
python -m tools.compute_ldf_root_stats --config configs/ldf.yaml
```

工具默认写入HumanML配置的`data.root_stats_path`，复用自然parent窗口、逐样本均匀H、固定5-token active band和相同anchor分布。`ldf_multi.yaml`直接引用同一文件，不再根据BABEL mixture重算或切换归一化尺度。`train_ldf.py`除检查所有外部文件外，还校验50-token总预算、active chunk一致性、XZ horizon上限和两种dropout。

普通loss validation每1,000 steps运行。teacher-continuation固定覆盖early/middle/late H；启用训练self-forcing后，validation自动使用训练`k_schedule`中的最大K，不再维护第二套validation history/K参数。dense XZ每5,000 steps读取`data.test_probe_meta_paths.dense_xz`指定的小型split；HumanML T2M每10,000 steps遍历完整`val_meta_paths`。生成评测支持固定buffer `stream`与真实窗口滚动`rolling`并只使用EMA模型；当前实验配置只启用`stream`，需要同周期比较时可将modes改为`[stream, rolling]`。评测语义、指标和输出目录见[`07_LDF_TRAINING_EVALUATION.md`](07_LDF_TRAINING_EVALUATION.md)。

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
