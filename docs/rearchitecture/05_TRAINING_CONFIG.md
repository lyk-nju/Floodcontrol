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
  cold_start_replay: 0.1
  k_schedule: [[0, 1], [100000, 2], [200000, 3], [300000, 5]]
  teacher_replay: {2: 0.1, 3: 0.1, 5: 0.1}

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
    dense_probability: 1.0
    waypoint_probability: 0.0
    goal_probability: 0.0
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
    enabled: false
    steps: 10000
```

`max_frames=200`与`window.max_tokens=50`共同定义每个sample最多10秒的parent窗口；短动作保留自然长度，长动作随机裁50 tokens，batch仅右padding。`chunk_size`与`window.generation_tokens`必须同为5。每个sample独立采样generation start，由此得到`H_i`和`F_i`，唯一预算是`H_i+5+F_i<=50`；不再存在独立history上限。`H=0`严格表示true cold start，只允许parent从真实序列第0 token开始，并要求VAE context为0、previous-root无效；中间parent的合法范围从`H=1`开始。`max_horizon_token=45`是future XZ采样上限，而不是固定长度：每个sample从0到扣除`K-1` rollout余量后的真实可用上限均匀采样，并在整次rollout内保持不变；因此K=1的完整cold窗口覆盖0～45，而靠近末端或K更大的窗口自动使用更小上限。`self_forcing`块是全部训练唯一的K课程入口：普通训练显式写成K=1，不再由`enabled`或phase字段切出另一条入口。`cold_start_replay=0.1`在K选择前独立应用于整个训练过程，使用K=1 ideal target并覆盖首次commit全部runtime denoise阶段与动态1–5 token visibility；它不执行前置no-grad solver rollout。K>1阶段会从同一预算预留`K-1`个rollout token，并自动过滤短于`5+K-1` tokens的训练sample。正式baseline使用dense XZ采样；constraint dropout与text dropout保持独立，从而覆盖joint/text/constraint/history四种条件组合。

正式validation的rolling buffer同样固定为50 tokens，使训练parent与训练期部署模拟使用同一容量。独立Web/runtime实验若覆盖其他buffer长度，必须在专用配置和报告中明确标记，不能反向改变50-token训练合同。

`k_schedule`的每一行是`[start_global_step,K]`：首行必须为`[0,1]`，step和K必须严格递增，所有stage必须落在`trainer.max_steps`内，`teacher_replay`只为K>1配置。启动入口会拒绝遗留`enabled/phase_start_step/phase_steps`字段，并校验最大K窗口预算。所有正式配置继承同一`[0,1]→[100000,2]→[200000,3]→[300000,5]`课程；它只依赖checkpoint恢复的absolute `global_step`，不因设备、数据集或路径覆盖而变化。

WandB的key/entity仍由共享环境配置提供，但实验级project由各训练配置所有：`vae*.yaml`显式使用`VAE_Flood`，`ldf*.yaml`显式使用`Floodcontrol`，避免共享路径配置改变某一类实验的冻结recipe。

`configs/ldf_base.yaml`是模型、50-token窗口、K课程、loss、optimizer与评测语义的唯一来源；`configs/ldf.yaml`只覆盖远端路径、8卡、batch与worker，`configs/ldf_s5.yaml`只覆盖本机路径、单卡资源和本地评测开关，`configs/ldf_multi.yaml`只覆盖HumanML3D+BABEL数据源与联合文本表。普通loss validation、dense-XZ/video和完整T2M均可走同一DDP作业。LDF使用EMA并从独立checkpoint加载冻结VAE；VAE不进入LDF optimizer、EMA或checkpoint，UMT5不进入训练进程。root statistics和离线T5表都属于数据产物，因此路径统一放在`data.root_stats_path`和`data.text_embeddings_path`；`data.text_meta_paths`只指向处理后数据集级`all.txt`。resume比较内容身份、数值statistics、模型结构和训练合同，不比较绝对路径，因此同一资产复制到另一服务器仍可恢复。

训练DataLoader内部固定使用20-frame（1秒）长度bucket减少batch右侧padding。这是20 FPS项目合同下的加载实现细节，不改变crop或窗口语义，因此不暴露为正式YAML实验参数。DDP时sampler先构造`per_device_batch_size * world_size`的全局同长度batch，再为每个rank切分自己的per-device batch；不足一个全局batch的bucket会确定性重复少量样本，使所有rank拥有完全相同的step数。validation使用无重复的strided rank shard。由于这两条sampler都由项目显式拥有，Trainer固定`use_distributed_sampler: false`，禁止Lightning再次注入一层DistributedSampler。

每次LDF fit启动都会打印per-GPU显存预算。启动报告分别列出当前参数与buffer、梯度、Adam/AdamW moment、EMA shadow及可能的DDP通信bucket，并用设备总显存减去这些固定项，显示留给activation、CUDA context、allocator fragmentation和kernel workspace的余量。该值是保守的固定显存预算，不伪装成无法静态确定的硬上限。sanity check结束后会重置CUDA peak；首个完整训练step（包括backward和首次optimizer state分配）打印所有rank中的最大`allocated/reserved`实测值，后续validation或generation刷新峰值时继续报告，fit结束时输出整轮累计峰值。

正式YAML不再暴露`contract_validation_every_n_steps`。`debug: false`时只执行structure validation；需要定位动态合同问题时设置`debug: true`，代码会在debug路径执行完整plan/input检查并承担相应GPU同步开销。

canonical root statistics只通过HumanML配置与正式span sampler生成一次：

```bash
python -m tools.compute_ldf_root_stats --config configs/ldf_s5.yaml
```

工具默认写入HumanML配置的`data.root_stats_path`，读取独立`root_statistics`块中的50-token、C=5、uniform-legal-history anchor与uniform-yaw协议。它不读取cold replay或K schedule，因此这些策略调整不触发重算。`ldf_multi.yaml`直接引用同一文件，不再根据BABEL mixture重算或切换归一化尺度。`train_ldf.py`除检查所有外部文件外，还校验50-token总预算、active chunk一致性、XZ horizon上限、课程和两种dropout。无法确认来源的旧root stats只需按冻结协议重新生成一次。

普通loss validation每1,000 steps运行。teacher-continuation固定覆盖early/middle/late H；self-forcing validation始终使用统一课程`k_schedule`中的最大K，不再维护第二套validation history/K参数。dense XZ每5,000 steps读取`data.test_probe_meta_paths.dense_xz`指定的小型split；远端配置启用T2M，本机S5配置默认关闭。T2M通过`validation.t2m.cfg_mode: nocfg`固定使用joint文本条件的单分支forward，不继承轨迹实验的全局separated CFG；dense XZ仍使用模型全局CFG设置。生成评测支持固定buffer `stream`与真实窗口滚动`rolling`并只使用EMA模型；当前实验配置只启用`stream`，需要同周期比较时可将modes改为`[stream, rolling]`。评测语义、指标和输出目录见[`07_LDF_TRAINING_EVALUATION.md`](07_LDF_TRAINING_EVALUATION.md)。

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
