# LDF训练配置

## 1. 配置继承

`configs/ldf_base.yaml`是模型、窗口、课程、loss、optimizer和评测语义的唯一来源。子配置通过：

```yaml
base_config: ldf_base.yaml
```

继承。加载顺序是paths → recursive base → child → CLI override；base路径相对当前YAML解析，循环继承会明确失败，run snapshot保存完整resolved config。

职责固定为：

```text
ldf_base.yaml  所有训练/模型语义
ldf.yaml       远端路径、8卡、batch、worker
ldf_s5.yaml    本机路径、单卡资源、本地评测开关
ldf_multi.yaml HumanML+BABEL数据源与联合文本表
```

## 2. 正式默认

```yaml
training:
  text_dropout: 0.1
  constraint_dropout: 0.2
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

self_forcing:
  cold_start_replay: 0.1
  cold_start:
    persistent_probability: 0.5
    rollout_commits: 2
  k_schedule:
    - [0, 1]
    - [100000, 2]
    - [200000, 3]
    - [300000, 5]

model:
  params:
    cfg_mode: joint
    cfg_scale_joint: 3.0
    root_prediction_type: x0
    body_prediction_type: velocity

optimizer:
  target: AdamW
  params:
    lr: 1.0e-4
```

T2M evaluation显式使用joint CFG并继承scale 3.0。50-token parent、active C=5、Dynamic Future和K课程不在remote/local/multi中重复。

## 3. VAE配置

LDF配置只提供VAE checkpoint和网络结构：

```yaml
vae:
  checkpoint_path: /path/to/step_299999.ckpt
  params:
    latent_dim: 128
    hidden_dim: 512
    encoder_layers: 6
    decoder_layers: 6
    kernel_size: 3
    dropout: 0.0
```

不提供`motion_stats_path`或任何latent stats路径。公共loader从checkpoint恢复EMA参数及`body_cont_mean/std`、`local_root_mean/std`四个buffers。

Data配置只包含dataset、split、artifact/text、T5表和资源参数；不包含root statistics。LDF的root5是physical值，latent是raw deterministic `mu`。

## 4. Loss配置

```yaml
loss:
  root_weight: 1.0
  body_weight: 1.0
  rollout_weight: 1.0
  root_boundary_weight: 0.0
```

Root x0内部的XZ/height/heading三块默认等权。solver固定运输到raw x0，Root→Body使用projected physical view，persistent root只在commit时投影。配置中不存在投影模式开关、off-path beta floor、heading beta floor、cosine/vector heading auxiliary或统一`prediction_type`。

## 5. Resume合同

新LDF checkpoint只硬性比较：

- Root/Body Transformer网络结构；
- `root_prediction_type/body_prediction_type`；
- 实际加载的EMA VAE参数与physical buffers；
- strict LDF state dict。

window、K课程、cold比例、loss权重、CFG、batch、worker和绝对路径不作为resume门闩。文本表身份保存为诊断metadata，但不是训练策略锁。旧LDF checkpoint缺少新合同会明确失败；本次Root-x0训练默认从头开始。

## 6. 启动

```bash
# remote
python train_ldf.py --config configs/ldf.yaml

# local
python train_ldf.py --config configs/ldf_s5.yaml

# multi dataset
python train_ldf.py --config configs/ldf_multi.yaml

```

启动校验检查50/C5窗口预算、horizon、dropout、constraint采样和prediction类型；不要求历史root/latent NPZ。
