# LDF 模型层实现设计

状态：`MODEL_CORE_IMPLEMENTED`。公共合同、Root/Body主干、CFG和Hybrid流式状态已经落地；Body VAE、真实数据、训练与Web runtime仍待后续接线。具体正式网络规模、loss权重和训练schedule仍由实验决定。

## 0. 2026-07-14模型核心里程碑

- 已实现`HybridMotion/LDFInput/LDFCondition/LDFPrediction/LDFStreamState`与三个condition创建函数。
- 已实现`RootTransformer/BodyTransformer/LDF`，并保留v-predict和三角调度。
- CFG先组合唯一root velocity，再从唯一clean root派生local root，最后运行共享该root的Body CFG branches。
- Body Stage不直接读取raw current/future root constraints；root constraint只通过Root Stage的clean/local root影响body。
- 已物理删除ControlNet、专用轨迹编码器、FlexTraj、tiny专用模型、外置root planner和RootPlan runtime闭环。
- 真实训练与Web入口在Body VAE/data/runtime接入前明确fail-fast。

本文只回答新版 LDF 在代码层怎样组织，以及怎样在不丢失 FloodDiffusion 流式状态机的前提下，用 Root/Body 两阶段主干接管旧 FloodNet ControlNet 路径。物理 root 定义、VAE 协议、数据构造、active-window 坐标事务和训练超参数分别由其他文档负责。

## 1. 已冻结的实现原则

- `LOCKED`：公开模型文件继续叫 `models/diffusion_forcing_wan.py`，公开模型类叫 `LDF`。
- `LOCKED`：Root Transformer 与 Body Transformer 都直接实现在 `diffusion_forcing_wan.py` 中；两阶段是 `LDF` 内部结构，不出现在公共模型名或额外 `two_stage` 文件名中。
- `LOCKED`：公共 hybrid motion 字段统一为 `root_motion` 与 `latent_motion`。
- `LOCKED`：`local_root_motion` 是从 clean `root_motion` 派生的 condition，不属于 `HybridMotion` 的生成字段。
- `LOCKED`：LDF 的结构、条件和流式状态 dataclass，以及条件编译纯函数，统一放在 `utils/conditions/ldf.py`；不新建 `contracts.py`、`struct.py` 或独立 `conditioning.py`。
- `LOCKED`：保留 FloodDiffusion 的三角 per-token noise、v-predict、逐 token commit、rolling window、`stream_generate_step` 和 snapshot/restore 语义。
- `LOCKED`：删除 ControlNet 旁路，但保留并迁移 text/constraint CFG。
- `LOCKED`：删除 FlexTraj 专用 attention；future constraints 使用普通 non-causal Transformer、validity mask 和显式 position IDs。
- `LOCKED`：Root/Body LDF 不使用跨 commit attention KV cache；VAE decoder causal state 是独立协议。

## 2. 最终文件边界

```text
models/
├── diffusion_forcing_wan.py
│   ├── RootTransformer
│   ├── BodyTransformer
│   └── LDF
├── tools/
│   ├── attention.py
│   ├── wan_model.py
│   ├── t5.py
│   ├── tokenizers.py
│   └── wan_vae_1d.py
└── vae_wan_1d.py

utils/
└── conditions/
    └── ldf.py
```

### `models/diffusion_forcing_wan.py`

负责所有模型参数和完整 LDF 调用路径：

```text
RootTransformer
BodyTransformer
LDF.forward
LDF.generate
LDF.stream_generate
LDF.stream_generate_step
LDF.init_stream_state
LDF.create_stream_snapshot
LDF.create_stream_state_from_snapshot
三角 scheduler update
CFG 分支 forward 与结果组合
```

### `utils/conditions/ldf.py`

只依赖 PyTorch/dataclass 等基础模块，不反向导入 `models`。它负责：

```text
HybridMotion / LDFInput / LDFCondition / LDFPrediction / LDFStreamState
shape、dtype、device、mask 和时间所有权校验
create_window_condition：absolute timeline -> active-window 条件裁剪
create_ldf_condition：frame -> 4-frame token 对齐，编译 current/future constraint tensors
create_cfg_condition：生成 text-only / constraint-only / history-only CFG conditions
训练 condition dropout 与推理空条件的一致语义
```

可学习的 projection 不放在这里，仍属于 `LDF`、`RootTransformer` 或 `BodyTransformer`。

### `models/tools/attention.py`

只保留通用 attention kernel：

```text
flash_attention
attention / SDPA fallback
```

它不感知 root、trajectory、future goal、token region 或 ControlNet。

### `models/tools/wan_model.py`

保留通用 Wan building blocks、文本/时间 embedding、RoPE 和 Transformer block。它可以支持：

```text
non-causal self-attention
有效 token lengths/padding
显式 position IDs
sample-aligned 或 token-aligned text context
```

它不再支持 `traj_emb`、`traj_seq_lens`、`traj_token_mask`、`latent_pad_len`、`traj_pad_len` 或 `controlnet_residuals`。

## 3. 公共命名与最小结构

### `HybridMotion`

```python
@dataclass
class HybridMotion:
    root_motion: torch.Tensor    # [B, T, 4, 5]
    latent_motion: torch.Tensor  # [B, T, D]
```

含义：

- `root_motion`：每 token 四个连续物理帧的显式 `[x,y,z,cos(yaw),sin(yaw)]` root；具体处于 world/session/translation-anchor 哪个 frame 由强类型坐标元数据记录，不编码进字段名。
- `latent_motion`：body tokenizer/VAE 的 latent motion token；显式 root 不重复藏进该字段。

`HybridMotion` 可以承载 clean motion、noise、velocity 或 noisy motion；语义由外层字段名确定，例如：

```text
LDFInput.noisy_motion
LDFPrediction.velocity
LDFStreamState.noisy_motion
```

### `LDFInput`

```python
@dataclass
class LDFInput:
    noisy_motion: HybridMotion
    beta: torch.Tensor                 # [B,T], 0=clean, 1=pure noise
    history_mask: torch.Tensor         # bool [B,T]
    generation_mask: torch.Tensor      # bool [B,T]
    timeline_position_ids: torch.Tensor  # absolute long [B,T]
    rope_position_ids: torch.Tensor      # generation-centered long [B,T]
    previous_root_frame: torch.Tensor | None  # [B,5]
    condition: LDFCondition
```

不建立额外 `TokenRegions` 类型。history/generation 的更新所有权直接通过字段表达；future constraints 不属于 noisy motion，因此不放进这两个 mask。

两套position不能合并：`timeline_position_ids`是stream scheduler、commit和窗口裁剪使用的绝对坐标；`rope_position_ids`只供Transformer位置编码使用，并以当前第一个generation token为0。每个样本的两者必须相差同一个常数`rope_origin`。因此history RoPE位置为负、generation从0开始、future为正，与ARDY的generation-centered token index同构。

### `previous_root_frame`

不建立额外 `RootBoundary` 类型。`previous_root_frame[B,5]` 只是当前窗口第一物理帧之前的 root sample，用于 backward `LocalRootMotionCodec`：

```text
rolling window: 取窗口前一个 committed root frame
cold start:      复制当前 clean frame 0，使第一条差分为零并标记相应速度 feature invalid
```

它不是 SE(2) anchor、不是生成 token、也不参与 CFG。

### `LDFCondition`

第一版协议目标为：

```python
@dataclass
class LDFCondition:
    text_context: list
    text_null_context: list

    root_condition_value: torch.Tensor | None
    root_condition_mask: torch.Tensor | None

    body_condition_value: torch.Tensor | None
    body_condition_mask: torch.Tensor | None

    future_root_condition_value: torch.Tensor | None
    future_root_condition_mask: torch.Tensor | None
    future_timeline_position_ids: torch.Tensor | None
    future_valid_mask: torch.Tensor | None
```

第一版只为 future root trajectory 建立正式协议：`future_root_condition_value/future_root_condition_mask` 按四帧 root patch 表达，推荐形状为 `[B,N_future,4,5]`；`future_timeline_position_ids[B,N_future]` 使用absolute timeline坐标，必须位于当前motion window之后；Root Stage使用当前`rope_origin`把它转换为generation-centered future RoPE位置。`future_valid_mask[B,N_future]` 区分真实future token与batch padding，padding不能伪装成零值约束。

第一版不使用泛化的 `future_value/future_feature_mask/future_type_ids`。只有真正接入 future body、end-effector 等异构约束时，才扩展 typed future schema；不为尚未实现的类型提前引入公共协议。也不建立额外 `FutureGoalBatch` 类型，future 字段直接属于 `LDFCondition`。

### 单一 constraint 语义

`root_condition_value/root_condition_mask` 与 `body_condition_value/body_condition_mask` 是唯一的 current-window 空间条件。它们使用 ARDY/Kimodo 风格的 masked input view 与 condition projection：constraint branch 可以看到 observation-infused input 和 mask，text-only/history-only branch 则清空这些条件。

`LOCKED`：V1 不区分 `exact/soft`，不在 Root Stage 输出后 hard-replace clean root，也不从替换后的 clean root重算所谓 effective velocity。所有用户 trajectory、waypoint、heading 和稀疏 pose 都是模型内生的 constraint condition，通过 `cfg_scale_constraint` 调节影响。

ARDY 与 Kimodo 都使用统一的 `observed_motion + motion_mask`，在 denoiser 的只读输入视图中做 masked infill，并通过 separated CFG 决定哪个 branch 携带约束；它们都没有最终 clean-root hard-fusion contract。Floodcontrol保持同一所有权。

这里的 masked input view 不是旧 FloodNet root replacement：它不修改 persistent noisy state，不覆盖模型最终 clean root，不经过 VAE decoder，也不把替换结果 re-encode 回历史。它只是 constraint branch 的网络输入构造。

### Condition 创建函数

三个函数都是 `utils/conditions/ldf.py` 中无可学习参数、不得修改 persistent state 的纯数据函数：

```text
absolute condition timeline
        |
        v
create_window_condition()
        |
        v
create_ldf_condition()
        |
        v
create_cfg_condition()
        |
        v
Root/Body LDF forward
```

- `create_window_condition(...)`：根据 active-window 起点、长度和 lookahead 范围，选择 in-window 与 future constraints，丢弃过期项，并把时间索引转换为当前窗口坐标。返回值可以是模块内部 mapping，不增加公共 `WindowCondition` dataclass。
- `create_ldf_condition(...) -> LDFCondition`：完成 frame-to-token/phase 对齐、normalization contract 检查、dense root/body constraint tensors、future root packing、padding和所有 shape/mask 断言。
- `create_cfg_condition(...)`：从同一个 `LDFCondition` 创建 text-only、constraint-only、history-only 或 joint conditions。History、`previous_root_frame` 和 noisy state在所有 branches 中相同；text 与 current/future constraints按 branch置空。

`create_cfg_condition` 只创建条件，不组合模型输出；CFG 数学组合仍由 `LDF` 内部独立的 prediction composer 负责。

### `LDFPrediction`

```python
@dataclass
class LDFPrediction:
    velocity: HybridMotion
    clean_root_motion: torch.Tensor
    local_root_motion: torch.Tensor
```

`velocity.root_motion` 与 `velocity.latent_motion` 是 scheduler 唯一消费的网络结果。`clean_root_motion` 和 `local_root_motion` 用于 stage boundary、loss、诊断和 commit-time decoder condition。

### `LDFStreamState`

```python
@dataclass
class LDFStreamState:
    noisy_motion: HybridMotion
    current_step: int
    commit_index: int
    window_origin: int
    epoch: int
```

实现时还可包含确定性恢复所需的 length、noise metadata 或 RNG 信息，但不能包含 Root/Body Transformer KV cache 或 trajectory embedding cache。

## 4. `LDF` 的参数层级

```text
LDF
├── text_encoder
├── root_transformer
│   ├── history_input_projection
│   ├── generation_input_projection
│   ├── future_input_projection
│   ├── WanTransformerBlock × N_root
│   └── root_output_projection       -> 4×5
├── body_transformer
│   ├── history_input_projection
│   ├── generation_input_projection
│   ├── WanTransformerBlock × N_body
│   └── latent_output_projection     -> D
├── root/body observation projections
└── shared non-parameter logic
    ├── LocalRootMotionCodec
    ├── CFG condition creator / prediction composer
    └── triangular hybrid scheduler
```

Root 与 Body Transformer 参数独立。两者共享原始 text encoder 输出和 condition 语义，但不共享 Transformer blocks 或 stage-specific input/output projection。

## 5. 单分支 Root/Body forward

```text
LDFInput
  noisy root_motion + noisy latent_motion
  clean history
  beta / history_mask / generation_mask
  timeline_position_ids / rope_position_ids
  current/future constraints
                         |
                         v
                 RootTransformer
                         |
                  root velocity model
                         |
          clean root = root_t + beta * root_v
          physical heading manifold projection
                         |
                         v
             backward LocalRootMotionCodec
                         |
        training: stop_gradient(local_root_motion)
                         |
                         v
                 BodyTransformer
                         |
                 latent motion velocity
```

每个 denoising/flow step 都执行 Root first、Body second。Body Stage 接收本分支预测并处理后的 clean root 派生 condition，不接收另一条 trajectory prediction，也不使用 ControlNet residual。

## 6. CFG 接管方式

### 所有分支共享

```text
noisy HybridMotion
committed clean history
beta / region / position
previous_root_frame
坐标 frame metadata
```

这些状态不是 CFG 对象。

### ARDY/Kimodo separated 条件分支

```text
text-only:       real text, empty constraints
constraint-only: empty text, real constraints
history-only:    empty text, empty constraints
```

Root Stage先执行text/constraint/history branches并组合root velocity：

```text
root_v = root_v_history
  + cfg_scale_text       * (v_text       - v_history)
  + cfg_scale_constraint * (v_constraint - v_history)
```

组合后的root velocity恢复出唯一clean root并派生唯一local root。Body Stage随后执行自己的text/body-constraint/history branches，所有Body branches共享上述local root，再使用同一线性公式组合latent velocity。这样constraint CFG仍沿用ARDY/Kimodo的分支语义，但不会产生“最终body对应的不是最终root”的两阶段不一致。

heading manifold projection只生成供local-root codec和Body Stage使用的合法clean-root view，不修改scheduler消费的组合velocity，也不执行任何condition replacement。Body不直接读取raw root/future-root constraint token。

### Regular/joint 基线

```text
v = v_history
  + cfg_scale_joint * (v_text_constraint - v_history)
```

保留该模式用于联合条件基线和消融。公共配置字段为：

```yaml
cfg:
  mode: separated
  cfg_scale_text: <float>
  cfg_scale_constraint: <float>
  cfg_scale_joint: <float>  # mode=joint 时使用
```

具体 scale 数值属于训练/评测配置，不在模型结构文档中冻结；不继续暴露 `cfg_scale_traj` 作为最终名称。

## 7. 完整流式外壳的接管点

旧实现中 `generate`、`stream_generate` 与 `stream_generate_step` 都调用 `_denoise_with_cfg`。新版将这一唯一 seam 替换为：

```python
_predict_hybrid_with_cfg(ldf_input) -> LDFPrediction
```

以下外层概念保留：

```text
current_step / dt
chunk_size
start_index / end_index
triangular beta
commit_index
rolling active window
stream buffer metadata
snapshot / restore
```

更新从单一 latent tensor：

```text
generated += predicted_velocity * dt
```

变为共享 active slice 的两个字段：

```text
state.noisy_motion.root_motion[active]
    += prediction.velocity.root_motion[active] * dt

state.noisy_motion.latent_motion[active]
    += prediction.velocity.latent_motion[active] * dt
```

达到 clean/commit 条件后输出一个 `HybridMotion` token。runtime 使用 committed `root_motion` 派生 `local_root_motion`，再调用 causal VAE `decode_step(latent_motion, local_root_motion, decoder_state)`。LDF denoising loop 不调用 decoder。

## 8. Attention 清理合同

### 从 `models/tools/attention.py` 删除

```text
_flextraj_vl_vt
flextraj_query_valid
_normalize_traj_token_mask
flextraj_self_attn_bias
flextraj_sdpa_self_attention
_flextraj_flash_available
flextraj_flash_split_self_attention
flextraj_self_attention
```

以及对应 `__all__` exports。

### 从 `models/tools/wan_model.py` 删除

```text
flextraj_self_attention import
rope_apply_latent_traj
_prepare_traj_attn_mask
traj_enc_dim / traj_in_proj / traj_type_embed
traj_emb / traj_seq_lens / traj_token_mask
latent_pad_len / traj_pad_len 专用分支
controlnet_residuals 及逐 block residual injection
```

### 新增的通用能力

future constraints 需要通用的 explicit-position RoPE：

```text
apply_rope_with_position_ids(q_or_k, rope_position_ids)
```

模型将 `[history | generation | packed future constraints]` 作为普通有效 token 序列送入 non-causal Transformer。future token 的输出被忽略，不被 scheduler 更新或 commit；不再需要 trajectory-query/latent-key 的专用非对称 mask。

`create_window_condition()`从`window_origin + window_tokens`开始生成absolute future timeline IDs，不能在rolling后重新从`window_tokens`起算。future horizon不得反向扩大`HybridMotion`、`beta`、history/generation mask或Body Stage长度。

token-aligned text context 与 text embedding dedup 属于在线文本能力，不是 FlexTraj，应保留并去除其中仅服务 trajectory token 的分支。

## 9. 已执行的迁移顺序

### 阶段 A：结构与单分支核心

1. 在 `utils/conditions/ldf.py` 落地 dataclass、校验，以及 `create_window_condition/create_ldf_condition/create_cfg_condition`。
2. 在 `diffusion_forcing_wan.py` 实现 RootTransformer、BodyTransformer 和无 CFG 的单分支 forward。
3. 使用随机 `latent_motion` 完成 shape、v/x0 恒等式、heading、只读 masked input view 和 stage detach 测试，不等待最终 VAE。

### 阶段 B：CFG 与流式 scheduler

4. 实现 `create_cfg_condition` 的 regular/separated branches 与完整两阶段 branch batching。
5. 将三条旧 `_denoise_with_cfg` 调用迁移到 `_predict_hybrid_with_cfg`。
6. 将 generated buffer、update、commit 和 snapshot/restore 改为 `LDFStreamState`。
7. 验证固定噪声下 offline/stream step、一致 snapshot restore 和在线条件更新。

### 阶段 C：后续VAE/训练接线

8. 接入 causal Body VAE 与 commit-time decoder transaction。
9. 训练改为 root velocity 与 latent velocity 直接监督；删除旧 decoder trajectory auxiliary loss 主路径。
10. 训练 condition dropout 与推理 CFG 空分支复用 `create_cfg_condition` 的置空语义。

### 阶段 D：已完成的物理删除

经用户明确批准，本次模型核心里程碑在VAE接线前先删除了：

```text
WanControlNet
TrajectoryEncoder
ControlNet configs/checkpoint loading
legacy traj embedding/cache
FlexTraj attention
外置root planner/post-decode projection正式路径
```

训练与Web入口同步改为`BLOCKED_ON_BODY_VAE`，因此不会回退到旧VAE或留下伪兼容路径。后续恢复端到端能力时直接围绕新合同建设。

## 10. 最低验收测试

```text
HybridMotion root/latent shape 与时间轴一致
history beta=0 且 scheduler 不修改 history
Root/Body 使用同一 beta 和 active mask
root v -> clean root 恒等式
heading physical unit-circle projection
root/body constraint 只进入 constraint/joint 分支
constraint input view 不修改 persistent noisy state或最终 clean root
每条 CFG branch 的 Body Stage 使用该分支自己的 clean local root
joint 模式下 cfg_scale_joint=1 等于 joint conditional forward
text-only 不构造无意义 constraint branch
full-window 与 rolling-window local-root 边界一致
stream snapshot -> mutate -> restore 后确定性复现
无 ControlNet/TrajectoryEncoder/FlexTraj 活跃引用
VAE full decode 与 cached decode_step parity
```

## 11. 尚未在本文冻结

- Root/Body Transformer 的层数、宽度、head 数和是否同规模。
- `latent_motion` width、VAE/AE/FSQ 选择和 tokenizer loss。
- body observation projector 的最终 feature layout，以及何时扩展 root-only future schema。
- separated additive 是否作为最终训练/评测默认，或只作为 regular/joint 的补充模式。
- condition dropout 概率、root/latent loss 权重、self-forcing 和 scheduled training。
- runtime translation-anchor epoch 的切换策略。
