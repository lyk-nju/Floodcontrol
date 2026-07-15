# 01 模型结构与输入输出

状态：`LDF_BODY_VAE_AND_WEB_RUNTIME_IMPLEMENTED / LDF_TRAINING_OPEN`

## 设计顺序与文档顺序

本设计采用两遍法，不能把“文档怎样组织”误当成“架构怎样产生”。

### 第一遍：设计决策自上而下

先冻结新版系统必须具备的能力和模型总体结构，再让上层需求决定下层接口：

```text
产品/任务能力
    -> 总体生成范式
    -> 模块划分与职责
    -> 训练/推理总数据流
    -> 模块边界 I/O
    -> 底层表示、codec、mask 和 cache
```

这一遍先回答：为什么需要这些模块、每个模块负责什么、哪些变量必须由模型生成、哪些只是条件或 runtime 状态。此时可以暂不冻结所有维度和字段。

### 第二遍：实现协议自下而上

总体架构确定后，正式文档和代码接口再从基础类型向上组合：

```text
coordinate/value types
    -> root/body codecs
    -> tokenizer interfaces
    -> hybrid states
    -> two-stage model call
    -> streaming/cache contract
```

这样既能保证架构由上层需求驱动，又能保证最终实现没有含糊的 tensor、坐标系或隐式切片。

## 本文只回答什么

- 每一种物理量和张量的语义、shape、坐标系、归一化所有者和 mask。
- body tokenizer/VAE 的 encoder、decoder、streaming decode 接口。
- clean hybrid token、noisy hybrid token、history 和 future observation 的结构。
- LDF root stage 与 body stage 的信息流和输出参数化。
- VAE decoder causal state 与其他 persistent generation state 中允许保存什么，以及怎样与 token commit 做事务同步。

## 本文不回答什么

- 数据从哪个文件读取、怎样随机裁剪。
- active window 何时 rebase 或 buffer roll。
- loss 权重、优化器、训练步数和 batch size。
- 最终层数、宽度等实验超参数。

## 第一遍：自上而下的架构讨论顺序

1. 冻结目标能力与非目标：text-only motion、结构化 root 生成、body fidelity、精确/稀疏空间约束、长时 streaming、在线条件更新，以及 V1 明确不解决的能力。
2. 冻结总体生成范式：`explicit root + latent body` 是否是唯一 clean generative state；trajectory 是否完全从附加 ControlNet 条件提升为 root observation；是否保留四帧token协议与 persistent triangular diffusion forcing。
3. 冻结顶层模块图和责任边界：body tokenizer/VAE、root stage、body stage、observation projection、scheduler/update，以及哪些功能属于 runtime 而不是模型。
4. 冻结训练与推理的顶层 dataflow：clean history、noisy generation、future goals 如何进入模型；每个 denoising step 如何先 root 后 body；最终 commit 输出什么。
5. 冻结顶层模型调用接口和主要结构化返回值，不在这一步决定所有内部维度。
6. 对每个模块做失败模式检查：无 history cold start、转向、路径更新、长序列、padding/tail、constraint conflict、cache/rebase。

## 顶层目标与技术来源（第一轮结论）

### 建设目标

- `PROVISIONAL`：新版以原始 `FloodDiffusion` 为生成与流式调度基础，而不是在 `FloodNet` 的 ControlNet/root-refiner/post-decode correction 架构上继续打补丁。
- `PROVISIONAL`：必须保留真正的在线 streaming diffusion forcing：模型持有 clean history、partially noisy active band 和 pure-noise frontier，并逐 token commit。
- `LOCKED`：clean generative state 从单一 full-motion VAE latent 改为结构化的 `explicit root + latent body`。
- `LOCKED`：root 是 LDF 原生生成变量，也是 body tokenizer decoder 与 LDF body stage 的条件；它不再只是外部 trajectory supervision。
- `LOCKED`：采用 ARDY 的 interleaved root-first/body-second 思想，在每个 denoising/flow step 内先得到 clean root，再条件化 body prediction。
- `PROVISIONAL`：路径、waypoint、heading 等输入被建模为对 root 生成变量的 typed observations，而不是送入独立 ControlNet 的平行条件流。
- `LOCKED`：删除 ControlNet 只删除其旁路 residual 架构，不删除 constraint CFG；外部 trajectory/root goal 的 guidance 迁移为 ARDY/Kimodo 风格的主干内生 constraint CFG。
- `LOCKED`：V1 的 Root Transformer 与 Body Transformer 都是当前有限窗口上的 non-causal/bidirectional Transformer；不使用跨 commit attention KV cache。
- `LOCKED`：body tokenizer/VAE 在 token 时间轴上必须 causal；V1 保留 FloodDiffusion 的逐 token cached-decode思想，但不继承旧 `1+4n` 布局或旧 cache 张量协议。decoder 持久化 `VAEDecoderState`，encoder V1 不要求 persistent cache；必须验证 full decode 与 cached step decode parity。

### 三个代码库的角色

| 来源 | 新版中承担的角色 | 不直接继承的部分 |
|---|---|---|
| `FloodDiffusion` | 线性 flow/v-predict 候选、三角 per-token noise、persistent active window、逐 token commit、文本条件主干 | full-motion monolithic VAE latent 作为唯一生成状态 |
| `ARDY` | explicit root + latent body、body decoder 的 local-root condition、每步 root-first/body-second、typed sparse/future observations | block-wise DDIM 调度、完整照搬其表示维度或 FSQ 选择 |
| `FloodNet` | runtime timeline、状态事务、评测和已暴露的失败案例可作为工程参考 | ControlNet 是唯一 trajectory consumer、RootRefiner 外挂、post-decode root projection/re-encode 作为正式闭环 |

`Floodcontrol` 当前由 `FloodNet` 复制而来，但这只代表迁移起点，不代表新版架构必须保持 FloodNet 的模块边界或 checkpoint 兼容性。

## 代码落地状态

### 当前状态：新版LDF模型核心已落地

- `models/diffusion_forcing_wan.py` 已公开 `RootTransformer/BodyTransformer/LDF`，两阶段不出现在公共文件名或类名中。
- `generate/stream_generate/stream_generate_step` 已迁移为显式 `HybridMotion/LDFStreamState`，并通过合成张量的commit、rolling和snapshot/restore测试。
- 旧附加控制网络、专用轨迹编码器、专用attention、tiny模型和外置root planner已经物理删除；constraint CFG由主干接管。
- body VAE核心、唯一`humanml265`转换、全量本地motion artifacts/VAE statistics、首个300k EMA tokenizer、`body265 -> latent_motion`和显式`VAEDecoderState`已经实现；真实LDF训练改为冻结EMA encoder在线产生deterministic `mu`，其context sampler、latent statistics和hybrid batch已经落地。Web通过`InferenceSession`完成LDF commit、causal decode、四帧chunk与session锁接线；真实LDF训练仍等待正式H/G/F/C、noise/beta、condition与v-predict loss，Web模型加载则等待由该训练冻结的checkpoint合同。

### 删除状态

上述旧模块删除门槛已经由新版forward、CFG、Hybrid stream与全仓残留搜索满足。真实LDF teacher-training已接入冻结EMA VAE、离线UMT5 embedding lookup、逐token prompt、flow-v loss、optimizer和EMA；Web runtime结构已经完成，但模型加载在正式LDF checkpoint冻结前以`BLOCKED_ON_LDF_CHECKPOINT` fail-fast。HumanML263只作为离线物理表示来源，不作为新版VAE或LDF的运行时接口。

## 在线因果性不等于 LDF causal attention

V1 明确区分两种概念。

### 系统级在线因果性

系统只能使用：

```text
已提交的 clean motion history
当前 persistent noisy generation window
当前时刻已知的 text/typed observations
允许显式提供的 future goals
```

系统不能读取未来 GT motion。左端 token 只有达到 commit 条件后才输出，已提交 token 不再被未来模型结果重写。

### LDF window attention

Root/Body Transformer 可以在本次调用所提供的有限窗口内做双向 self-attention，包括 clean history、不同 beta 的 active tokens，以及显式 future-goal tokens。这不破坏系统级在线性，因为这些位置都是当前可用的生成状态或条件，不是未来 GT motion。

因此：

- streaming 能力来自 triangular noise schedule、visible-region contract、persistent noisy state 和 commit policy；
- 不来自 causal self-attention；
- LDF 每个 step 对当前窗口重新完整 forward；
- V1 不设计、不保存也不失效重放 Root/Body Transformer KV cache。

这与参考实现一致：FloodDiffusion/FloodNet LDF 默认 `causal=False`；ARDY 的 two-stage denoiser也是完整 window Transformer，而 ARDY tokenizer 才使用 causal attention。

## 顶层模型候选

### 1. Body tokenizer/VAE

```text
body265 ----------------------> BodyEncoder ------> latent_motion -------+
                                                                         |
root_motion -------> LocalRootMotionCodec --> local_root_motion ---------+
                                                                         v
                                                                   BodyDecoder
                                                                         |
                                                                         v
                                                               reconstructed body265
```

顶层职责：

- encoder 压缩 body，而不是把 authoritative explicit root 混入不可解释 latent；
- decoder 使用由 explicit root 派生的 local/kinematic root condition，使身体、落脚和位移动态一致；
- explicit root 旁路 body reconstruction，最终再与 decoded body 组装为评测/播放表示；
- VAE/AE/FSQ、latent width 和具体 decoder protocol 留到第二遍冻结。

### 2. Interleaved two-stage LDF

```text
inputs at one flow/denoising step
  noisy root_t + noisy latent_t
  clean hybrid history
  text
  typed root/body observations + masks
  time/noise level + region masks
                |
                v
        Root Transformer
                |
        root prediction (v or x0)
                |
        recover/fuse clean root_x0
                |
        derive local root condition
                v
        Body Transformer
                |
        body prediction (v or x0)
                |
                v
  structured LDFPrediction
                |
                v
  shared FloodDiffusion scheduler updates persistent hybrid_t
```

责任边界：

- Root Transformer 负责全局移动、heading、速度变化和 root observations；
- Body Transformer 负责在 clean root plan 下生成 body dynamics，而不是再独立决定另一条隐式轨迹；
- scheduler 是 noisy state 更新的唯一所有者，Transformer 不直接修改 persistent buffer；
- runtime 只负责 world/session/epoch 变换、condition compilation、commit 和事务，不做第二个生成器；
- body tokenizer只负责 body representation，不拥有 root/world timeline。

## LDF 详细输入输出候选 v0

本节把顶层 two-stage 结构展开为可以继续审查的模型协议。本轮已经冻结：结构化 root/body state、共享三角 `beta`、每步 root-first/body-second、non-causal/no-KV、stage 2 只使用 clean-root-derived condition、root 主监督不经过 VAE、scheduler 独占 noisy-state 更新，以及 V1 stage-boundary detach。未明确标记的内部维度和 observation/CFG 细节仍为 `PROVISIONAL`。

### 1. 顶层调用

```text
LDF.forward(
    inputs: LDFInput,
) -> LDFPrediction
```

LDF 不接收 world-space route，不拥有 runtime timeline，也不在 `forward()` 内修改 persistent buffer。传入模型的 root/observations 已经由上游转换到同一个模型坐标并分别归一化。

### 2. `LDFInput` 与 `HybridMotion`

```text
HybridMotion
  root_motion:        float [B,T,4,5]
  latent_motion:      float [B,T,Z_body]

LDFInput
  noisy_motion:       HybridMotion
  beta:               float [B,T]
  history_mask:       bool  [B,T]
  generation_mask:    bool  [B,T]
  timeline_position_ids: int64 [B,T]
  rope_position_ids:     int64 [B,T]
  previous_root_frame: float [B,5] | None
  previous_root_valid_mask: bool [B] | None
  condition:           LDFCondition
```

语义：

- `noisy_motion.root_motion` 是四帧一 token 的 normalized explicit root diffusion state；语义保持 `[4,5]`，只在 input projection 内 flatten 为 20D；
- `noisy_motion.latent_motion` 是同一 token 的 normalized body-latent diffusion state；
- `beta` 使用 FloodDiffusion 的方向：`0=clean, 1=pure noise`；
- history 区域满足 `beta=0`，其中 root/body 是 clean committed state；
- generation 区域只包含当前denoiser forward已经可见或本步即将更新的active band；它可以包含刚进入调度、当前`beta=1`但`next_beta<1`的边界token；
- pure-noise frontier继续存在于固定shape的`noisy_motion`中，但满足`~history_mask & ~generation_mask & beta==1`，不进入本步attention，也不增加独立结构类型；
- history/generation 在有效区域内互斥，future constraints 不属于 `HybridMotion`；
- `timeline_position_ids`是runtime absolute timeline坐标，服务于三角调度、commit、rolling和condition slicing；
- `rope_position_ids`只服务于Transformer，并以当前first-generation token为0；history为负、generation从0开始、future为正；
- 两者必须按样本相差同一个`rope_origin`，禁止再次用attention可见长度或future horizon隐式推导motion位置。

`LOCKED`：Root 与 body 使用相同的 per-token `beta` 和 active region，使两阶段在同一个三角时间面上更新；root/body 可以使用不同 normalization 和 loss scale，但 V1 不引入两套独立 scheduler。

### 3. `LDFCondition`

```text
LDFCondition
  text_context / text_null_context
  root_condition_value / root_condition_mask
  body_condition_value / body_condition_mask
  future_root_condition_value / future_root_condition_mask
  future_timeline_position_ids / future_valid_mask
```

约束：

- text context 支持 token 时间轴上的 prompt segment/change；Root/Body Transformer共享同一语义文本条件和 text encoder，不各自维护文本历史；
- root/body observation 使用 typed value + mask；进入 LDF 前已完成坐标变换、normalization 和 active-window 裁剪；
- observation projector 在模型内部产生 in-window observation features 与 constraint-only future-goal tokens；
- future timeline IDs使用absolute坐标并严格位于当前motion window之后；Root Stage根据`rope_origin`派生future RoPE IDs，future token数量不得改变motion、beta或Body Stage长度；
- `previous_root_frame/previous_root_valid_mask`由`LDFInput`成对携带，只供backward local-root codec使用，不伪装成noisy generation token或CFG条件；全batch cold start时两者均为`None`，随机crop混合batch则使用physical `[B,5]`与逐样本bool `[B]`；
- condition dropout 与 CFG 分支由 `utils/conditions/ldf.py` 的 `create_window_condition/create_ldf_condition/create_cfg_condition` 纯函数创建，不增加公共 wrapper dataclass；
- 不存在 `controlnet_condition`、`root_refiner_plan` 或 post-decode feedback 输入。

### 4. Observation compilation

对当前 state 范围内的 root observation，形成：

```text
root_condition_value: float [B,T,4,5]
root_condition_mask:  bool  [B,T,4,5]
```

Root stage 使用一个只读 input view：

```text
root_visible = where(
    root_condition_mask,
    root_condition_value,
    noisy_motion.root_motion,
)
```

同时把 value 与 mask 投影给网络。这个 overwrite 只改变本次 Root Transformer 的可见输入，不原地修改 `LDFInput.noisy_motion.root_motion`。

稀疏 body observations 通常是 frame-level body265/joint/keyframe constraints，并不天然等于 causal body latent。因此 V1 只把它们通过 observation projector 作为条件，不对 latent state 或最终输出做 hard overwrite。

超出 current generation state 的 observations 被编译为 future constraint tokens。它们参加 non-causal window attention，但没有 `root_t/latent_t/beta`，也不会被 scheduler 更新或 commit。

### 5. Root stage

Root stage 的信息源：

```text
history token:
  clean root0 + clean body0 + region/position

generation token:
  root_visible_t + noisy latent_t
  + root/body observation features and masks
  + beta/region/position

future-goal token:
  future root condition values and masks
  + future relative position

shared conditions:
  text
```

采用针对语义区域的 input projections，再送入一个独立的 non-causal Root Transformer。Root stage 必须看到 `latent_t`，原因是 clean body history 和当前 noisy body state包含动作阶段、步态相位、接触与姿态信息；root 不应退化成与动作脱离的路径拟合器。

`LOCKED`：V1 保留 FloodDiffusion velocity convention：

```text
root_t = (1-beta) * root_x0 + beta * root_eps
root_v = root_x0 - root_eps
root_x0_model = root_t + beta * root_v_model
```

Root Transformer 的主要网络输出是：

```text
root_v_model: float [B,T,4,5]
```

模型 forward 内立刻恢复 `root_x0_model`，因为 body stage、root loss 和 local-root codec 都需要 clean explicit root。heading unit-circle projection只产生供 codec/body 使用的合法 clean-root view；它不使用 condition value替换模型输出，也不改写 scheduler 消费的网络 velocity。

### 6. Root-to-body stage boundary

```text
local_root_motion = LocalRootMotionCodec(
    root_x0_physical,
    previous_root_frame,
    boundary_validity,
)
```

`LocalRootMotionCodec` 输出 frame-level `local_root_motion`，再按四帧 patch 投影。body stage 接收的是由 clean `root_motion` 派生的 backward/current-heading-local condition，不是 noisy `root_t`，也不是另一条外部 trajectory prediction。

`LOCKED`：第一版训练采用：

```text
local_root_motion_for_body = stop_gradient(local_root_motion)
```

这与 ARDY 的稳定训练边界一致，并防止 body loss 在第一版中反向劫持 root planner。是否允许 body-to-root gradient 作为后续消融，不改变 forward 语义。推理本身在 no-grad 下不受 detach 影响。

### 7. Body stage

Body stage 的信息源：

```text
history token:
  clean body0 + derived clean root condition

generation token:
  noisy latent_t + derived clean local_root_motion
  + body/root observation features and masks
  + beta/region/position

future-goal token:
  future root condition values and masks

shared conditions:
  text
```

`LOCKED`：Body stage 不再接收 noisy `root_t` 作为第二个 root source；root constraint只影响 Root Transformer 的条件分支，Root Transformer预测的 clean root dynamics通过 `LocalRootMotionCodec` 唯一进入 body generation。

Body Transformer 与 Root Transformer参数独立，采用 non-causal window attention，并输出：

```text
latent_v_model: float [B,T,Z_body]
latent_x0_model = latent_t + beta * latent_v_model
```

body constraints只参与 condition projection 和相应 decoded consistency loss；V1 不对 latent state或最终 body output做 hard overwrite。

### 8. `LDFPrediction`

```text
LDFPrediction
  velocity:                  HybridMotion
    root_motion:             float [B,T,4,5]
    latent_motion:           float [B,T,Z_body]
  clean_root_motion:         float [B,T,4,5]
  local_root_motion:         LocalRootMotionBatch
```

用途：

- scheduler 只消费 `velocity.root_motion/velocity.latent_motion` 和 region masks；
- root/body diffusion losses监督 `*_v_model`；
- direct root/path losses监督 `root_x0_model`，不经过 VAE decoder；
- body stage和body decoder共享同一个 `local_root_motion` codec结果，避免训练/推理派生协议分叉；
- constraint-following diagnostics 可以作为训练日志，但不扩张最小公共 dataclass。

### 9. 一个 denoising step 的唯一状态更新路径

```text
immutable LDFStreamState
        |
        v
LDF.forward
        |
        v
LDFPrediction
        |
        v
HybridScheduler.step(prediction, state)
        |
        v
new LDFStreamState
```

`LOCKED`：Root/Body Transformer、observation projector和VAE都不得原地修改 persistent state。这样 self-forcing、正式 streaming、snapshot/rollback 和离线 generation 可以复用同一个 scheduler 状态转移。

### 10. 为什么称为 interleaved

一次 flow step 内先 root、后 body；scheduler更新两者后，下一个 flow step 的 Root Transformer 会看到上一步已经更新的 `noisy_motion.latent_motion`。因此 root影响当前 body，body又通过下一步的 noisy hybrid state影响后续 root。它不是两个完全独立的串联系统，也不是先生成整段 root 再单独生成整段 body。

## `root_motion [B,F,5]` 的物理定义 v2

`LOCKED`：参考 ARDY 的变量职责，但把公共 hybrid 字段压缩为 `root_motion/latent_motion`；坐标 frame 由配套元数据表达，不编码进字段名：

```text
ARDY global_root_motion -> Floodcontrol root_motion
ARDY local_root_motion  -> Floodcontrol local_root_motion
ARDY latent_body_motion -> Floodcontrol latent_motion
```

`root_motion` 表示逐帧显式 root 序列。当前协议使用一个在 persistent state 生命周期内固定、只含 XZ 平移的坐标 frame，不是随 active window 旋转的 SE(2) anchor；静态原点元数据命名为 `anchor_origin_xz`。因此 anchor 是坐标元数据，不是 `HybridMotion` 字段名的一部分。

### 1. 权威物理量

对物理帧 `f`：

```text
root_motion[f]: RootMotion
    = [x_anchor, root_y, z_anchor, cos(yaw), sin(yaw)]
```

五个量依次为：

| 通道 | 名称 | 单位/范围 | 精确语义 |
|---:|---|---|---|
| 0 | `x_anchor` | meter | stable anchor 坐标轴中的 pelvis/root X；只减 anchor 的 XZ 平移原点 |
| 1 | `root_y` | meter | 相对数据地面的 pelvis/root 高度；不随 XZ rebase 改变 |
| 2 | `z_anchor` | meter | stable anchor 坐标轴中的 pelvis/root Z；只减 anchor 的 XZ 平移原点 |
| 3 | `heading_cos` | `[-1,1]` | `cos(yaw)` |
| 4 | `heading_sin` | `[-1,1]` | `sin(yaw)` |

坐标约定与现有 geometry contract 对齐：Y-up，yaw 绕 `+Y`，`yaw=0` 面向 `+Z`，forward XZ direction 为 `[sin(yaw), cos(yaw)]`。

这里的 `root_y` 不能删除。body265不包含权威global root；如果explicit root只保留XZ与heading，最终没有权威变量能够恢复pelvis height。

### 2. Anchor 只消除平移，不消除首帧朝向

设 anchor 的 world/session 平移原点为：

```text
anchor_origin_xz = [origin_x, origin_z]
```

则：

```text
x_anchor = x_session - origin_x
z_anchor = z_session - origin_z
yaw_anchor_axes = yaw_session
root_y = root_y_session_relative_to_ground
```

即模型坐标轴始终和该 session/sample 的稳定轴对齐。anchor 不保存 `origin_yaw`，也不执行：

```text
yaw_relative = yaw - yaw_at_epoch
xz_relative = R(-yaw_at_epoch) * (xz - origin_xz)
```

因此 anchor 起点通常满足 `xz=0`，但 heading 可以是任意角度。active window 的第一帧既不要求 `xz=0`，也不要求 `yaw=0`。这直接避免“每次 crop 都把首帧重新恢复成零点零朝向”的旧数据错误。

一个 persistent `LDFStreamState` 在其生命周期内只引用一个明确的 translation-anchor epoch。何时开启新 anchor epoch、如何 beta-aware 地变换 partially noisy state，属于数据/runtime 文档；LDF `forward()` 不隐式 rebase。

### 3. 首帧 yaw augmentation

HumanML legacy recovery 会天然给完整 clip 的第一帧 `xz=0, yaw=0`。如果直接训练 `RootMotion`，模型会把零朝向误学成 cold-start 常量。训练数据应在 full-root recovery 之后、window crop 之前采样一个全局 yaw offset `phi`：

```text
xz_aug[f]  = R(phi) * xz[f]
yaw_aug[f] = wrap(yaw[f] + phi)
```

对应的root/path/body observations必须同步旋转。body265包含global rotations与global velocities，因此它不会在全局yaw变换下保持数值不变；必须验证的是local-root velocity不变量以及root/body同步旋转后的几何等价性。

这借鉴 ARDY 的 `randomize_first_heading()`，但不照搬其所有数据表示。ARDY 同样使用 `[x,y,z,cos,sin]` 作为生成 root，并在 runtime 只做 XZ translation recenter；它的 derived local-root XZ velocity仍在稳定的世界轴上。

### 4. 从物理 root 到 diffusion `root_t`

必须区分物理值、normalized clean target 与 noisy state：

```text
root_phys:  RootMotion [B,F,5]
root_x0:    RootMotionStats.normalize(root_phys) [B,F,5]
root_token: reshape(root_x0, [B,T,4,5])
root_t:     (1-beta) * root_token + beta * root_eps
```

同一个 token 的标量 `beta[B,T]` broadcast 到其四帧、五个通道；四帧都是真实连续帧，不设置“首 token 只有一帧”的例外。尾部 padding 依靠 frame/token validity mask 排除，不用伪 root 值参与 loss 或 codec。

`RootMotionStats` 建议使用5个 feature-wise statistics `[5]`，四个 token phase 共享，而不是 `[4,5]` 的 phase-specific statistics。这样 frame shift 不会改变同一种物理量的 normalization；phase-specific stats 只有在数据证明确有收益时才作为消融。

heading unit-circle约束定义在物理空间，而不是 normalized 空间：

```text
root_x0_model_normalized
  -> RootMotionStats.unnormalize
  -> normalize([heading_cos, heading_sin]) onto unit circle
  -> RootMotionStats.normalize
  -> root_x0_manifold_safe
```

精确 heading observation 的 mask 必须成对出现：`mask_cos == mask_sin`。单独约束其中一个通道不代表一个定义完整的物理 heading。

### 5. 为什么五维中不直接放速度

- XZ pose 可以直接接受 waypoint/trajectory overwrite 和计算误差，不需要先积分模型速度；
- cos/sin 避免标量 yaw 在 `-pi/pi` 处的不连续；
- root velocity、yaw velocity和local root motion都能从相邻 `RootMotion` 确定性派生；
- 如果同时生成 absolute pose 与 velocity，会引入两套可能不一致的 root source，需要额外一致性约束和状态更新规则。

`LOCKED`：因此 `root_motion: RootMotion` 是唯一权威 root state，速度不作为额外生成通道。`local_root_motion` 和 `HumanMLLegacyRootMotion` 都是有明确边界、差分方向和 validity mask 的派生视图，不能写回或替代 `root_t`。

## `local_root_motion [B,F,4]` 的派生定义 v1

`LOCKED`：沿用 ARDY 的 `local_root_motion` 职责名称，但使用适配 cached decoder 的 Floodcontrol语义：backward difference、current-heading-local planar velocity、feature-wise validity。

### 1. 数值定义

对当前帧 `f` 与前一帧 `f-1`：

```text
p[f]     = [x_anchor[f], z_anchor[f]]
yaw[f]   = atan2(heading_sin[f], heading_cos[f])
delta_p  = p[f] - p[f-1]
delta_yaw = wrap(yaw[f] - yaw[f-1])

velocity_heading[f] = R(-yaw[f]) * delta_p * FPS
yaw_rate[f]         = delta_yaw * FPS

local_root_motion[f] = [
    yaw_rate[f],
    heading_velocity_x[f],
    heading_velocity_z[f],
    root_y[f],
]
```

单位依次为 `[rad/s, m/s, m/s, m]`。旋转必须复用正式XZ geometry utility并通过：`yaw=0`沿anchor `+Z`移动得到positive local `vz`；`yaw=90°`沿anchor `+X`移动也得到positive local `vz`。

### 2. 为什么不是ARDY原公式

ARDY同样从显式5D root派生4D local root，并同时供给body stage与tokenizer decoder；但其公开实现使用forward difference和world-axis XZ velocity。Floodcontrol保留跨commit的causal decoder state，因此改为backward/current-heading-local定义：

```text
ARDY:        forward + stable-axis velocity + full/window re-decode
Floodcontrol: backward + current-heading-local velocity + cached step decode
```

后向差分使token `k` 的条件只依赖：

```text
previous committed root frame
+ current token's four root frames
```

token commit后这些输入全部冻结，后续active root更新不会使已保存的decoder state失效。frontier仍服务三角调度，但不服务decoder root condition。

### 3. Boundary与validity

```text
LocalRootMotionBatch
  values:        float [B,F,4]
  feature_valid: bool  [B,F,4]
```

正常连续生成或随机crop `s>0` 使用真实 `root_motion[s-1]` 作为 `previous_root_frame`，四个feature均有效。混合batch通过`previous_root_valid_mask [B]`逐样本表达该边界；完整clip/cold start没有真实previous pose时，在clean root预测后复制frame 0的XZ/heading，使第一条速度数值为零，但标记：

```text
feature_valid[:,0] = [False, False, False, True]
```

`root_y[f]`来自当前帧，不依赖boundary。所有invalid feature必须在各自normalization之后置零，再与独立validity embedding共同进入消费者；不能让伪数值穿过value projection。

### 4. 两个消费者

```text
clean predicted root_motion
        |
        v
LocalRootMotionCodec
        |
        v
local_root_motion
        |
        +--> stop-gradient --> Body Transformer
        |
        +--> commit-time ----> cached VAE Decoder
```

Body Transformer在每个denoising step重算当前active span；VAE Decoder只在token最终commit时消费该token的4帧local root，并将候选decoder state与root/body/world timeline原子commit。

`HumanMLLegacyRootMotion` 仍保持dataset-specific forward-row/per-frame-displacement convention，只用于263D组装和兼容评测，不与 `local_root_motion` 共用codec。

## Cache/state 分类（顶层已决定）

后续文档不得再用一个不加限定的 `cache` 同时指代以下状态：

| 名称 | V1 是否存在 | 所有者 | 是否是 LDF attention KV |
|---|---:|---|---:|
| `LDFStreamState` | 是 | scheduler/runtime transaction | 否 |
| `VAEEncoderCausalState` | V1 默认否 | 不适用 | 否 |
| `VAEDecoderState` | 是 | body tokenizer decoder/runtime transaction | 否；可由 causal-conv features 或 Transformer KV 构成 |
| `LDFRootKVCache` | 否 | 不适用 | 是 |
| `LDFBodyKVCache` | 否 | 不适用 | 是 |
| `TextEmbeddingCache` | 可选 | text encoder/runtime | 否 |
| `CompiledObservationEmbeddingCache` | V1 默认否 | observation projection | 否 |

其中：

- `LDFStreamState` 保存 clean history、partially noisy active root/body、pure-noise frontier、每 token noise level 和窗口元数据；它是生成状态，不是神经网络 cache；
- VAE encoder/decoder仍使用 causal attention/causal temporal operators；V1 decoder逐token消费 committed `latent_motion + local_root_motion`，持久化显式 `VAEDecoderState`；
- 推荐 `decode_step(committed_state) -> (body_frames, candidate_state)` 的out-of-place事务接口：校验成功后才交换committed state，失败时丢弃candidate；命令式实现则必须提供等价snapshot/restore或checkpoint/replay；
- parity 测试要求相同body tokens、local root、position IDs和mask下，full causal decode与逐token cached decode在容差内一致；
- observation 在 world/session/epoch 或 active window 改变后由 typed values/masks 重新投影，V1 不为了省一次 projection 引入坐标相关 embedding cache；
- text embedding 可以按文本内容缓存，因为它不依赖 motion coordinate epoch。

## 三条主调用路径

三条路径应尽量共用同一个 denoiser 和同一种 hybrid state，只通过 observation value/mask 区分：

```text
text-only
  text + empty spatial observations

root/path controlled
  text + typed root observations/masks

sparse pose controlled
  text + typed root/body observations/masks
```

### CFG 所有权与分支协议

`LOCKED`：V1 保留独立的 text CFG 与 constraint CFG，但不再使用名称和实现都绑定旧 ControlNet 的 `cfg_scale_traj`。公共命名为：

```text
cfg_scale_text
cfg_scale_constraint
```

如果 V1 暂时只有 root 类空间约束，内部可以使用更窄的 `cfg_scale_root_constraint`，但公共协议不应暗示约束永远只有一条 XZ trajectory。

参考 ARDY/Kimodo，CFG 的“无条件”基线实际是 `history-only`，不是完全无运动条件。以下输入在所有分支中相同：

```text
LDFInput.noisy_motion
committed clean history
history/generation region masks
position/time/noise level
previous_root_frame
```

文本与空间约束参与 CFG。Separated 模式定义三条批处理分支：

```text
text-only:       real text, empty constraints
constraint-only: empty text, real constraints
history-only:    empty text, empty constraints
```

ARDY/Kimodo 对应组合为：

```text
pred = pred_history
     + cfg_scale_text       * (pred_text       - pred_history)
     + cfg_scale_constraint * (pred_constraint - pred_history)
```

该公式不计算 `text + constraint` 联合分支，因而在两个 scale 都为 1 时一般不严格等于联合条件预测。V1 实现必须同时保留 regular/joint 模式作为基线：

```text
pred = pred_history
     + cfg_scale_joint * (pred_text_constraint - pred_history)
```

是否采用 ARDY/Kimodo additive separated 作为默认，还是采用包含联合条件分支、在 scale=1 时严格恢复联合预测的 hierarchical composition，留给训练方法与消融决定；这不影响以下已经冻结的所有权：

- 模型内部生成并传给 Body Stage 的 clean root 是结构化生成变量，固定以 scale 1 使用，不是 constraint CFG 对象；
- generated clean `root_motion`、committed history、VAE decoder state/cache 和坐标参考不是 CFG 对象；
- 用户可编辑的 current/future root path、waypoint、heading 和 sparse pose 都属于 constraint condition，可以使用 constraint CFG；
- V1 不提供最终 root/body hard replacement；必须连续的过去由 committed history和`previous_root_frame`状态协议保证，而不是伪装成 constraint；
- text-only 时退化为普通两分支 text CFG，不构造无意义的 constraint 分支。

因此，不应为路径控制重新建立 ControlNet；也不应在删除 ControlNet 时把 constraint guidance 能力一并删除。

## 预期收益及其适用边界

### 显式 root 作为时序规划变量

`PROVISIONAL`：直接预测 root 给模型增加了明确的全局运动规划目标，并把该预测作为 body stage 的中间条件。这比让一个低维 full-motion latent 隐式同时承担路径、速度、转向和姿态更强地约束时序结构与动作变化关系。

这是一种结构性归纳偏置，不应表述成未经验证的“必然提高认知”。旧模型的 latent 也隐式包含 root 信息；新版的优势在于 root 变得可解释、可直接监督、可观察、可约束，并成为 root/body factorization 的显式瓶颈。必须通过 one-stage hybrid 与 two-stage hybrid 消融验证收益。

### 轨迹主监督不再穿过 VAE decoder

`LOCKED`：root/path 的主要训练损失直接作用于 explicit root prediction，不再执行：

```text
predicted latent
  -> frozen full-motion VAE decoder
  -> legacy root velocity
  -> trajectory integration
  -> XZ loss
  -> gradient through decoder back to LDF
```

这会移除 FloodNet trajectory loss 的高成本 decoder forward/backward 激活路径，并避免 root control gradient 受 full-motion VAE reconstruction error 污染。

边界：

- body tokenizer/VAE 自身训练仍需要 decoder；
- 可选 decoded-body、FK、contact、foot-skate 或 end-effector consistency loss 仍可能经过 decoder；
- 因而准确目标是“root/trajectory 主监督不依赖 VAE decoder”，而不是“整个 LDF 训练永远不调用 decoder”。

## V1 顶层非目标

- 不把 ControlNet 作为 trajectory control 的正式主路径。
- 不以 post-decode root projection 或完整 263D re-encode 作为状态闭环。
- 不要求兼容旧 full-motion VAE/LDF/ControlNet checkpoint 或旧 token cache。
- 不在模型结构文档中提前承诺 self-forcing schedule、具体 loss 权重、网络层数或训练资源配置。
- 不因为参考 ARDY 就自动替换 FloodDiffusion 的三角调度和 v-predict；调度是否保留由训练方法文档单独验证。

## 第二遍：自下而上的协议冻结顺序

1. 冻结命名和坐标类型：world/session root、`root_motion`、`local_root_motion`、`HumanMLLegacyRootMotion`。
2. 冻结body265 explicit representation，确认global-yaw同步变换的几何等价性与local-root不变量。
3. 冻结 tokenizer/VAE I/O：严格四帧 patch、causal decoder state、backward local-root boundary、padding和deterministic body code。
4. 冻结 hybrid clean/noisy state：root patch 与 body token 是 named fields，不能只依赖扁平拼接的隐式切片。
5. 冻结两阶段 LDF：root stage 的输入输出、clean-root 恢复、stage boundary detach、body stage condition。
6. 冻结 typed observations、history/generation/future masks 和 CFG 接口。
7. 冻结 VAE cached-step transaction、hybrid persistent-state contract 与最低 shape/round-trip/full-vs-cached parity 测试。

## 必须产出的接口草案

```text
BodyTokenizerInput
BodyTokenizerOutput
LocalRootMotionBatch
HybridMotion
LDFInput
LDFCondition
LDFPrediction
LDFStreamState
VAEDecoderState
```

每个接口至少记录：

```text
semantic type
shape
physical/model coordinate frame
normalization owner
validity mask
temporal ownership
cacheability
```

## 当前待讨论问题

- 顶层目标和模块图已有第一轮 `PROVISIONAL` 候选，仍需逐项反例审查后才能 `LOCKED`。
- CFG 的所有权与输入分支已经冻结；仍需通过训练方法文档和消融决定 separated additive、regular joint 或 hierarchical composition 的默认公式及 condition-dropout 概率。
- explicit root 已冻结为 translation-only stable anchor 下的 `root_motion=[x,y,z,cos(yaw),sin(yaw)]`；`local_root_motion` 使用backward/current-heading-local派生，速度不进入生成state，也不设置独立physical velocity loss。
- body encoder是否严格只看body265；decoder是否仅以派生local-root4为外部条件。
- VAE、连续 deterministic autoencoder 和 FSQ 的第一版对比边界。
- V1 候选已选择 root/body 都预测 FloodDiffusion convention 下的 `v`，并在 stage boundary 恢复 clean `root_x0`；仍需用数值测试确认与现有 scheduler 的符号和端点完全一致。
- cold start 的初始 root/heading 是 clean boundary、typed observation，还是单独 prefix token。
- VAE decoder最终采用causal-conv feature state还是Transformer KV；这不改变 `VAEDecoderState` 的事务接口。
- cached decoder state的checkpoint频率、内存布局和rollback replay策略。

## 冻结条件

- 首先有一张不依赖具体 hidden dim 的顶层模块图，以及训练/推理各一条完整 dataflow。
- 每个模块只有一个明确的状态所有者和职责；模型、scheduler、runtime 之间没有重复修改同一状态。
- 所有核心接口可以用 dataclass/TypedDict 表达而没有重名或隐式约定。
- root codec、token packing、padding、causal prefix 和 observation mask 有明确测试规格。
- LDF接口不存在跨commit KV-cache生命周期；VAE decoder有独立的causal state所有权、事务commit和full-vs-cached parity规格。
- 不引用任何尚未确定的训练 loss 权重或配置值。
