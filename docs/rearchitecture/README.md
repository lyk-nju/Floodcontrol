# Floodcontrol 新版框架讨论索引

状态：`MODEL_CORE_IMPLEMENTED / STRICT4_VAE_AND_RUNTIME_OPEN`。Hybrid LDF、Root/Body两阶段主干、CFG和合成张量stream核心已经落地；strict-4 VAE、真实训练数据与Web runtime仍未接线。未标记为`LOCKED`的内容继续视为讨论项。

这个目录用于把新版 Floodcontrol 的设计讨论拆成可独立审阅的主题，避免把模型协议、数据语义、在线状态、训练目标和实验超参数混在同一份文档里。

## 文档边界

1. [`01_MODEL_ARCHITECTURE_AND_IO.md`](01_MODEL_ARCHITECTURE_AND_IO.md)
   只讨论模型结构和输入输出协议，包括 body tokenizer/VAE、hybrid token、root-first/body-second LDF，以及各类 mask 和 observation 如何进入网络。
2. [`02_DATA_PIPELINE.md`](02_DATA_PIPELINE.md)
   只讨论离线表示构建、完整 clip 恢复、数据增强、window/epoch 采样、token cache 和 Dataset/DataLoader。
3. [`03_STREAMING_ACTIVE_WINDOW.md`](03_STREAMING_ACTIVE_WINDOW.md)
   只讨论推理时 authoritative world state、SessionFrame、OriginEpoch、active noisy window、buffer roll、cache 生命周期和原子更新。
4. [`04_TRAINING_METHOD.md`](04_TRAINING_METHOD.md)
   只讨论优化目标和训练过程，包括三角调度、v-predict、teacher/self forcing、ARDY 可借鉴的训练策略，以及训练阶段顺序。
5. [`05_TRAINING_CONFIG.md`](05_TRAINING_CONFIG.md)
   只在前四份协议稳定后记录可执行配置、初始超参数、消融矩阵、资源预算和验收指标。
6. [`06_LDF_IMPLEMENTATION_DESIGN.md`](06_LDF_IMPLEMENTATION_DESIGN.md)
   记录 LDF 的最终代码层级、公共命名、`utils/conditions/ldf.py` 协议、Root/Body forward、CFG 接管、流式迁移 seam，以及 ControlNet/FlexTraj 的删除门槛。

用户提出的第 2 项被拆成离线数据和在线 active-window 两份文档，因为二者共享坐标协议，但状态所有权、失败模式和测试方法不同。

## 建议设计顺序

```text
上层任务能力与非目标
    -> 模型总体架构和模块职责
    -> 顶层训练/推理 dataflow
    -> 精确模型 I/O 和底层类型
    -> 数据能否无歧义构造这些 I/O
    -> runtime 能否在 active window 中保持同一语义
    -> 训练目标与 rollout 策略
    -> 具体超参数和实验配置
```

设计决策采用上述自上而下顺序；接口文档和实现验证则从基础类型、codec 和 mask 自下而上组合。每一阶段只冻结当前文档的决定。发现跨文档冲突时回到上游协议修改，不在下游用隐式 workaround 掩盖。

## 跨文档状态标签

- `LOCKED`：已经讨论、测试并同意，后续修改需要显式变更记录。
- `PROVISIONAL`：当前首选方案，但仍需代码或实验验证。
- `OPEN`：尚未决定，不应出现在生产接口中。
- `REJECTED`：讨论过但不采用，保留理由避免反复回到同一方案。

## 当前可作为讨论起点的事实

- `LOCKED`：统一采用严格的 `4 frames / token`，不保留首 token 单帧例外。
- `LOCKED`：生成状态采用 explicit root 与 latent body 的结构化组合，root 不再只是外部 ControlNet 条件。
- `LOCKED`：physical root、decoder root condition 和 HumanML legacy root 是不同类型，必须使用显式 codec。
- `LOCKED`：LDF 第一版保留 FloodDiffusion 的线性 flow/v-predict 与三角 token noise schedule，不因采用 ARDY 结构而替换调度器。
- `LOCKED`：LDF 公共 hybrid 字段为 `root_motion/latent_motion`；root 使用 translation-only stable anchor 坐标元数据，五维为 `[x,y,z,cos(yaw),sin(yaw)]`。
- `OPEN`：OriginEpoch 触发策略、VAE/FSQ 选择、latent width、self-forcing 方案和全部训练超参数。

历史实现与实验材料不在新版仓库重复保存；本文只保留对FloodDiffusion、FloodNet和ARDY设计来源的必要说明。
