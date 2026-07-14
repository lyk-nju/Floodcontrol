# 03 在线推理与 Active Window

状态：`OPEN`

## 本文只回答什么

- runtime 中谁拥有 authoritative world state。
- SessionFrame、OriginEpoch 和模型坐标之间的变换及生命周期。
- persistent buffer 中 clean history、partially noisy active band 和 pure-noise frontier 的语义。
- commit、buffer roll、origin rebase、observation recompilation、状态失效和 rollback 的事务边界。
- 在线更新 trajectory/waypoint/text 后怎样重建当前条件。

## 本文不回答什么

- 网络层结构。
- 训练数据随机采样细节。
- loss 和超参数。

## 需要明确区分的状态

```text
authoritative world timeline
fixed SessionFrame transform
translation-only OriginEpoch offset
clean committed hybrid history
active noisy root/body states and beta per token
pure-noise frontier
compiled model-space observations
committed VAE decoder causal state
optional text/content cache
```

## 当前候选方向

- `PROVISIONAL`：SessionFrame 在一次 session 中固定方向基底。
- `PROVISIONAL`：OriginEpoch 只允许 X/Z translation，不做 rolling yaw re-anchor。
- `PROVISIONAL`：world route/observations 保持权威值，epoch 变化后重新编译到模型坐标。
- `PROVISIONAL`：线性路径下 noisy root translation rebase 使用 scheduler 的 clean coefficient；具体公式和 Flood v 约定在训练文档共同验证后才能锁定。
- `PROVISIONAL`：V1 不保存跨 commit 的 root/body Transformer KV；如将来加入，cache key 必须包含 origin epoch。
- `LOCKED`：V1保存显式VAE decoder causal state，每次只decode最终committed token；decoder candidate state必须与root/body/world timeline原子commit。encoder V1默认不持久化cache。

## 当前待讨论问题

- OriginEpoch 的触发器：固定 token 周期、LDF buffer roll、数值阈值或组合策略。
- translation delta 的符号、normalized-space 变换和四帧 root patch 广播。
- clean history 与 active state 在一次 atomic rebase 中的精确更新顺序。
- VAE decoder state采用causal-conv features还是Transformer KV，以及checkpoint频率、rollback replay和position IDs。
- buffer roll 与 origin rebase 是否允许在同一个 commit 发生。
- snapshot/rollback 必须覆盖哪些 RNG、persistent hybrid state、committed VAE decoder state、可选内容 cache 和 epoch metadata。
- runtime 输出怎样保证 rebase 前后 world root/joints 完全连续。

## 冻结条件

- 给定同一噪声和 condition，rebase 前后映射回 world 的下一步结果满足数值等价性目标。
- pure-noise frontier、partial-noise active root 和 clean root 的变换分别有单元测试。
- VAE decoder cache只依赖committed `latent_body_motion + backward heading-local local_root_motion`及token position；anchor XZ translation rebase不得改变它。其他任何可选cache也必须声明依赖。
- commit、roll、rebase、condition update 和异常 rollback 有明确状态机或事务顺序。
