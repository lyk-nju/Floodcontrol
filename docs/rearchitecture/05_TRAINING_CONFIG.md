# 05 训练配置与实验矩阵

状态：`BLOCKED_BY_PROTOCOL`

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

