# 04 训练方法

状态：`OPEN`

## 本文只回答什么

- tokenizer/VAE 和 two-stage LDF 的训练阶段与冻结关系。
- 三角 token noise schedule、prediction target 和采样更新方程。
- root-first/body-second 在每个 denoising step 内如何训练。
- teacher forcing、self forcing、scheduled rollout/curriculum 的定义和边界。
- observation dropout/CFG、history corruption 和 rollout distribution matching。
- loss 的语义组成，但不在本文写死最终数值权重。

## 本文不回答什么

- 数据文件组织和 cache 路径。
- runtime rebase 事务实现。
- batch size、learning rate、层数、训练卡数等配置值。

## 需要分开的训练阶段

```text
Stage A: body tokenizer/VAE
Stage B: deterministic full-clip body-code export
Stage C: one-stage hybrid sanity baseline（是否需要，待定）
Stage D: root-first/body-second LDF teacher-forced training
Stage E: self-forcing / scheduled rollout adaptation
Stage F: constraint/long-horizon curriculum and final finetuning
```

## 已冻结的 LDF target

- root/body 共用 FloodDiffusion 的三角 per-token `beta` 和线性 flow convention；网络分别预测 normalized diffusion velocity `x0-epsilon`。
- 每个 denoising step 内先恢复clean `anchor_root_x0_model`，融合root observations并投影heading后，通过 `LocalRootMotionCodec` 派生backward/current-heading-local `local_root_motion`；第一版训练detach后送入body stage。
- `anchor_root_motion` 不增加速度生成通道，也不设置独立physical velocity loss；派生速度只作为Body Transformer/VAE Decoder condition或不参与优化的诊断指标。
- diffusion velocity loss 明确命名为：

```text
L_anchor_root_flow_v
L_latent_body_flow_v
```

这里的 `flow_v=x0-epsilon` 是 normalized diffusion-space target，不是物理速度。root 的额外直接监督只作用于 clean position/height/heading 与声明过的 constraint terms；不默认加入相邻帧速度差分 loss。

## 当前待讨论问题

- 三角 schedule 中 history、active band、frontier 的 loss region 与 weighting。
- ARDY 的训练策略中哪些是明确证据：variable history、random generation window、future constraints、random Y rotation、clean-history autoregression；哪些不能误称为 scheduled sampling。
- self forcing 的 rollout 单位、起始概率、增长 schedule、是否反传穿过历史，以及 train/inference state parity。
- GT history、model-generated clean history、partially denoised history corruption 是否需要分阶段引入。
- constraint overwrite/clamp 在不同 beta 下使用 clean value、matched-noise value还是独立 observation channel。
- decoded/FK/contact/skate loss 的梯度路径和启用阶段。

## 必须先通过的代数测试

- `add_noise`、v target、`predict_x0` 和一步 update 对所有 beta 一致。
- translation-only root rebase 与当前 v convention 的符号一致。
- root/body 两阶段返回的 structured prediction 与扁平 update 完全等价。
- self-forcing rollout 使用的 scheduler 与普通训练/正式推理是同一实现。

## 冻结条件

- 每个训练阶段有明确输入 checkpoint、冻结参数、数据分布、输出 artifact 和退出指标。
- “self forcing”“scheduled training”“history corruption”不再是模糊同义词。
- 所有目标方程先通过小型合成测试，再进入超参数文档。
