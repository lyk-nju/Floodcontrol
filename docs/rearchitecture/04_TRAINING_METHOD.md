# 04 训练方法

状态：`TEACHER_TRAINER_IMPLEMENTED / SELF_FORCING_OPTIONAL`

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
Stage B: frozen EMA VAE online deterministic body encoding
Stage C: one-stage hybrid sanity baseline（是否需要，待定）
Stage D: root-first/body-second trajectory-conditioned teacher-forced training
Stage E: self-forcing / scheduled rollout adaptation
Stage F: long-horizon/replanning curriculum and final finetuning
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

## 已实现的固定span训练内核

- source span固定为batch共享的`S∈[10,50]` tokens，active band固定为`A=chunk_size=5`，初始划分满足`S=H+A+F_frontier`。
- 第`i`步使用`history=[0,H+i)`、`active=[H+i,H+i+5)`、`frontier=[H+i+5,S)`。persistent noisy state仍保存完整S和固定frontier noise，但Transformer只接收`history+active`有效前缀；尚未开始更新的pure-noise frontier不进入self-attention。它与span外future-root condition token仍是两类对象。
- 一个rollout只采样一次per-sample phase和absolute-token root/body Gaussian noise。active右移时只改变beta，同一token不会跳到另一条diffusion path。
- translation anchor由初始H确定并在整个K步保持不变；region boundary移动不触发OriginEpoch rebase。
- 前K-1步保持`model.train()`并在`torch.no_grad()`内运行，只用`x_beta+beta*v_pred`替换最左active token；最终一步保留梯度并只监督当前5-token active band。
- teacher baseline使用K=1；fine-tune默认K=2/3/5，并支持20%/10%/10%的K=1 replay。curriculum进度从显式`phase_start_step`起算，并在`phase_steps`内从0推进到1，不使用包含baseline的全局训练进度。replay概率是实验配置，不是模型协议。

训练实现分为`flow.py`的代数、`batch.py`的固定S输入合同、`losses.py`的root/body flow-v reduction和`self_forcing.py`的plan/state rollout。`LDFLightningModule`通过冻结EMA VAE的`tokenize_window()`在线得到deterministic normalized μ；冻结UMT5只在训练前运行一次并生成caption-to-embedding table，训练热路径按prompt字符串lookup token-aligned context，不在LDF GPU上加载11GB文本编码器。LDF checkpoint只保存LDF/EMA，但附带VAE/statistics/text路径与VAE统计用于resume前校验。它不调用也不复制正式runtime的commit、roll或rebase状态机。

## 已实现的XZ轨迹条件训练

每个rollout先从translation-anchored、random-yaw增强后的clean normalized root采样一次absolute XZ约束计划；teacher/self-forcing所有step复用这份计划，再按当前active窗口将其编译为两类只读条件。约束只包含XZ，不暴露root y或heading。

```text
dense trajectory（50%）:
    从首个active token开始标记所有真实帧XZ

sparse waypoints（25%）:
    在active + future lookahead内随机标记1–4个独立帧XZ

future goal（25%）:
    active约束为空，只标记一个严格位于首个active band之后的未来帧XZ
    样本没有真实future frame时退化为一个active waypoint
```

稀疏性在`[B,T,4,5]`的frame维采样，不会因选择一个token而自动暴露该token全部四帧。每个step只把落在当前active band的选中帧写入current mask；其后的选中帧按absolute timeline position压紧打包为future tokens。goal随self-forcing窗口前移会自然从future token变为active内约束，不会重新采样。lookahead只使用当前sample/span内真实有效token；短span末尾自然缩短，不用零值冒充未来轨迹。future value在进入projection前按feature mask清零，因此未观测的root y与heading不能泄露给Root Stage。Body Stage不读取raw active/future XZ，只读取Root Stage组合后的唯一clean root派生的local root和heading condition。

训练以sample为单位分别采样text keep/drop与constraint keep/drop，两个Bernoulli变量相互独立，并在同一个self-forcing rollout的所有step复用同一constraint决定。这使单分支训练自然覆盖history-only、text-only、constraint-only和joint四种分布；推理时`create_cfg_condition()`再显式构造对应分支并进行separated CFG。当前正式配置为：

```yaml
training:
  text_dropout_probability: 0.1
  constraint_dropout_probability: 0.1
  future_root_lookahead_tokens: 20
  constraint_sampling:
    dense_probability: 0.5
    waypoint_probability: 0.25
    goal_probability: 0.25
    max_waypoints: 4
```

三种采样概率之和必须为1。constraint dropout在采样计划之后按sample清空整份约束，因此它是唯一产生无轨迹条件样本的机制；mode sampler不会用短span意外制造无约束样本。训练入口要求lookahead为正、两种dropout位于`[0,1]`且`max_waypoints>0`；不允许把没有轨迹条件的text-only训练误标为`training_ready`。

文本条件遵循FloodDiffusion的局部性：HumanML3D的一条动作caption在span内重复；BABEL的区间caption编译到对应token。Root/Body Stage的每个motion query直接cross-attend自身prompt；后续Transformer层仍可通过可见motion token之间的non-causal self-attention传播已经注入的文本信息，因此该协议是`direct token-aligned cross-attention`，不是严格文本隔离。pure-noise frontier不进入当前attention，未到达active band的未来prompt不能提前传播。训练以sample级text dropout构造空文本分布，推理CFG复用同一空文本语义。

Validation使用固定seed的三个独立probe：teacher cold start固定`H=0,K=1`；teacher continuation使用显式`continuation_span_frames`，并按稳定sample index在同一loader中轮换early/middle/late source位置，固定`H=5,K=1`；启用self-forcing后再增加相同位置覆盖的固定`H=1,K=5` probe。continuation数据视图会提前排除无法在span前保留至少一个真实token的短clip，不能再把整段clip从token 0开始却标成continuation。phase、root/body noise和文本选择对同一batch保持确定，使checkpoint loss可横向比较。

## 当前待讨论问题

- root/body flow-v的最终相对权重与是否加入decoded auxiliary loss。
- GT history、model-generated clean history、partially denoised history corruption 是否需要分阶段引入。
- constraint overwrite/clamp 在不同 beta 下使用 clean value、matched-noise value还是独立 observation channel。
- decoded/FK/contact/skate loss 的梯度路径和启用阶段。

## 必须先通过的代数测试

- fixed-noise `add_noise`、v target、self-forcing clean recovery和一步update对所有beta一致。
- translation-only root rebase 与当前 v convention 的符号一致。
- root/body 两阶段返回的 structured prediction 与扁平 update 完全等价。
- active band沿固定absolute-token noise path推进，frontier保持pure noise，只有已提交history被detached prediction替换。

## 冻结条件

- 每个训练阶段有明确输入 checkpoint、冻结参数、数据分布、输出 artifact 和退出指标。
- “self forcing”“scheduled training”“history corruption”不再是模糊同义词。
- 所有目标方程先通过小型合成测试，再进入超参数文档。
