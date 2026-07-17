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
- 每个denoising step中，Root Stage同时读取完整noisy root与独立的root observation value/mask并预测velocity；随后由`x_beta+beta*v`恢复clean `anchor_root_x0_model`，投影heading，再通过`LocalRootMotionCodec`派生backward/current-heading-local `local_root_motion`。observation不覆盖`x_beta`或恢复后的clean root；第一版local-root在训练时detach后送入body stage。
- `anchor_root_motion` 不增加速度生成通道，也不设置独立physical velocity loss；派生速度只作为Body Transformer/VAE Decoder condition或不参与优化的诊断指标。
- diffusion velocity loss 明确命名为：

```text
L_anchor_root_flow_v
L_latent_body_flow_v
```

这里的 `flow_v=x0-epsilon` 是 normalized diffusion-space target，不是物理速度。root 的额外直接监督只作用于 clean position/height/heading 与声明过的 constraint terms；不默认加入相邻帧速度差分 loss。

## 已实现的scaled-ARDY窗口训练内核

- 每个sample保留自己的自然parent长度`N_i=min(sample_tokens,50)`；长动作随机裁一个50-token父窗口，短动作不人为缩短。batch只在右侧padding，因此真实长度由`span_token_count[B]`表达。
- active band固定为`C=chunk_size=5`。K=1时每个sample独立均匀采样`H_i∈[0,N_i-C]`，并令`F_i=N_i-H_i-C`，唯一预算为`H_i+C+F_i<=50`，不再分别截断history与future。K-step self-forcing时额外保留`R=K-1`，采样上界变为`N_i-C-R`。
- 第`i`步使用`history=[0,H+i)`、`active=[H+i,H+i+5)`、`frontier=[H+i+5,S)`。persistent noisy state仍保存完整S和固定frontier noise，但Transformer只接收`history+active`有效前缀；尚未开始更新的pure-noise frontier不进入self-attention。它与span外future-root condition token仍是两类对象。
- K=1 ideal baseline只采样一次per-sample continuous phase和absolute-token root/body Gaussian noise，并用`x_beta=(1-beta)x0+beta*epsilon`及`v*=x0-epsilon`训练当前active band；该路径保持原有输入、随机分布和MSE不变。
- K>1表示实际rollout并commit的token数，不表示denoise step数。persistent rollout从runtime commit boundary开始，只在起点调用一次`mix_fixed_noise()`，此后root与latent noisy state都通过共享`LDF.denoise_step()`的Euler更新持续演化，不再为下一commit重建理想`x_beta`。
- 每个commit只编译一次text/XZ condition；事务内全部denoise steps共享同一condition。前K-1个commit以及最终commit除最后一次denoise step外全部在`torch.no_grad()`中运行，只有最后一次denoise step保留梯度。
- 每次训练commit后把预测token detach为history，并以其最后一帧physical XZ执行与runtime相同的`(1-beta)`root state rebase；clean GT root、generated history、previous-root boundary和future XZ condition一起换到新model origin，latent与fixed Gaussian source保持不变。
- persistent rollout要求至少一个真实history token；true cold-start及其较长首次warm-up事务继续由K=1路径覆盖。启用self-forcing时训练Dataset会过滤到至少`C+K`个token，保证`H>=1`仍有足够frontier完成K次commit。
- 当前300k正式训练使用硬分段curriculum：`[0,100k)`为K=1 ideal bridge，`[100k,200k)`为K=2 persistent rollout，`[200k,300k)`为K=5 persistent rollout。`phase_start_step=100000`之前由硬gate保持K=1；之后以`phase_steps=200000`计算进度，并在0.5阈值切换到K=5。两个rollout阶段的`teacher_replay`均为0，因此不会随机退回K=1。

训练实现分为`flow.py`的flow/endpoint代数、`batch.py`的ideal与arbitrary-state输入合同、`self_forcing.py`的K curriculum/window plan、`rollout.py`的persistent denoise/commit/rebase循环，以及`losses.py`的ideal flow-v与off-path endpoint reduction。训练与runtime共同调用`LDF.denoise_step()`，因此root/body beta、`delta_beta`和Euler更新只有一个实现；buffer rolling仍只属于runtime。数据内容合同由CPU collator在构造时保证；GPU热路径只保留shape/dtype检查，完整plan/input内容校验仅按debug周期抽查，不能在每个denoise step重复同步。`LDFLightningModule`通过冻结EMA VAE的`tokenize_window()`在线得到deterministic normalized μ；冻结UMT5只在训练前运行一次并生成caption-to-embedding table，训练热路径按prompt字符串lookup token-aligned context，不在LDF GPU上加载11GB文本编码器。LDF checkpoint只保存LDF/EMA，但附带VAE/statistics/text路径与VAE统计用于resume前校验。

K>1最终denoise step不继续使用只对ideal bridge成立的`x0-epsilon`target，而是在实际solver state上恢复：

```text
x0_hat = x_current + beta * v_pred
L_offpath = SmoothL1(x0_hat, x0) / max(beta, 0.1)^2
```

root与latent分别按现有权重归约，`rollout_weight`控制整条off-path监督。该目标在ideal bridge上退化到原v-predict误差，在off-path区域是显式的endpoint-stabilizing extension；`offpath_beta_min`避免小beta放大异常target。physical XZ boundary displacement辅助loss覆盖history→active、active内部及Patch4边界，但`root_boundary_weight`默认0，不改变正式主loss。

Validation整轮只在开始时把EMA shadow换入模型一次，所有loss probe和inline generation复用这组参数，并在epoch-end恢复训练参数。临时`collected_params`恢复后立即释放且不进入checkpoint；异常由统一callback回滚，checkpoint若在EMA参数仍激活时触发会fail-fast，避免把EMA权重误写成主训练state。

计算层保持相同可见性语义但压缩无效工作：Root在有future约束时向量化打包`visible motion | future root`，无future时只运行batch内最大visible prefix；Body同样只运行最大visible prefix并把尾部补零。FlashAttention varlen打包与恢复使用布尔prefix mask，不再逐row执行Python scalar读取。pure-noise frontier仍保留在persistent state且不进入本步Transformer，compact只删除已被mask排除的pointwise/FFN计算，不改变non-causal visible attention。

## 已实现的XZ轨迹条件训练

每个rollout先从translation-anchored、random-yaw增强后的clean normalized root采样一次absolute XZ约束计划；teacher/self-forcing所有step复用这份计划，再按当前active窗口将其编译为两类只读条件。约束只包含XZ，不暴露root y或heading。

```text
dense trajectory（50%）:
    从首个active token开始标记动态future范围内的所有真实帧XZ

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
```

三种采样概率之和必须为1。constraint dropout在采样计划之后按sample清空整份约束，因此它是唯一产生无轨迹条件样本的机制；mode sampler不会用短span意外制造无约束样本。训练入口要求`max_horizon_token`为正、两种dropout位于`[0,1]`且`max_waypoint_count>0`，并直接检查所需statistics/checkpoint/data，不再依赖人工状态字符串。

文本条件遵循FloodDiffusion的局部性：HumanML3D的一条动作caption在span内重复；BABEL的区间caption编译到对应token。Root/Body Stage的每个motion query直接cross-attend自身prompt；后续Transformer层仍可通过可见motion token之间的non-causal self-attention传播已经注入的文本信息，因此该协议是`direct token-aligned cross-attention`，不是严格文本隔离。pure-noise frontier不进入当前attention，未到达active band的未来prompt不能提前传播。训练以sample级text dropout构造空文本分布，推理CFG复用同一空文本语义。

Validation使用固定seed的独立probe：teacher cold强制parent从真实序列第0 token开始，并固定`source_start=0, context=0, previous-root invalid, H=0, K=1`；teacher continuation在确定性的中间parent窗口上按early/middle/late标签选择合法`H>=1`，默认中点probe等价于固定`H=5,K=1`配置；启用self-forcing后再增加固定`H=1,K=5` probe。source parent、phase、root/body noise和文本选择对同一batch保持确定，使checkpoint loss可横向比较。

## 当前待讨论问题

- root/body flow-v的最终相对权重与是否加入decoded auxiliary loss。
- persistent rollout启用后的`rollout_weight`、ideal replay比例和最大K需要通过现有checkpoint微调消融确定。
- constraint overwrite/clamp 在不同 beta 下使用 clean value、matched-noise value还是独立 observation channel。
- decoded/FK/contact/skate loss 的梯度路径和启用阶段。

## 必须先通过的代数测试

- fixed-noise `add_noise`、v target、self-forcing clean recovery和一步update对所有beta一致。
- translation-only root rebase 与当前 v convention 的符号一致。
- root/body 两阶段返回的 structured prediction 与扁平 update 完全等价。
- active band沿一次采样的fixed source通过真实Euler state推进，frontier保持pure noise，root/latent均不在commit间回到ideal bridge。
- 每次commit后的model-origin变换满足beta=0完整平移、beta=1 source不变，且不与buffer rolling重复执行。

## 冻结条件

- 每个训练阶段有明确输入 checkpoint、冻结参数、数据分布、输出 artifact 和退出指标。
- “self forcing”“scheduled training”“history corruption”不再是模糊同义词。
- 所有目标方程先通过小型合成测试，再进入超参数文档。
