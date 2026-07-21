# 04 训练方法

状态：`TEACHER_TRAINER_IMPLEMENTED / SELF_FORCING_AND_PERSISTENT_COLD_IMPLEMENTED`

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

- 每个sample保留自己的自然parent长度`N_i=min(sample_tokens,50)`；长动作随机裁一个50-token/200-frame父窗口，短动作不人为缩短。batch只在右侧padding，因此真实长度由`span_token_count[B]`表达。
- active band固定为`C=chunk_size=5`。普通batch的每个sample独立均匀采样`H_i∈[1,N_i-C]`，并令`F_i=N_i-H_i-C`；只有显式cold-start ideal/persistent mixture使用`H_i=0`。唯一预算为`H_i+C+F_i<=50`，不再分别截断history与future。K-step self-forcing时额外保留`R=K-1`，采样上界变为`N_i-C-R`。
- 第`i`步使用`history=[0,H+i)`、`active=[H+i,H+i+5)`、`frontier=[H+i+5,S)`。persistent noisy state仍保存完整S和固定frontier noise，但Transformer只接收`history+active`有效前缀；尚未开始更新的pure-noise frontier不进入self-attention。它与span外future-root condition token仍是两类对象。
- K=1 ideal baseline只采样一次per-sample continuous phase和absolute-token root/body Gaussian noise，并用`x_beta=(1-beta)x0+beta*epsilon`及`v*=x0-epsilon`训练当前active band；该路径保持原有输入、随机分布和MSE不变。
- K>1表示实际rollout并commit的token数，不表示denoise step数。persistent rollout从runtime commit boundary开始，只在起点调用一次`mix_fixed_noise()`，此后root与latent noisy state都通过共享`LDF.denoise_step()`的Euler更新持续演化，不再为下一commit重建理想`x_beta`。
- 每个commit只编译一次text/XZ condition；事务内全部denoise steps共享同一condition。前K-1个commit以及最终commit除最后一次denoise step外全部在`torch.no_grad()`中运行，只有最后一次denoise step保留梯度。
- 每次训练commit后把预测token detach为history，并以其最后一帧physical XZ执行与runtime相同的`(1-beta)`root state rebase；clean GT root、generated history、previous-root boundary和future XZ condition一起换到新model origin，latent与fixed Gaussian source保持不变。
- steady-state persistent rollout要求至少一个真实history token；其起点是runtime commit boundary，每次commit只需`noise_steps/chunk_size`次更新。true-cold persistent rollout则从`H=0,current_step=0`开始，首次commit执行完整`noise_steps`次更新，后续commit回到`noise_steps/chunk_size`步事务。两种起点进入同一个solver循环并共享相同的input/update/commit/rebase原语。
- `self_forcing`是K=1 baseline与K>1 persistent rollout的统一课程入口，不存在独立enable或phase开关；`k_schedule`直接写成`[absolute_global_step,K]`并固定在0/100k/200k/300k进入K=1/2/3/5。`cold_start_replay=0.1`在选择当前K之前独立采样，从第0步到训练结束将10% global batch强制放到真实序列起点；随后使用rank-independent global-step RNG按1:1选择ideal或persistent cold。ideal分支继续均匀采样单个`denoise_step`，persistent分支在前两个commit的12-step生命周期中均匀采样一个可微microstep。
- K、teacher-replay与是否进入cold-start replay都是global-batch决策：K使用只依赖`seed/global_step`的CPU RNG；cold replay使用batch sampler提供的跨rank一致seed。具体cold denoise step与root/body noise、H、yaw和constraint一样使用rank-local RNG，因为它们不改变计算图或collective顺序。Ideal与rollout路径始终返回相同且顺序固定的loss metric键，未启用的指标显式记零，避免不同rank的标量日志collective与梯度bucket错位。

训练实现分为`flow.py`的flow/endpoint代数、`steps.py`的ideal/cold/arbitrary-state输入合同、`window.py`的K curriculum/window plan、`solver.py`的K=1/cold/persistent denoise-commit-rebase流程，以及`losses.py`的ideal flow-v与off-path endpoint reduction。训练与runtime共同调用`LDF.create_input()/denoise_step()/commit_step()`，因此visibility、root/body beta、Euler更新、heading投影和rebase只有一个实现；buffer rolling仍只属于runtime。数据内容合同由CPU collator在构造时保证；GPU热路径只保留shape/dtype检查，完整plan/input内容校验仅在debug与测试执行。`LDFLightningModule`通过冻结EMA VAE的`tokenize_window()`在线得到deterministic normalized μ；冻结UMT5只在训练前运行一次并生成caption-to-embedding table，训练热路径按prompt字符串lookup token-aligned context，不在LDF GPU上加载11GB文本编码器。LDF checkpoint只保存LDF/EMA，并比较VAE checkpoint内容、text table内容、root/local/latent statistics、模型结构和50-token训练合同，不比较绝对路径。

## 已冻结并实现的persistent cold训练合同

### 实验事实

`debug/body_prefix_oracle_experiment.py`使用`step_160000.ckpt`的EMA权重，在完整GT root5、joint CFG、dense XZ、固定初始Root/Body噪声和完整stream rollout下，对Body latent的cold前缀做了反事实干预。核心坏例是HumanML3D `000021 / seed 4322`：

| Body分支 | feet/root平均角度 | `>=135°`反向帧比例 | 首次反向token |
|---|---:|---:|---:|
| 全部模型预测 | `122.50°` | `0.517` | `0` |
| 只在第1个token commit后换GT | `87.18°` | `0.386` | `2` |
| 前2个committed tokens换GT | `7.69°` | `0` | 无 |
| 第1个token在active生命周期内钳制为GT | `8.76°` | `0` | 无 |
| 全部GT body latent | `8.31°` | `0` | 无 |

该结果说明错误在第一个active token的生命周期内已经产生，并会先污染同一active band，再通过commit写入causal history。只在第一个token提交之后修正已经偏晚；修正前两个committed tokens或在active阶段修正第一个token均能恢复整段动作。

正常对照不支持“GT前缀总会强制修好任何动作”的解释：`001168 / seed 4322`的普通rollout为`12.84°`且没有反向帧；`000021`的seed 4321/4323普通rollout分别为`6.88°/8.44°`，所有前缀干预后仍保持正常。异常是sample与source noise共同决定的离散下肢mode，不是确定的数据旋转错误。

平均latent误差也不能识别该mode：正常seed 4321的normalized latent RMSE为`1.361`，反而高于反脚seed 4322的`1.278`。因此继续单纯压低全维latent flow loss，不能保证消除局部错误mode。

`debug/direct_vs_fk_foot_audit.py`进一步比较body265 direct joint positions和child-global rotation FK。坏例中两者相对Root均反向，分别为`122.50°/124.67°`，二者相互角度仅`17.85°`；前2-token GT干预后两者同时恢复为`7.69°/12.21°`。这排除了“只有position head反向、rotation head正常”的简单VAE分支冲突，主要错误是生成latent选中了整体错误的lower-body mode。现有可选VAE FK loss使用parent-global rotation，在GT近似输出上产生约`0.275 m`全关节MPJPE，因此在修正HumanML global-rotation FK合同前不得直接启用该loss作为补丁。

代码覆盖审计与实验定位一致：当前cold replay只在任意beta直接构造理想`x_beta`，而K>1 persistent solver明确排除`H=0`。因此训练已经覆盖“cold时各个beta上的理想状态”和“有真实history时的persistent off-path状态”，唯独没有覆盖部署首次生成真正访问的“从H=0 source连续积分得到的persistent off-path状态”。

### 因果判断

上述证据不表示VAE脚部约束已经完美；它只确定当前最高优先级不是继续加强Root heading，也不是立即打开尚未校准的FK loss，而是补齐cold solver-state训练分布。目标是让Body Transformer学会在自己早期预测形成的状态上纠正错误mode，避免其在5-token active band内传播并成为历史。

### 冻结合同

- 所有global batch中仍有`10%`进入真实序列起点的cold-start路径，且各DDP rank必须选择同一计算分支。
- cold batch内部固定为`50% ideal bridge + 50% persistent cold rollout`。ideal分支继续均匀采样beta并使用原始`x0-epsilon` flow目标，防止vector field只向endpoint regressor退化；persistent分支负责部署状态分布匹配。
- persistent cold必须从`H=0,current_step=0`和同一份fixed Gaussian source开始。第一次commit执行完整`noise_steps=10`个microsteps；第二次commit执行`noise_steps/chunk_size=2`个microsteps，默认最多覆盖前两个commit的12-step生命周期。
- persistent过程中只在起点调用一次`mix_fixed_noise()`；之后每个microstep都把上一步`LDF.denoise_step()`输出作为下一步输入，禁止重新按clean motion和beta构造理想状态。
- 训练与runtime复用同一个`create_input()/denoise_step()/commit_step()`事务。Dynamic Future随每一步实际`history_mask | generation_mask`移动；第一次commit后执行相同的heading投影、token提取和`(1-beta)`translation rebase，并同步变换noisy state、clean GT、previous root与future XZ坐标。
- 为控制显存，可微监督点在这12-step cold生命周期内采样；其之前的microsteps仍由当前模型真实执行，但在`torch.no_grad()`下生成detached persistent state。只有选中的当前step保留计算图并计算off-path endpoint loss。这样覆盖早期、中期、commit前与第二commit状态，同时避免保存整条12-step反向图。
- persistent cold使用现有带`offpath_beta_min`保护的endpoint目标；ideal cold继续使用原flow-v目标。不得在每个microstep重新采样source，也不得通过decode→encode重建body latent。
- `cold_start_replay=0.1`、cold内部`ideal:persistent=1:1`和最多2 commits是冻结的首轮训练默认；它们属于训练采样合同，不改变50-token root statistics协议，也不要求重算`root_stats.npz`。

该合同目前是`LOCKED / IMPLEMENTED`：实现与合成parity测试已经落地，但尚未进行新的GPU微调，因此这里的“已实现”不等于已经证明训练收益。是否降低真实反脚率仍必须通过同一固定噪声实验矩阵验证。

K>1最终denoise step不继续使用只对ideal bridge成立的`x0-epsilon`target，而是在实际solver state上恢复：

```text
x0_hat = x_current + beta * v_pred
L_offpath = SmoothL1(x0_hat, x0) / max(beta, 0.1)^2
```

root与latent分别按现有权重归约，`rollout_weight`控制整条off-path监督。该目标在ideal bridge上退化到原v-predict误差，在off-path区域是显式的endpoint-stabilizing extension；`offpath_beta_min`避免小beta放大异常target。physical XZ boundary displacement辅助loss覆盖history→active、active内部及Patch4边界，但`root_boundary_weight`默认0，不改变正式主loss。

Root heading额外使用两项互补的physical-space监督。两项都从投影前endpoint `raw_x0=x_current+beta*v_pred`读取heading，并乘`1/max(beta,root_heading_beta_min)`，抵消endpoint恢复对velocity梯度的一次beta缩放：`root_heading_cosine_weight`控制单位圆角度loss，`root_heading_vector_weight`控制raw heading到GT单位向量的SmoothL1。后者在精确180度反向mode仍有非零梯度并约束投影前模长；cosine保留圆周几何。为避免raw heading接近零时归一化Jacobian与倒beta权重叠加，投影分母最小为`root_heading_cosine_min_norm=0.05`，且低于该阈值的frame不参与cosine，仅由raw-vector项恢复模长与方向。训练日志同时保存未加权/加权loss、raw norm均值与p10、low-norm ratio及`dot<-0.9`的antipodal ratio。现有ideal flow-v和K>1 beta平方补偿off-path主loss保持不变；heading两项是辅助监督，不替代主目标。

Validation整轮只在开始时把EMA shadow换入模型一次，所有loss probe和inline generation复用这组参数，并在epoch-end恢复训练参数。临时`collected_params`恢复后立即释放且不进入checkpoint；异常由统一callback回滚，checkpoint若在EMA参数仍激活时触发会fail-fast，避免把EMA权重误写成主训练state。

计算层保持相同可见性语义但压缩无效工作：Root在有future约束时向量化打包`visible motion | future root`，无future时只运行batch内最大visible prefix；Body同样只运行最大visible prefix并把尾部补零。FlashAttention varlen打包与恢复使用布尔prefix mask，不再逐row执行Python scalar读取。pure-noise frontier仍保留在persistent state且不进入本步Transformer，compact只删除已被mask排除的pointwise/FFN计算，不改变non-causal visible attention。

## 已实现的XZ轨迹条件训练

每个rollout先从translation-anchored、random-yaw增强后的clean normalized root采样一次absolute XZ约束计划；teacher/self-forcing所有step复用这份计划，再按当前active窗口将其编译为两类只读条件。约束只包含XZ，不暴露root y或heading。

```text
dense trajectory（正式baseline为100%）:
    从首个active token开始标记动态future范围内的所有真实帧XZ

sparse waypoints（当前正式baseline关闭，保留为消融）:
    在active + future lookahead内随机标记1–4个独立帧XZ

future goal（当前正式baseline关闭，保留为消融）:
    active约束为空，只标记一个严格位于首个active band之后的未来帧XZ
    样本没有真实future frame时退化为一个active waypoint
```

稀疏性在`[B,T,4,5]`的frame维采样，不会因选择一个token而自动暴露该token全部四帧。`max_horizon_token`只是训练上限；每个sample独立从`[0,min(max_horizon_token,真实可用future)]`均匀采样一个`future_horizon_tokens[B]`，并在整次K-step rollout内冻结。真实可用future在采样前扣除`K-1`个rollout位置，因此后续commit不会把10-token lookahead逐步缩成6。absolute约束计划额外覆盖`K-1`次active后移；每个commit再从`active_start+1`到`active_end+sampled_horizon`编译按absolute timeline position压紧的候选superset。每个denoise microstep以`history_mask | generation_mask`的真实可见末端为边界，只保留其后至多该sample所采horizon范围内的候选。这样cold start中尚未可见的active XZ会先作为future条件，成为motion query后立即从future attention视图移除，同一absolute token不会扮演两种query。horizon为0只关闭not-yet-visible future读取，当前可见motion上的XZ仍然存在；constraint dropout才会同时删除current/future XZ。goal随self-forcing窗口前移会自然从future候选变为active内约束，不会重新采样。lookahead只使用当前sample/span内真实有效token，不用零值冒充未来轨迹。future value在进入projection前按feature mask清零，因此未观测的root y与heading不能泄露给Root Stage。Body Stage不读取raw active/future XZ，只读取Root Stage组合后的唯一clean root派生的local root和heading condition。

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
    dense_probability: 1.0
    waypoint_probability: 0.0
    goal_probability: 0.0
    max_waypoint_count: 4
```

三种采样概率之和必须为1。constraint dropout在采样计划之后按sample清空整份约束，因此它是唯一产生无轨迹条件样本的机制；mode sampler不会用短span意外制造无约束样本。训练入口要求`max_horizon_token`为正、两种dropout位于`[0,1]`且`max_waypoint_count>0`，并直接检查所需statistics/checkpoint/data，不再依赖人工状态字符串。

文本条件遵循FloodDiffusion的局部性：HumanML3D的一条动作caption在span内重复；BABEL的区间caption编译到对应token。Root/Body Stage的每个motion query直接cross-attend自身prompt；后续Transformer层仍可通过可见motion token之间的non-causal self-attention传播已经注入的文本信息，因此该协议是`direct token-aligned cross-attention`，不是严格文本隔离。pure-noise frontier不进入当前attention，未到达active band的未来prompt不能提前传播。训练以sample级text dropout构造空文本分布，推理CFG复用同一空文本语义。

Validation使用固定seed的独立probe：teacher cold强制parent从真实序列第0 token开始，并固定`source_start=0, context=0, previous-root invalid, H=0, K=1`；teacher continuation在确定性的中间parent窗口上按early/middle/late标签选择合法`H>=1`，默认中点probe等价于固定`H=5,K=1`配置；启用self-forcing后再增加固定`H=1,K=5` probe。source parent、phase、root/body noise和文本选择对同一batch保持确定，使checkpoint loss可横向比较。

## 当前待讨论问题

- root/body flow-v的最终相对权重与是否加入decoded auxiliary loss。
- persistent cold落地后的实际吞吐、各生命周期phase的采样均匀性以及是否需要decoded foot auxiliary，需通过现有checkpoint微调消融确定；cold总概率、ideal/persistent比例和首轮2-commit上限已经冻结。
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
