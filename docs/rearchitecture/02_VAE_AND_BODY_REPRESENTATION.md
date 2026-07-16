# 02 Body VAE 与显式动作表示

状态：`TOKENIZER_AND_RUNTIME_READY / LDF_TRAINING_PENDING`

## 目标

新版动作状态由explicit root与latent body组成。VAE只编码body，root保持为LDF可直接生成和约束的结构化变量。该设计参考ARDY的hybrid representation、Patch4 tokenizer和local-root-conditioned decoder，但保留FloodDiffusion的因果卷积工程基础。

## 物理表示

```text
root_motion [B,F,5]
  root xyz                                3
  cos/sin global heading                 2

body_motion [B,F,265]
  non-root planar-root-relative positions 21*3 = 63
  global joint rotation 6D                22*6 = 132
  backward global joint velocities        22*3 = 66
  foot contacts                                  4
```

positions只从x/z减去planar root，y保留世界高度。rotations和velocities保持全局朝向，因此随机yaw增强必须同步旋转root、positions、rotations和velocities。

运行时代码只允许通过`utils/motion_process.py`处理该物理表示。该模块拥有全部物理维度常量、heading单位圆投影与唯一local-root codec；LDF/VAE condition模块只定义tensor合同并单向依赖它，不能重新实现root派生。公共接口固定为`pack_body/unpack_body`、`rotation_to_matrix/matrix_to_rotation`、`compute_joint_velocities/build_motion`、`project_root_heading`、`recover_root_yaw/recover_local_root/recover_joint_positions`和两种yaw旋转函数；模块不包含263D、272D或trajectory7分支。HumanML 263D的root积分、世界关节恢复、层级rotation组合与唯一263→265转换只存在于离线`tools/convert_motion_263_to_265.py`，Dataset和模型不得导入该工具。

文件职责按依赖方向冻结为`conditions → motion_process → coordinate_transform`。`utils/coordinate_transform.py`只保留无表示所有权的yaw、XZ点和XZ向量坐标变换；`utils/training/vae/checkpoint.py`提供训练、LDF、评测和runtime共用的EMA checkpoint加载函数；离线`tools/convert_motion_263_to_265.py`和`tools/build_motion_artifact.py`负责source转换与最小NPZ写入。运行时已删除`utils/motion_artifact.py`，VAE不建立bundle、模型身份或artifact血缘层。

## Token与因果合同

- `utils/token_frame.py`唯一拥有`FRAMES_PER_TOKEN=4`及frame/token换算；模型输入帧数必须整除4，模型、Dataset和runtime不得自行向上取整或静默截断。
- `utils/token_frame.py`同时唯一定义`MOTION_FPS=20.0`。VAE直接读取这个项目时间协议，不在`vae.params`中暴露一个可与数据协议产生分歧的`fps`实验参数。
- encoder在进入网络前将连续四帧reshape为一个patch；不存在首帧特殊token。
- encoder/decoder均在token轴使用causal convolution，token t不读取t+1。
- decoder每次读取一个128D latent token和一个`[4,4]` local-root patch，严格输出四帧。
- decoder cache由调用者以`VAEDecoderState(caches)`显式持有，不允许module-global cache或`first_chunk`开关；模型只更新cache tensor。
- 当前encoder与decoder的精确因果历史均为24 tokens（96 frames）；模型分别通过`encoder_context_tokens`与`decoder_context_tokens`公开该值。

## BodyVAE公共接口

公共方法按latent空间固定语义，不使用布尔参数切换输入解释：

```text
encode                       physical body -> raw posterior（VAE训练）
decode / decode_step         raw latent -> physical body
tokenize                     full physical body -> normalized deterministic mu
tokenize_window              context + active body -> normalized active mu（LDF训练）
detokenize / detokenize_step normalized latent -> physical body
forward                      只接受VAEInput
```

`BodyVAE`只持有网络、前向所需statistics buffers与上述计算接口。构造时直接读取普通`motion_stats.npz`与可选`latent_stats.npz`，只校验数组shape、finite和正std，不解析metadata，也不生成模型/session identity。训练、LDF、评测和runtime统一调用`load_vae_checkpoint(model, checkpoint_path, use_ema=True)`加载同一个训练checkpoint。

## Local root

```text
local_root_motion [B,T,4,4]
  [yaw_rate, current-heading-local vx, current-heading-local vz, root_y]
```

差分使用当前帧减前一帧。裁剪位于原序列中间时必须携带真实previous root；cold start的首帧yaw/velocity无效并置零，root height仍有效。该处有意不同于ARDY公开代码的forward/world-axis velocity，以满足persistent decoder在token边界不读取未来帧的要求。

## 统计所有权

```text
local root statistics    [4]
body continuous stats    [261]
latent mu statistics     [128]
```

VAE statistics只拥有decoder local root与body；LDF global root statistics由后续LDF数据协议独立生成。contacts保持0/1，不参与z-score。physical stats只从train split计算；latent stats只在VAE冻结后用deterministic posterior mu计算。两个statistics文件都是普通NPZ：physical文件保存四组mean/std，latent文件只保存`mean/std`。latent buffers为non-persistent，不进入训练checkpoint。正确checkpoint与statistics配对由实验配置负责，模型不追踪文件hash、yaw policy或artifact manifest。

VAE train collator在物理空间施加独立`Uniform(0,2π)`全局yaw，因此第一帧yaw不会固定为零。该变换同步作用于root xz/heading、root-relative positions、global rotations、global velocities和previous-root boundary；contacts保持不变，current-heading-local root velocity在变换前后保持不变。Dataset本身不做任何随机变换。physical statistics使用`0/90/180/270`度四点quadrature；由于所有受影响通道都对yaw的cos/sin线性，该方法精确匹配连续均匀yaw的一阶与逐维二阶矩，同时保持统计结果确定性。

ARDY使用一个physical stats集合的root/local/body切片，并为tokenizer latent保存额外统计；Floodcontrol保留physical/latent分离，但使用普通数组文件而非强血缘协议。

## VAE与loss

encoder输出`mu/logvar [B,T,128]`。VAE训练通过reparameterization采样，validation同时报告sample reconstruction与deterministic-mu reconstruction，LDF训练固定使用deterministic `mu`。正式tokenizer固定为同一checkpoint的EMA encoder与EMA decoder；raw/EMA混用和缺少EMA的checkpoint均fail-fast。

训练目标冻结为`L_total = L_recon + 0.01 L_skate + 1e-5 L_KL`，不使用KL warmup。其中`L_recon`由三个分别归一化、分别masked-mean的continuous SmoothL1 block（position、rotation、velocity）和contact BCE相加，避免132D rotation仅凭维数支配重建目标。`L_skate`从预测global foot positions的相邻帧差分直接派生足速，再使用GT contact加权；它排除crop首帧、padding及无效position transition，因此直接约束最终展示使用的位置轨迹。独立velocity feature仍只受velocity reconstruction监督，velocity consistency默认关闭。geodesic、FK-to-GT、direct/FK position consistency和backward velocity consistency不进入正式配置，只保留为后续独立消融能力。

## LDF训练时在线编码

正式LDF训练不读取预编码latent artifact。Dataset只返回完整确定性的physical `root5/body265`；LDF collator派生crop、encoder context与previous-root边界，未来trainer再由冻结EMA encoder在线计算deterministic `mu`：

```text
load deterministic root5/body265
  -> choose token-aligned crop and encoder history context
  -> apply one shared yaw to root/body/context/previous boundary
  -> frozen EMA tokenizer.tokenize_window() under no_grad
  -> normalized deterministic mu
  -> construct HybridMotion(root_motion, latent_motion)
  -> LDF v-predict
```

LDF训练不得采样posterior，也不得通过VAE encoder反传。一个training batch只执行一次encoder；scheduled training或self-forcing的多次LDF update复用同一份detached target。当前data contract使用普通worker RNG执行train crop/可选yaw，validation固定取前缀且关闭随机增强；更强的epoch/DDP可复现策略待真实trainer恢复时确定，不在Dataset中加入hash协议。

`utils/training/ldf/lightning_module.py`是唯一VAE→LDF训练边界：从正式checkpoint加载并冻结EMA VAE，以`tokenize_window()`在线产生normalized deterministic μ；同时通过LDF自己的`normalize_root()`归一化physical active root并reshape为`[B,T,4,5]`，最终构造共享token mask的`HybridMotion`。LDF latent维度、local-root statistics和collator需要的encoder context长度都直接来自VAE实例，不在配置中维护第二份数值。bridge必须逐样本验证active root token数等于去除context后的body token数，不能只比较批内最大tensor shape。root/latent在normalization后都重新清零padding。

由于encoder是token-causal，同一个active token不能因LDF crop起点不同而得到不同target。每个样本在线encode时携带`min(window_start_token, encoder_context_tokens)`个真实历史token；当前每个residual block含两个kernel为`k`的causal convolution，因此最大历史为`encoder_layers * 2 * (k - 1)`个token。批内输入固定为`[真实context | active crop | 右padding]`，`context_token_count [B]`给出逐样本active offset；`tokenize_window()`一次batch前向后逐样本gather active deterministic μ并将active padding置零。窗口接口不返回padding posterior，从而不存在对伪造logvar采样的歧义。真实序列起点历史不足由causal convolution自身的左边界处理，collator不得插入假零token。context、active crop与previous-root boundary必须共享同一个yaw。

latent statistics仍是必需的独立小型文件，但不保存逐样本latent。VAE冻结后使用同一EMA encoder、相同context协议和确定性均匀yaw扫描train split，得到`mean/std [128]`。正式LDF训练始终在线编码，不提供逐样本latent cache工具。

正式训练从每段动作中随机裁剪20–200帧（1–10秒、四帧对齐）的片段，共训练300k optimizer steps。优化采用FloodDiffusion验证过的AdamW `2e-4`与constant-after-warmup schedule：前1k steps线性升至基础学习率，之后保持恒定，便于在300k后按验证曲线直接续训。HumanML和HumanML+BABEL配置统一使用单卡、`strategy: auto`和实际batch size 128，不通过梯度累积或静默缩小batch改变有效训练协议。

## LDF接口补充

body保存global rotations，因此pure-noise cold start还需要绝对heading。Body Stage从唯一clean root派生首个有效帧的`[cos(yaw), sin(yaw)]` heading condition；该值与local root一起在stage boundary detach，不读取raw constraints，也不把Body loss传回Root Stage。

## HumanML T2M评测适配

HumanML预训练movement/motion evaluator固定消费标准263D特征，因此FID、Diversity、Matching Score与R-Precision不得直接读取root5/body265，也不为新版表示重新训练另一套embedding模型。统一通过`metrics.humanml.convert_root5_body265_to_humanml263()`恢复标准root4、heading-canonical positions、21-joint local rotation6d、heading-local joint displacement和contacts，再沿用FloodDiffusion相同的evaluator mean/std与checkpoint。

标准HumanML每一行保存当前pose以及到下一pose的forward transition，所以`F`帧physical motion只严格确定`F-1`行263D。`tail="drop"`用于验证可观测部分的数学round-trip；与固定长度FloodDiffusion结果做正式横向比较时显式使用`tail="approximate"`，以最后一次观测transition外推不可见的尾transition并保持`F`行，避免T2M内部`length // 4`少掉一个完整movement token。128条真实验证中该保长适配的FID漂移为`1.49e-4`，而drop后直接对完整reference评测的漂移为`2.59e-3`。适配器必须对全局translation/yaw不变，并通过原始263 → root5/body265 → 263 round-trip回归。`tools.compare_humanml_adapter`同时报告分feature误差、严格drop、保长近似及drop-vs-full的预训练T2M embedding/FID漂移。

## 当前阻塞

HumanML3D-only VAE已完成300k steps。`step_299999.ckpt`中的EMA encoder+decoder可由公共checkpoint loader直接加载；历史`latent_stats.npz`仍可作为现有实验资产使用，新统计工具则用显式seed普通随机生成器扫描Dataset full sample。完整EMA validation的deterministic-mu total为`0.0051869`，sample total为`0.0051965`；posterior sigma mean为`0.00341`。LDF real-context collator、逐样本window tokenization、正式128D EMA在线编码和normalized HybridMotion bridge均已实现；正式H/G/F/C训练窗口、self-forcing、noise/beta、condition与root/latent v-predict loss仍未接入，因此真实LDF训练继续fail-fast。当前`root_stats.npz`只匹配临时简单crop/rebase策略，正式窗口冻结后必须通过共享sampler重新计算。Web已通过`InferenceSession`在每个LDF commit后原子执行`detokenize_step()`，并将结果作为严格四帧chunk传输；模型加载仍等待正式LDF checkpoint合同。现有300k权重是在position-derived skating启用前训练的，重新加载不会让该checkpoint获得新的loss收益。
