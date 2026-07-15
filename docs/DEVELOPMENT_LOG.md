# Floodcontrol 开发变更记录

本文件记录 Floodcontrol 中已经完成的代码相关修改。目的不是复述设计讨论，而是留下“实际改了什么、为什么改、怎样验证”的可追溯事实。

## 记录规则

- 采用时间正序，只在文件末尾追加新条目；不得覆盖、删除或偷偷修正旧记录。
- 每个完成的代码相关任务至少记录：日期、任务、改动内容、改动理由、验证结果和涉及文件。
- 代码、配置、测试、脚本、运行时协议，以及会约束后续实现的架构文档变更都需要记录。
- 纯只读分析、没有落盘修改的讨论不记录。
- 验证没有执行时必须明确写“未运行”，不能把静态检查或代码阅读描述成测试通过。
- 后续若撤销某项修改，追加一条 revert 记录并引用原条目，不改写历史条目。

## 条目模板

```text
## YYYY-MM-DD · 简短任务名称

类型：代码 / 配置 / 测试 / 脚本 / 文档协议 / 仓库流程

改动内容：
- ...

改动理由：
- ...

验证：
- ...

涉及文件：
- `path/to/file`

后续事项：
- 无；或明确列出尚未完成的工作。
```

## 2026-07-14 · 建立开发日志与自动维护规则

类型：仓库流程 / 文档

改动内容：

- 新增本开发变更记录，定义只追加的条目格式和真实性要求。
- 新增仓库级 `AGENTS.md`，要求 coding agent 在每次代码相关修改完成后、最终回复前更新本文件。
- 新增兼容入口 `AGENT.md`，将只识别单数文件名的工具指向正式规则。
- 在仓库 `README.md` 中加入开发日志入口。

改动理由：

- 新版 Floodcontrol 会经历多阶段清理和重构，需要把实际实现变更与架构讨论分开追踪。
- 强制记录改动理由与验证结果，可以减少后续会话重复调查历史决策，也避免把设计文档误认为已经实现。

验证：

- 已确认 `AGENTS.md`、`AGENT.md` 和 `docs/DEVELOPMENT_LOG.md` 均位于 Floodcontrol 仓库内。
- 已检查主 README 的相对链接指向 `docs/DEVELOPMENT_LOG.md`。
- 本任务未修改 Python 代码，因此未运行模型或单元测试。

涉及文件：

- `AGENTS.md`
- `AGENT.md`
- `README.md`
- `docs/DEVELOPMENT_LOG.md`

后续事项：

- 从下一次代码相关修改开始，在任务完成后持续追加记录。

## 2026-07-14 · 落地Hybrid LDF模型核心并删除旧控制体系

类型：代码 / 配置 / 测试 / 文档协议 / 文件删除 / 迁移守卫

改动内容：

- 将模型公共入口重写为`RootTransformer/BodyTransformer/LDF`，生成状态统一为`HybridMotion(root_motion, latent_motion)`。
- 实现Root-first/Body-second forward、normalized v-predict、物理heading单位圆投影、backward/current-heading-local root codec及训练时Root到Body的detach边界。
- 实现`nocfg/joint/separated` constraint CFG；Root CFG先形成唯一clean root，Body CFG branches共享其local-root condition。
- 实现显式`LDFStreamState`、三角beta、逐token commit、rolling、RNG snapshot/restore和离线/流式Hybrid更新；LDF不持久化attention KV cache。
- 重写`utils/conditions/ldf.py`，加入Hybrid/LDF dataclass、condition校验、absolute-window裁剪、4-frame token编译、future packing和CFG branch创建函数。
- 将Wan工具层收敛为普通1D Transformer block、显式position-ID RoPE、通用FlashAttention/SDPA；移除专用trajectory attention路径。
- 新增`configs/ldf_core_tiny.yaml`和21项模型核心、condition、CFG、stream、迁移守卫测试。
- 物理删除旧附加控制网络、专用轨迹编码器、tiny专用LDF、外置root planner、RootPlan、旧LDF训练包及其配置/诊断脚本；保留通用route、timeline和trajectory metric代码。
- 将`train_ldf.py`与Web模型入口改为`BLOCKED_ON_STRICT4_VAE`显式错误，防止回退到旧263D VAE或产生随机ImportError。
- 更新README和架构文档，区分“模型核心已实现”与“strict-4 VAE/data/training/Web尚未接线”。

改动理由：

- root必须成为LDF原生生成变量并通过唯一clean/local-root边界控制body，而不是继续作为ControlNet旁路或decoder后的替换结果。
- 分支内masked observation与separated CFG可以提供可调轨迹约束，同时不污染persistent noisy state。
- 显式Hybrid stream state使root/latent共享同一三角时间面、commit和事务恢复协议，为后续VAE decoder state接线提供稳定边界。
- 旧训练与Web当前依赖错误的full-motion VAE协议；明确阻断比保留语义错误的临时适配层更安全。

验证：

- `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests -q`：`21 passed`。
- 全仓Python文件执行`py_compile`：通过。
- 已验证`models.diffusion_forcing_wan.LDF`可由`configs/ldf_core_tiny.yaml`实例化。
- 已验证训练和Web入口均抛出包含`BLOCKED_ON_STRICT4_VAE`的明确错误。
- 排除历史设计文档后，全仓搜索无旧控制网络类、专用轨迹编码器、FlexTraj、旧轨迹CFG字段、外置root planner或RootPlan活跃引用。
- 未运行真实训练、旧checkpoint加载、VAE decode或Web生成；这些能力按本里程碑要求主动阻断。

涉及文件：

- `models/diffusion_forcing_wan.py`
- `models/tools/attention.py`
- `models/tools/wan_model.py`
- `utils/conditions/ldf.py`
- `utils/inference/`
- `utils/training/`
- `web_demo/model_manager.py`
- `web_demo/runtime/model_loader.py`
- `train_ldf.py`
- `configs/ldf_core_tiny.yaml`
- `configs/stream.yaml`
- `tests/`
- `README.md`
- `docs/rearchitecture/01_MODEL_ARCHITECTURE_AND_IO.md`
- `docs/rearchitecture/06_LDF_IMPLEMENTATION_DESIGN.md`
- 删除：`models/tools/wan_controlnet.py`、`models/tools/traj_encoder.py`、`models/diffusion_forcing_wan_tiny.py`、外置root planner与旧LDF训练/配置/专用脚本。

后续事项：

- 实现strict 4-frame body VAE、causal decoder state及full/cached decode parity。
- 重构数据集，生成explicit root、body latent和真实root/local statistics。
- 接入root/latent velocity loss、scheduled training与self-forcing后恢复`train_ldf.py`。
- runtime使用`create_window_condition()`编译route observations，并在commit事务中接入VAE decoder后恢复Web生成。

## 2026-07-14 · 统一LDF阶段、预测与流式辅助接口命名

类型：代码 / 测试 / 文档接口

改动内容：

- 将内部Transformer阶段基类从`_StageTransformer`改为`TransformerStage`，继续由`RootTransformer`和`BodyTransformer`继承。
- 将condition准备、有效长度、root恢复、Root/Body预测和CFG组合辅助函数统一为`_prepare_condition`、`_get_valid_lengths`、`_recover_root`、`_predict_root`、`_predict_body`和`_compose_cfg`。
- 保留已经能够准确表达职责的`_as_stats`和`_local_root`。
- 将流式输入与滚窗函数改为`_create_step_input`和`_roll_window`。
- 将snapshot公共接口成对改为`create_stream_snapshot()`和`create_stream_state_from_snapshot()`，并同步测试与LDF实现文档。

改动理由：

- `TransformerStage`实际包含输入/条件投影、多层`WanTransformerBlock`与输出投影，使用Stage可避免与单层Transformer block混淆。
- 去掉`branch`、`separated`等偏实现细节的命名，让调用处直接表达“预测Root/Body”和“组合CFG”的目的。
- 滚动操作改变的是active window而非执行一次denoise step，因此使用`_roll_window`保持调度语义准确。
- snapshot转换只在Tensor字典和`LDFStreamState`之间进行，不涉及文件I/O；成对的create接口明确输入与产物。

验证：

- `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests -q`：`21 passed`。
- `models/diffusion_forcing_wan.py`及受影响测试执行`py_compile`：通过。
- 已搜索受影响代码、测试和LDF实现文档，确认本轮替换的旧名称无活跃引用。

涉及文件：

- `models/diffusion_forcing_wan.py`
- `tests/test_ldf_cfg.py`
- `tests/test_ldf_stream.py`
- `docs/rearchitecture/06_LDF_IMPLEMENTATION_DESIGN.md`
- `docs/DEVELOPMENT_LOG.md`

后续事项：

- 本次仅调整命名，不改变Root-first/Body-second forward、CFG公式或Hybrid流式状态语义。

## 2026-07-14 · 拆分absolute timeline与generation-centered RoPE坐标

类型：模型合同 / Root-Body位置协议 / 流式调度 / 测试 / 文档

改动内容：

- 将`LDFInput.position_ids`拆分为`timeline_position_ids`与`rope_position_ids`：前者是absolute stream坐标，后者以当前first-generation token为0。
- 在`LDFInput.validate()`中校验motion timeline连续、两套坐标只相差单一`rope_origin`、first-generation RoPE位置为0，以及future timeline严格位于当前motion window之后。
- 将future字段改为`future_timeline_position_ids`；Root Stage通过`LDFInput.timeline_to_rope()`派生generation-centered future RoPE位置。
- 修复`create_window_condition()`在`window_origin>0`时仍从`window_tokens`生成future位置的问题；future absolute timeline现在从`window_origin + window_tokens`开始。
- `triangular_beta()`与stream commit继续只使用absolute timeline IDs；`WanSelfAttention`和`WanTransformerBlock`只接收`rope_position_ids`。
- 保持Root Stage可见`T+F`个current/future token，但只输出前`T`个root prediction；Body Stage始终只处理`T`个motion token，future horizon不扩大noisy state、beta或Body长度。

改动理由：

- old/Floodmain曾把latent window长度与trajectory attention长度合并为同一个`seq_len`，使RoPE、time embedding、text context和future horizon相互污染；分离调度坐标与模型位置坐标可以从合同层阻止该问题复发。
- generation-centered RoPE与ARDY的显式token index同构：history为负、first generation为0、future为正，同时absolute timeline仍可稳定承担rolling与三角调度。
- rolling后future token必须继续absolute timeline，而不能与当前窗口后半段复用position ID。

验证：

- `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests -q`：`24 passed`。
- 受影响的模型、condition和测试文件执行`py_compile`：通过。
- 新增测试覆盖`window_origin>0`的future absolute IDs、timeline到RoPE转换、future/current重叠拒绝、Root future RoPE拼接及Body长度不随future horizon扩展。
- 搜索确认LDF合同、模型、测试和架构文档中不再存在旧`future_position_ids`或单一`LDFInput.position_ids`引用。

涉及文件：

- `utils/conditions/ldf.py`
- `models/diffusion_forcing_wan.py`
- `models/tools/wan_model.py`
- `tests/test_ldf_conditions.py`
- `tests/test_ldf_forward.py`
- `tests/test_ldf_cfg.py`
- `docs/rearchitecture/01_MODEL_ARCHITECTURE_AND_IO.md`
- `docs/rearchitecture/06_LDF_IMPLEMENTATION_DESIGN.md`
- `docs/DEVELOPMENT_LOG.md`

后续事项：

- strict-4数据与训练接线时必须直接提供两套position IDs，并使用相同回归测试验证随机训练裁窗与stream rolling的一致性。
- runtime route compiler恢复时只负责absolute future timeline；generation-centered RoPE转换继续由LDF合同统一完成。

## 2026-07-14 · Public v0.1首版发布卫生清理

类型：公开仓库准备 / 配置 / 工具 / 文档 / 文件删除

改动内容：

- 新增根目录`.gitignore`，排除Python缓存、测试缓存、日志、实验追踪目录、本地数据/依赖/输出目录以及checkpoint和数组artifact。
- 将`configs/paths_default.yaml`中的开发机绝对路径替换为相对默认目录，并支持`FLOODCONTROL_DATA/FLOODCONTROL_DEPS/FLOODCONTROL_OUTPUTS`环境变量覆盖。
- 移除`run_pytest.sh`中的个人Conda路径fallback，保留当前Conda环境或`python3`的可移植选择。
- 删除仍调用不存在的legacy LDF配置、ablation工具和stream benchmark的`bench_body_7d_ablation.sh`，以及包含个人数据机绝对路径的一次性`shuffle_train_hard_txt.sh`。
- `pretokenize_t5_text.py`不再静默使用已删除的`configs/ldf.yaml`，改为要求显式`--config`。
- `metrics.stream`改为使用仓库内motion recovery，并提供本地numeric summary，解除对同级FloodNet/eval包的运行时依赖。
- README、Web README、Web包标识和rearchitecture索引更新为当前`MODEL_CORE_IMPLEMENTED / BLOCKED_ON_STRICT4_VAE`真实状态，不再宣称旧Web生成可用或依赖本地同级仓库。
- 更新`compute_z_stats.py`说明，移除已经删除的`WanModel.load_z_stats/mask_emb`旧所有权描述。

改动理由：

- public首版不能提交缓存、日志、模型artifact、credential或开发机绝对路径。
- 首版只保证Hybrid LDF模型核心；保留会调用不存在旧入口的脚本和可用Web demo说明会误导使用者。
- 仓库内工具应能独立导入，不应要求发布范围之外的FloodNet Python package。

验证：

- `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests -q`：`24 passed`。
- 全仓Python文件执行`py_compile`：通过。
- `metrics.stream`、新版LDF模型和condition模块独立import smoke：通过。
- shell入口执行`bash -n`：通过。
- credential扫描只命中从环境变量读取的W&B配置，没有发现硬编码token、password或private key。
- 大文件/模型artifact扫描未发现待提交文件；`web_demo/app.log`为零字节并由`.gitignore`排除。

涉及文件：

- `.gitignore`
- `README.md`
- `configs/paths_default.yaml`
- `scripts/run_pytest.sh`
- `tools/pretokenize_t5_text.py`
- `tools/compute_z_stats.py`
- `metrics/stream.py`
- `web_demo/README.md`及Web包标识
- `docs/rearchitecture/README.md`
- `docs/DEVELOPMENT_LOG.md`
- 删除：`scripts/bench_body_7d_ablation.sh`、`scripts/shuffle_train_hard_txt.sh`

后续事项：

- public仓库当前未声明统一项目许可证；第三方来源文件继续保留各自版权头，项目级许可证需由仓库所有者单独决定。
- strict-4 VAE、真实训练配置和Web runtime完成后再发布下一里程碑。

## 2026-07-14 · 删除重复的AGENT入口文件

类型：仓库维护 / 文件删除

改动内容：

- 删除根目录`AGENT.md`。
- 保留Codex正式识别并覆盖整个仓库的`AGENTS.md`作为唯一agent规则来源。

改动理由：

- `AGENT.md`仅重复指向`AGENTS.md`，不是本仓库需要维护的正式规则文件；同时保留两者会造成规则入口含糊。

验证：

- 搜索仓库引用后确认，除`AGENT.md`自身外，其他文件只引用`AGENTS.md`。
- 本次只删除重复文档入口，没有修改运行代码，因此未运行模型或测试套件。

涉及文件：

- 删除：`AGENT.md`
- `docs/DEVELOPMENT_LOG.md`

后续事项：

- 无。

## 2026-07-14 · Strict-4 Body VAE、body265协议与原生旋转数据边界

类型：模型重构 / 数据协议 / 训练与推理接口 / 测试 / 文档

实际改动内容：

- 将旧full-motion `VAEWanModel/WanVAE_`物理替换为body-only `BodyVAE/Strict4CausalVAE`。encoder先显式patch四帧再在token轴执行causal convolution；decoder读取128D latent与`[4,4]` local-root patch并严格输出四帧。
- 新增`VAEInput/VAEPosterior/BodyPrediction/VAEPrediction/VAEDecoderState`合同。decoder cache改为调用方持有的不可共享状态，删除module-global cache、`first_chunk`和`1+4n`首帧特殊协议。
- 冻结root5/body265表示：21个非root位置、22个global rotation6d、22个backward global velocity和4个contact；实现pack/unpack、yaw同步旋转、backward velocity、contact与backward/current-heading-local root codec。
- 增加root/local-root/body-continuous/latent-mu四组统计所有权。真实模型必须加载physical statistics；VAE冻结前允许显式缺少latent stats，但`tokenize/detokenize`在latent stats未就绪时会fail-fast。
- 实现分block normalized SmoothL1、contact BCE、ARDY式0.01 skating loss和warmup KL；geodesic、FK-to-GT、direct/FK position consistency和backward velocity consistency为默认零的独立可选项，FK相关项缺少版本化HumanML22 skeleton时明确报错。
- 新增strict4 artifact dataset/collate、native-rotation NPZ预处理、train-split motion statistics和deterministic-mu latent artifact工具。输入缺少原生rotations时明确报`STRICT4_NATIVE_ROTATIONS_REQUIRED`，不使用IK或旧263D降级。
- LDF Body Stage增加从唯一clean root派生的首有效帧heading condition；heading与local root一起在训练stage boundary detach，保持Body loss不反传Root Stage，所有CFG body分支共享同一root/heading。
- 将共享token/frame工具及HumanML/BABEL token crop改为`token k -> frames [4k,4k+4)`；删除旧特殊映射和越界fallback。
- 删除仍依赖已不存在`stream_execution`、旧root feedback和`first_chunk`的`StreamRuntimeSession`；Web入口继续fail-fast，等待正式VAE/latent artifact与commit-time decoder事务接线。
- 删除旧`configs/vae_wan_1d.yaml`，新增正式/tiny strict4配置；重写`train_vae.py`使用新dataset、BodyVAE和loss，并在manifest或statistics缺失时明确阻断。
- 新增独立VAE/body表示设计文档，并更新模型、数据、README和Web状态说明。

改动理由：

- 旧VAE把首帧与后续四帧token混合，无法与LDF严格四帧时间轴、逐token commit和稳定cache事务一致。
- explicit root必须保持LDF原生生成变量；body encoder不应重新隐藏root，而decoder需要local root运动学条件来降低foot skating。
- body使用global rotations时，local-root4不包含绝对heading；从clean root补充heading可以解决pure-noise cold start的朝向不可辨识，同时不引入raw constraint旁路。
- 当前HumanML legacy数据没有原生joint rotations，不能假装能够无损生成ARDY式body表示，因此把数据缺失暴露为正式前置条件。

验证：

- `python -m py_compile`覆盖仓库Python文件，通过。
- `python -m pytest -q tests`：44 passed；仅出现运行环境无法初始化NVML的PyTorch warning，不影响CPU测试。
- 新增测试覆盖body265 round-trip、contacts不归一化、strict4映射、backward local root、全局yaw下local velocity不变量、encoder因果性、offline/stream decode parity、session state隔离、snapshot/restore、KL warmup、独立geometry loss开关、短样本overfit、native NPZ到dataset集成和迁移守卫。
- `python train_vae.py --config configs/vae_strict4.yaml`已验证在缺少真实statistics时抛出`STRICT4_MOTION_STATS_REQUIRED`。
- `git diff --check`通过；`ruff`未运行，因为flooddiffusion环境未安装该模块。
- 全仓活跃源码搜索未发现`WanVAE_`、`VAEWanModel`、旧`(crop_start+3)//4`、`4N-3`或可执行`first_chunk`路径。

涉及文件：

- 模型与合同：`models/vae_wan_1d.py`、`models/tools/wan_vae_1d.py`、`models/diffusion_forcing_wan.py`、`utils/conditions/vae.py`、`utils/motion_representation.py`、`utils/token_frame.py`。
- 数据与训练：`datasets/strict4.py`、`train_vae.py`、`utils/training/vae_loss.py`、`tools/preprocess_strict4_smpl.py`、`tools/compute_vae_stats.py`、`tools/pretokenize_body_latents.py`、strict4配置。
- 测试与文档：VAE/token-frame测试、rearchitecture文档、README/Web状态与本日志。
- 删除：旧VAE配置、旧`utils/inference/stream_runtime/session.py`。

尚未完成的后续事项：

- 提供并审核原生SMPL/AMASS rotations数据路径、retarget skeleton offsets/parents和真实train/val/test manifest；当前只用合成native rotations验证了数据工具。
- 用真实train split计算physical statistics，训练VAE，冻结checkpoint并生成带hash的latent statistics/artifacts。
- 真实数据上评估rotation/FK consistency、foot contact阈值、重建和skating，再逐项决定是否启用几何消融loss。
- 将正式latent artifact接入LDF dataset/loss，完成Hybrid commit与`VAEDecoderState`原子snapshot/restore后再解除`train_ldf.py`与Web守卫。

## 2026-07-15 · 统一VAE训练入口命名

类型：训练入口重命名 / 可读性维护

实际改动内容：

- 将`Strict4VAELightningModule`重命名为`VAELightningModule`，避免在通用训练封装名称中重复强调四帧协议。
- 将输入和数据集构造函数统一为`_create_input()`与`_create_dataset()`；将局部变量`module`、`checkpoint`明确为`lightning_module`、`checkpoint_callback`。
- 保留`_step()`名称，因为它是`BasicLightningModule`训练流程调用的覆写钩子。
- 将训练入口的缺失数据与统计错误码改为`NATIVE_ROTATION_DATA_REQUIRED`和`MOTION_STATISTICS_REQUIRED`；模型内部仍严格执行四帧一token合同。

改动理由：

- strict-4是VAE的时间协议，而不是每个训练类和辅助函数都需要携带的实现前缀。训练入口使用职责命名更简洁，也更便于后续保持单一正式VAE实现。
- 显式区分Lightning封装与checkpoint callback，减少`main()`内通用名称带来的歧义。

验证：

- `python -m py_compile train_vae.py`通过。
- VAE合同、数据、loss、模型与迁移守卫测试：`27 passed`；仅有运行环境无法初始化NVML的PyTorch warning。
- `train_vae.py`内搜索确认不再包含`Strict4/strict4`，`git diff --check`通过。

涉及文件：

- `train_vae.py`
- `docs/DEVELOPMENT_LOG.md`

尚未完成的后续事项：

- 本轮不重命名仍承担协议版本辨识职责的dataset、配置文件、artifact contract或底层模型类；是否进一步统一由后续命名讨论决定。

## 2026-07-15 · VAE薄训练入口与模型专属训练包

类型：训练代码重构 / 文件迁移 / 测试

实际改动内容：

- 将`train_vae.py`缩减为只导入并调用`run_vae_training()`的命令行薄入口，不再直接定义LightningModule、dataset、dataloader、logger、checkpoint或Trainer组装逻辑。
- 新增`utils/training/vae/`模型专属训练包：`lightning_module.py`保存`VAELightningModule`，`data.py`负责dataset/dataloader构造，`runner.py`负责训练运行时组装，`losses.py`保存`VAELoss`，并由包级`__init__.py`统一导出。
- 物理迁移并删除顶层`utils/training/vae_loss.py`；测试和`utils.training`公共导出切换到新包路径，不保留旧路径兼容壳。
- checkpoint命名和筛选对齐FloodNet LDF训练语义，统一监控`ckpt_absolute_step`并使用配置中的`validation.save_top_k`。
- 修正tiny配置未定义`motion_stats_path`时的可选字段读取，使`allow_identity_statistics: true`能够继续进入native-rotation manifest守卫，而不是触发OmegaConf缺键异常。
- 新增入口结构、训练包导出、数据前置条件和runner fail-fast回归测试。

改动理由：

- 训练入口应只承担CLI启动职责；模型专属Lightning封装、数据构建与运行时组装放入`utils/training/vae/`后，与FloodNet的`utils/training/<model>/`布局一致，后续增加验证指标或训练策略时无需继续膨胀入口文件。
- loss与LightningModule属于同一个VAE训练子系统，集中所有权比散落在`utils/training`根目录更清楚。
- `ckpt_absolute_step`是基础Lightning模块已经公开的恢复安全步数语义，checkpoint callback应直接复用该语义。

验证：

- `python -m py_compile`覆盖新入口、VAE训练包和新增测试，通过。
- `python -m pytest -q tests`：`48 passed`。
- 直接运行`python train_vae.py --config configs/vae_strict4_tiny.yaml`能够进入新runner，并按设计抛出`NATIVE_ROTATION_DATA_REQUIRED`；未出现旧模块导入错误或OmegaConf缺键错误。
- 全仓活跃源码搜索确认不存在`utils.training.vae_loss`旧导入；`git diff --check`在最终检查中执行。

涉及文件：

- `train_vae.py`
- `utils/training/__init__.py`
- 新增：`utils/training/vae/__init__.py`、`data.py`、`lightning_module.py`、`losses.py`、`runner.py`
- 删除：`utils/training/vae_loss.py`
- `tests/test_vae_loss.py`
- 新增：`tests/test_vae_training.py`
- `docs/DEVELOPMENT_LOG.md`

尚未完成的后续事项：

- 真实VAE训练仍需要原生SMPL/AMASS rotation manifest与train-split motion statistics；本轮只验证训练入口结构和前置条件守卫。
- LDF训练入口仍保持迁移期fail-fast，待strict-4 VAE checkpoint、latent statistics和新数据artifact接入后，再按相同薄入口结构恢复`utils/training/ldf/`。

## 2026-07-15 · 冻结首版VAE三卡训练配置

类型：训练配置 / loss默认值 / 设计文档 / 回归测试

实际改动内容：

- 将正式`configs/vae.yaml`设为300k optimizer steps、三卡DDP、每卡batch size 32（global batch 96）；正式启动时由`CUDA_VISIBLE_DEVICES=2,3,4`绑定物理GPU，配置内部使用`devices: 3`。
- 保持20–200帧（1–10秒）四帧对齐随机crop、AdamW `2e-4`、10k-step warmup与cosine schedule，并将scheduler总步数同步为300k。
- 冻结首版目标为`L_total = L_recon + 0.01 L_skate + 1e-5 L_KL`，关闭KL warmup；position、rotation、velocity和contact继续作为分block reconstruction，几何一致性loss不进入正式配置。
- 将`VAELoss`构造默认值同步为`beta_kl=1e-5`、`kl_warmup_steps=0`，避免未显式传参时偏离正式训练协议；warmup机制仍保留为可显式启用的消融能力。
- 正式实验名统一为`vae_body265`；测试切换到已统一命名的`configs/vae.yaml`和`configs/ldf.yaml`，不恢复已删除的旧strict/tiny配置文件名。
- 更新VAE设计文档，并新增正式配置合同测试，锁定steps、DDP、batch、crop、KL、optimizer和scheduler关键值。

改动理由：

- 300k是首轮可控训练预算，后续可根据验证曲线从checkpoint续训；每卡32能够明确控制三卡显存和global batch，不把原单卡batch 128意外放大为384。
- `1e-5`沿用FloodDiffusion的KL权重；ARDY式0.01 skating loss提供接触期足部静止约束，其余几何loss暂不与表示和数据协议重构同时启用。
- 正式配置、代码默认值和设计文档必须表达同一个训练目标，避免CLI配置遗漏时产生隐式训练差异。

验证：

- `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest -q tests/test_vae_training.py tests/test_vae_loss.py tests/test_migration_guards.py`：20 passed。
- `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest -q tests`：49 passed。
- 运行`python train_vae.py --config configs/vae.yaml`在实例化Trainer/GPU前按设计抛出`MOTION_STATISTICS_REQUIRED`，确认正式入口仍拒绝缺失真实train-split statistics的训练。
- `git diff --check`通过。

涉及文件：

- `configs/vae.yaml`
- `utils/training/vae/losses.py`
- `docs/rearchitecture/02_VAE_AND_BODY_REPRESENTATION.md`
- `tests/test_vae_loss.py`
- `tests/test_vae_training.py`
- `tests/test_migration_guards.py`
- `docs/DEVELOPMENT_LOG.md`

尚未完成的后续事项：

- 训练仍需要设置真实`STRICT4_MANIFEST`和`STRICT4_MOTION_STATS`；当前没有绕过原生SMPL/AMASS rotations与train-split statistics前置条件。
- 正式开跑前检查GPU 2/3/4空闲显存，并用一个短DDP smoke run验证真实数据吞吐和峰值显存，再启动300k训练。

## 2026-07-15 · VAE学习率调度切换为constant-after-warmup

类型：训练配置修正 / 设计文档 / 回归测试

实际改动内容：

- 将正式VAE scheduler从`diffusers.optimization.get_cosine_schedule_with_warmup`切换为`get_constant_schedule_with_warmup`。
- warmup从10k调整为1k steps；达到AdamW基础学习率`2e-4`后，在剩余300k训练预算内保持恒定。
- 删除constant scheduler不需要的`num_training_steps`参数，避免向构造函数传入无效配置。
- 同步更新VAE设计文档和正式配置合同测试。

改动理由：

- FloodDiffusion VAE实际使用AdamW `2e-4`与1k warmup后的恒定学习率；这是当前causal-convolution工程基础上更直接的已验证基线。
- ARDY的cosine schedule横跨4M steps且配合AdamAtan2 `2e-5`，将同一曲线压缩到300k会过早把Floodcontrol学习率降至零，也会使300k后的checkpoint续训变得不自然。

验证：

- VAE训练配置与loss测试：8 passed。
- 全部测试：49 passed；仅有测试环境无法初始化NVML的PyTorch warning。
- `git diff --check`通过。

涉及文件：

- `configs/vae.yaml`
- `docs/rearchitecture/02_VAE_AND_BODY_REPRESENTATION.md`
- `tests/test_vae_training.py`
- `docs/DEVELOPMENT_LOG.md`

尚未完成的后续事项：

- 正式训练前仍需提供真实manifest与motion statistics，并在GPU 2/3/4上完成短DDP smoke run。

## 2026-07-15 · VAE数据划分迁移为FloodNet风格TXT协议

类型：数据协议 / Dataset / 预处理与统计工具 / 配置 / 测试 / 文档

实际改动内容：

- 删除VAE训练对JSONL内嵌split manifest的依赖，改为`train_meta_paths/val_meta_paths/test_meta_paths`；每个TXT与FloodNet一致，每行只保存一个sample id。
- 冻结目录解析语义：TXT所在目录是数据根目录，`artifact_path`指定其下的artifact子目录，Dataset按`<txt-parent>/<artifact_path>/<sample-id>.npz`读取root5/body265。
- 正式配置指向`${dirs.raw_data}/HumanML3D_strict4/{train,val,test}.txt`与`artifacts/`，不再需要`STRICT4_MANIFEST`环境变量；`STRICT4_MOTION_STATS`仍独立提供train-split归一化统计。
- Dataset新增统一`load_artifact_records()`，验证split TXT、sample id、artifact存在性及artifact内部contract version；不提供legacy263或IK回退。
- 预处理工具改为读取已有sample-id TXT，只处理对应native-rotation NPZ，并分别输出`train.txt/val.txt/test.txt`；重复运行某个split不会覆盖另外两个split。增加显式`--skip-missing`以生成排除缺失原生旋转样本后的过滤列表。
- motion statistics工具改为接收一个或多个`--train-meta-paths`，仅在这些训练sample上统计并记录TXT digest。
- deterministic-mu latent工具改为接收train/val/test TXT，输出对应latent split TXT与`latents/<sample-id>.npy`，metadata记录输入split digest，不再生成latent JSONL manifest。
- 更新数据/VAE设计文档、README和配置合同测试；新增合成native rotations经过预处理、split TXT Dataset及statistics生成的集成测试。

改动理由：

- FloodNet已验证的TXT协议把“样本属于哪个split”和“样本内部保存什么”清晰分离，人工检查、替换difficult split和组合多个数据源都比JSONL内嵌split更直接。
- 原预处理每次写`manifest.jsonl`会让后生成的split覆盖先前split；独立`train.txt/val.txt/test.txt`从结构上消除了该冲突。
- artifact自行携带contract/source identity，split TXT只承担集合选择，避免两处重复维护artifact路径和split字段。

验证：

- `python -m py_compile`覆盖Dataset、VAE data builder、预处理、统计、latent工具和相关测试，通过。
- TXT数据管线、训练入口与迁移守卫测试：19 passed。
- 合成native rotations端到端测试成功生成`train.txt`、`artifacts/sample.npz`和`motion_stats.npz`。
- 全部测试：50 passed。
- `python train_vae.py --config configs/vae.yaml`在缺少真实statistics时仍按设计于Trainer/GPU初始化前抛出`MOTION_STATISTICS_REQUIRED`。
- 全仓活跃配置、工具、Dataset、测试和设计文档搜索不再存在`manifest_path`、`--manifest`或`manifest.jsonl`协议引用；`git diff --check`通过。

涉及文件：

- `configs/vae.yaml`
- `datasets/__init__.py`
- `datasets/strict4.py`
- `utils/training/vae/data.py`
- `tools/preprocess_strict4_smpl.py`
- `tools/compute_vae_stats.py`
- `tools/pretokenize_body_latents.py`
- `tests/test_vae_data_pipeline.py`
- `tests/test_vae_training.py`
- `README.md`
- `docs/rearchitecture/02_DATA_PIPELINE.md`
- `docs/rearchitecture/02_VAE_AND_BODY_REPRESENTATION.md`
- `docs/DEVELOPMENT_LOG.md`

尚未完成的后续事项：

- 需要获得与HumanML sample id对应的原生SMPL/AMASS rotations，分别用现有HumanML train/val/test TXT生成`HumanML3D_strict4`目录；缺少native NPZ的HumanAct12样本应通过显式过滤列表或`--skip-missing`排除并审核数量。
- 数据生成完成后计算真实`motion_stats.npz`，设置`STRICT4_MOTION_STATS`并执行三卡短DDP smoke run。

## 2026-07-15 · 移除Strict4迁移命名并统一Motion Dataset接口

类型：命名与公共接口迁移 / Dataset / VAE合同 / 工具 / 配置 / 测试 / 文档

实际改动内容：

- 将`datasets/strict4.py`迁移为`datasets/motion.py`，公共类和collate函数改为`MotionDataset`与`collate_motion`；`load_artifact_records()`继续作为不同来源数据共享的processed-motion解析层。
- 将原生旋转预处理入口迁移为`tools/preprocess_smpl_motion.py`，同步移除Dataset、统计工具、latent工具、错误消息和测试中的Strict4迁移名称。
- 正式VAE配置改为加载`datasets.motion.MotionDataset`，motion statistics环境变量从`STRICT4_MOTION_STATS`改为`MOTION_STATS`。
- 将底层`Strict4CausalVAE`改名为`CausalBodyVAE`，artifact contract version从`strict4-body265-v1`升级为`body265-v1`；旧迁移期artifact会因version mismatch被明确拒绝，不会静默混用。
- 将训练/Web迁移守卫统一改为`BLOCKED_ON_BODY_VAE`，并更新README、架构文档、Web状态与测试名称。
- 保留`FRAMES_PER_TOKEN = 4`、四帧整除校验、token对齐crop/padding和对应测试；本次只移除多余命名，不改变四帧协议及数值实现。
- 历史日志中的旧名称保持原样，以保留当时的真实开发状态；本条目取代其中关于当前环境变量、文件路径和fail-fast状态的后续指引。

改动理由：

- 四帧一token已经是Floodcontrol唯一有效的模型合同，不应继续作为可选模式渗透到类名、文件名、环境变量和错误码。
- `MotionDataset`表达的是统一root5/body265 artifact消费层；HumanML3D、BABEL等模块负责各自来源适配，避免在每个来源Dataset中重复crop、augmentation、mask和collate协议。
- 版本名使用`body265-v1`描述artifact本身，比使用迁移阶段名称更稳定，也能通过显式version bump防止新旧artifact误用。

验证：

- `python -m py_compile`覆盖Motion Dataset、BodyVAE、预处理、统计、latent工具及迁移守卫，通过。
- VAE contract、数据管线、模型、loss、训练配置和迁移守卫定向测试：35 passed。
- 全部测试：50 passed。
- 除本开发日志历史记录外，全仓搜索不存在活跃`strict4/STRICT4`名称、旧Dataset导入、旧预处理入口、旧底层VAE类、旧环境变量或旧fail-fast状态。

涉及文件：

- `datasets/motion.py`、`datasets/__init__.py`、`datasets/humanml3d.py`、`datasets/babel.py`
- `configs/vae.yaml`、`configs/stream.yaml`
- `models/vae_wan_1d.py`、`models/tools/wan_vae_1d.py`、`utils/conditions/vae.py`
- `tools/preprocess_smpl_motion.py`、`tools/compute_vae_stats.py`、`tools/pretokenize_body_latents.py`
- `train_ldf.py`、`web_demo/`、`metrics/stream.py`
- `tests/test_token_frame_contract.py`及VAE/迁移相关测试
- `README.md`、`docs/rearchitecture/`、`docs/DEVELOPMENT_LOG.md`

尚未完成的后续事项：

- 使用HumanML3D/BABEL split TXT和对应原生SMPL/AMASS rotations生成`body265-v1` motion artifacts。
- 仅从train split计算真实`motion_stats.npz`，设置`MOTION_STATS`后执行三卡短DDP smoke run。

## 2026-07-15 · HumanML3D 263D正式接入BodyVAE数据协议

类型：数据源决策修正 / motion codec / 预处理 / yaw增强与统计 / 配置 / 测试 / 文档

实际改动内容：

- 撤销此前“真实训练必须等待原生SMPL/AMASS rotations、不得使用HumanML263”的阻塞决策；第一版正式数据源改为现有HumanML3D `new_joint_vecs/*.npy`，artifact显式记录`source_representation=humanml3d-263-ik-v1`，不把IK-derived rotations描述为原生pose。
- 在`utils/motion_representation.py`实现`humanml263_to_root_body_motion()`：从root增量恢复物理root xyz与完整yaw，从RIC字段恢复global joint positions，将21个IK-derived local rotation6d按HumanML22层级组合为22个global rotations，并重新计算global backward joint velocity（m/s）。旧263D的heading-local forward velocity不直接复用，contacts原样保留。
- 显式区分HumanML root quaternion的三个语义：RIC/world position使用inverse quaternion，root5 heading使用`root_quat_to_physical_yaw()`的完整角与符号约定，IK global rotation沿用HumanML官方rotation-FK的root quaternion；避免把半角、physical yaw和rotation监督混为一体。
- 将原生SMPL预处理入口替换为`tools/preprocess_humanml3d.py`，读取现有split TXT与`new_joint_vecs/<id>.npy`，在独立`HumanML3D_motion`目录写入`body265-v1` artifacts和对应train/val/test TXT。工具拒绝输出到legacy HumanML3D根目录，防止覆盖baseline split。
- 正式VAE配置切换到`${dirs.raw_data}/HumanML3D_motion/{train,val,test}.txt`；Dataset和training builder的fail-fast语义改为`MOTION_ARTIFACT_DATA_REQUIRED`，不再声称需要native rotations。
- 保留训练期每次采样`Uniform(0,2π)`全局yaw：同一角度同步旋转root xz/heading、root-relative positions、global rotations、global velocities和previous-root boundary；contacts与validity保持不变。crop首帧原始yaw无论是否为零，加上独立均匀角后都服从均匀分布；current-heading-local root velocity保持不变量。
- motion statistics对每条train motion使用`0/90/180/270`度四点quadrature。所有受yaw影响的标量通道都是cos/sin的线性组合，因此四点法精确匹配连续均匀yaw的一阶与逐维二阶矩，同时避免随机统计结果；metadata记录`yaw_statistics=uniform-four-point-quadrature-v1`与实际source representation。
- 更新数据/VAE/模型/Web文档、README、迁移守卫文本、正式配置合同和数据集成测试，明确HumanML263只承担离线物理表示来源，不重新成为VAE/LDF运行时接口。

改动理由：

- 原生AMASS rotations是更高质量的数据来源，但不是BodyVAE结构成立的数学前提；为了建立可训练第一版而阻塞于当前不存在的AMASS源数据并不合理。
- 现有HumanML263已经包含root增量、RIC positions、21个IK rotation6d、velocities和contacts，足以显式转换为同构root5/body265；明确记录rotation来源即可控制数据质量风险。
- 将263D先离线转换成版本化artifact，能够保持新版模型接口纯净，并避免每个DataLoader worker重复执行root积分、rotation composition和velocity重算。
- 随机yaw若只修改样本而仍使用canonical-yaw统计，会造成训练分布和归一化统计不一致；确定性四点quadrature消除了该错位。

验证：

- `python -m py_compile`覆盖motion codec、MotionDataset、HumanML预处理、statistics和VAE data builder，通过。
- 合成HumanML263测试覆盖root/body shape、root积分、global backward velocity、contacts、physical yaw与IK root rotation约定、artifact加载、随机yaw全feature同步和local-root不变量。
- 真实`000000.npy`验证：恢复joint positions与现有HumanML RIC恢复最大误差为0；global rotation矩阵正交误差约`4.8e-7`；全部root/body值有限。
- HumanML官方rotation-FK与RIC恢复在真实样本上的joint position最大差约`8.0e-7`，确认rotation composition约定一致。
- 真实`000021`模块入口smoke成功生成motion artifact和全部四组statistics；metadata正确记录`body265-v1`、`humanml3d-263-ik-v1`和yaw quadrature。
- 数据完整性扫描：HumanML train/val/test分别包含23384/1460/4384个sample id，`new_joint_vecs`缺失数均为0。
- 全部测试：53 passed；仅有测试环境无法初始化NVML的PyTorch warning。
- `git diff --check`通过。

涉及文件：

- `utils/motion_representation.py`
- `tools/preprocess_humanml3d.py`、`tools/compute_vae_stats.py`
- `datasets/motion.py`、`utils/training/vae/data.py`
- `configs/vae.yaml`
- `tests/test_vae_data_pipeline.py`、`tests/test_vae_training.py`
- `README.md`、`docs/rearchitecture/`、`train_ldf.py`、`web_demo/`
- `docs/DEVELOPMENT_LOG.md`

尚未完成的后续事项：

- 在`/data1/yuankai/text2Motion/FloodDiffusion/raw_data/HumanML3D_motion`完成train/val/test全量artifact构建；本轮只在`/tmp`完成真实单样本smoke，未写入共享数据目录。
- 用全量train artifacts计算正式`motion_stats.npz`并设置`MOTION_STATS`。
- 对转换后的rotation-FK与position target做数据集级误差分布抽检，再执行GPU 2/3/4短DDP overfit/吞吐smoke；根据验证曲线决定是否后续升级为原生AMASS rotation监督。

## 2026-07-15 · VAE训练运行时装配迁回train_vae入口

类型：训练入口重构 / 文件删除 / 公共导出清理 / 测试

实际改动内容：

- 物理删除`utils/training/vae/runner.py`，不再用额外runner模块隐藏VAE训练的顶层执行流程。
- 将配置加载与statistics fail-fast、seed、DataLoader装配、run目录与代码快照、WandB logger、checkpoint callback、Lightning Trainer以及fit/validate分支全部迁入`train_vae.py`。
- `utils/training/vae/`继续只拥有模型专用组件：`data.py`、`lightning_module.py`和`losses.py`；移除`run_vae_training`在`utils.training.vae`与`utils.training`中的跨层导出。
- 训练入口公共函数统一为`train_vae.main()`，脚本执行与测试调用同一实现。
- 更新训练结构测试，要求`train_vae.py`可直接看到Trainer/DataLoader/LightningModule装配，同时继续禁止在入口文件内重新实现`VAELightningModule`。

改动理由：

- 当前runner只有一个调用方，把完整训练装配藏在额外文件中使入口过薄，并增加无实际复用价值的跳转层。
- 模型、loss和数据构建仍保持模块化，而实验级生命周期装配放在顶层入口，更便于审阅训练究竟如何启动、恢复、记录和验证。

验证：

- `python -m py_compile`覆盖新`train_vae.py`、training package exports和训练测试，通过。
- VAE训练入口与迁移守卫定向测试：17 passed。
- 全部测试：53 passed；仅有测试环境无法初始化NVML的PyTorch warning。
- 全仓活跃代码搜索不存在`run_vae_training`或`training.vae.runner`引用。
- `git diff --check`通过。

涉及文件：

- `train_vae.py`
- `utils/training/vae/runner.py`（删除）
- `utils/training/vae/__init__.py`
- `utils/training/__init__.py`
- `tests/test_vae_training.py`
- `docs/DEVELOPMENT_LOG.md`

尚未完成的后续事项：

- 全量构建`HumanML3D_motion` artifacts与statistics后，从新入口执行单卡overfit smoke和GPU 2/3/4三卡DDP smoke。

## 2026-07-15 · HumanML3DDataset直接接管processed motion artifacts

类型：Dataset职责收敛 / legacy删除 / 公共命名迁移 / 配置与工具更新 / 测试

实际改动内容：

- 删除旧`datasets/humanml3d.py`中的263D、旧VAE token、trajectory/ControlNet条件和文本拼装实现；新版训练不再保留这一未使用的legacy Dataset。
- 将`datasets/motion.py`的root5/body265 artifact加载、四帧对齐crop、translation rebase、previous-root boundary、随机全局yaw与collate逻辑迁入`datasets/humanml3d.py`，随后物理删除`datasets/motion.py`。
- 公共接口统一为`HumanML3DDataset`、`collate_humanml3d()`和`load_humanml3d_records()`；正式VAE配置切换为`datasets.humanml3d.HumanML3DDataset`。
- statistics与deterministic-latent工具同步改从`datasets.humanml3d`读取HumanML artifact records。
- 测试全部迁移到新接口；Dataset fail-fast测试显式使用临时缺失路径，不再依赖本机是否已经生成真实`HumanML3D_motion`。
- 设计文档将通用motion artifact Dataset表述收敛为已实现的`HumanML3DDataset`。

改动理由：

- 当前VAE正式数据源已经冻结为`HumanML3D_motion`，泛化的`MotionDataset`只增加一层不必要的抽象，而旧`HumanML3DDataset`又已完全脱离新版训练路径。
- 让source-specific Dataset直接消费其versioned processed artifacts，能够使文件名、类名、磁盘数据产品和配置target保持一致；未来BABEL可实现自己的Dataset并输出相同batch contract。
- 磁盘目录仍使用`HumanML3D_motion`与FloodDiffusion legacy数据隔离；Python侧`HumanML3DDataset`默认即代表新版root5/body265协议，不需要额外Motion后缀。

验证：

- `python -m py_compile`覆盖HumanML3DDataset、exports、statistics/latent工具及相关测试，通过。
- HumanML数据管线、训练配置和迁移守卫定向测试：23 passed。
- 全部测试：53 passed；仅有测试环境无法初始化NVML的PyTorch warning。
- `datasets/motion.py`物理不存在；除历史开发日志外，全仓不存在活跃`datasets.motion`、`MotionDataset`、`collate_motion`或`load_artifact_records`引用。
- `git diff --check`通过。

涉及文件：

- `datasets/humanml3d.py`
- `datasets/motion.py`（删除）
- `datasets/__init__.py`
- `configs/vae.yaml`
- `tools/compute_vae_stats.py`、`tools/pretokenize_body_latents.py`
- `tests/test_vae_data_pipeline.py`、`tests/test_vae_training.py`
- `docs/rearchitecture/02_DATA_PIPELINE.md`
- `docs/DEVELOPMENT_LOG.md`

尚未完成的后续事项：

- 使用全量train artifacts生成正式motion statistics并执行VAE训练smoke。
- 接入BABEL时新增独立`BABELDataset`，复用VAE batch contract而不恢复旧HumanML trajectory/token分支。

## 2026-07-15 · HumanML3D一键数据构建与正式统计完成

类型：数据构建器 / 断点续跑 / 外部数据产品 / 正式配置 / 测试 / 文档

实际改动内容：

- 将`tools/preprocess_humanml3d.py`增强为一键构建入口：从一个legacy HumanML3D根目录自动读取`new_joint_vecs`和train/val/test TXT，在独立`HumanML3D_motion`目录生成全部artifact及目标split。
- 增加多进程`--workers`、contract/source hash校验与断点续跑；已有artifact仅在`body265-v1`、`humanml3d-263-ik-v1`和source SHA256全部一致时跳过，源数据变化时自动重建。
- artifact和目标split均采用临时文件后原子replace；中断不会留下被目标split引用的半写文件。构建结束写入`build_summary.json`，记录split、missing、too-short、converted/skipped和帧数。
- 在全量构建遇到真实短样本后，将长度/shape检查提升到任务执行前的全量预检；默认`min_frames=20`与VAE训练合同一致，过短样本从目标split显式排除并按split计数，而不是让DataLoader阶段逐样本失败。
- 正式配置的`motion_stats_path`改为`${dirs.raw_data}/HumanML3D_motion/motion_stats.npz`，与数据split一样使用稳定配置路径，不再要求训练前设置环境变量。
- 在本机共享数据目录完成全量构建：目标split为train 23240、val 1450、test 4358，共29048个唯一artifact，目录约3.8 GB；源文件缺失为0，少于20帧而排除的样本为train 144、val 10、test 26。
- 使用train目标split生成正式`motion_stats.npz`；statistics metadata记录`body265-v1`、`humanml3d-263-ik-v1`、HumanML22、train split hash和`uniform-four-point-quadrature-v1`。
- 删除首次中断构建遗留的8个过短且未被任何目标split引用的artifact；最终artifact集合与三个split引用并集均为29048，无缺失、无临时文件、无孤儿文件。
- 更新README和rearchitecture状态：本地HumanML motion artifacts/statistics已经就绪，剩余阻塞转为GPU训练、正式VAE checkpoint、latent artifacts和Web/runtime接线。

改动理由：

- `${dirs.raw_data}/HumanML3D_motion/train.txt`是预处理成功后的数据产品，不应要求用户预先手工创建或复制源split；只有构建器确认artifact成功后才可以原子发布目标split。
- 全量数据构建属于长任务，必须可验证、可恢复且不会因中断污染训练集合；source hash和atomic publish保证重复运行安全。
- 原始HumanML split确实包含少于VAE最短训练长度的样本，预检过滤比在随机训练过程中抛错更稳定，也使排除数量可审计。
- statistics已是正式数据产品且位置固定，直接在配置中引用比环境变量更可复现。

验证：

- 构建器合成测试覆盖三split一次构建、重复运行跳过、source变更重建、短样本过滤、atomic artifact和statistics metadata。
- 全量构建summary：29048 unique artifacts，28132本轮转换、916断点复用；转换帧数3929200（不含复用artifact）。
- statistics六组数组shape正确、全部finite、全部std严格为正；global heading/xz在yaw quadrature后的mean接近0。
- `HumanML3DDataset`真实装配：train/val为23240/1450；`num_workers=0` smoke成功读取batch `[32,196,265]`、root `[32,196,5]`并实例化20,212,004参数的`BodyVAE`。
- 配置中的8-worker DataLoader在Codex受限沙箱因进程间socket权限失败；这是沙箱限制，未修改正式`num_workers: 8`，zero-worker同数据路径验证通过。
- 全部测试：53 passed；仅有测试环境无法初始化NVML的PyTorch warning。
- artifact文件数与split唯一引用数均为29048；`git diff --check`通过。

涉及文件与数据产品：

- `tools/preprocess_humanml3d.py`
- `configs/vae.yaml`
- `tests/test_vae_data_pipeline.py`、`tests/test_vae_training.py`
- `README.md`、`docs/rearchitecture/`、`docs/DEVELOPMENT_LOG.md`
- `/data1/yuankai/text2Motion/FloodDiffusion/raw_data/HumanML3D_motion/`

尚未完成的后续事项：

- 在GPU 2/3/4运行短DDP VAE smoke，验证8-worker真实吞吐、显存峰值、loss反传和checkpoint写入。
- 通过短样本overfit与validation reconstruction检查rotation/position/velocity/contact四个block的数值尺度，再决定是否启动300k正式训练。

## 2026-07-15 · HumanML3D artifact合同补全与静默截断清理

类型：Dataset fail-fast / 数据合同校验 / yaw工具收敛 / 测试

实际改动内容：

- `HumanML3DDataset`现在只接受`train/val/test`，并要求`min_frames`和`max_frames`为不小于4的四帧整数倍，不再静默向下取整配置。
- artifact加载时完整检查contract version、HumanML source representation、期望FPS、root/body/mask shape、共享帧长、四帧整除、finite值及可选`previous_root_frame [5]`。
- 删除`min(root_len, body_len)//4*4`容错；root/body长度不一致或artifact非四帧对齐时立即失败，损坏数据不能进入crop或collate。
- 将`humanml3d-263-ik-v1`收敛为motion representation层共享常量，预处理器和Dataset使用同一来源。
- 增加`rotate_root_yaw()`；随机yaw处理boundary root时不再伪造body输入或执行被丢弃的第二次body旋转。
- VAE Dataset装配从模型配置传入期望FPS，确保artifact物理时间单位与VAE local-root计算一致。
- 新增反例测试，覆盖split拼写、非四帧配置、长度不一致、7帧artifact、错误mask、30 FPS、错误source、NaN和错误boundary shape。

改动理由：

- 四帧协议、FPS和source representation属于训练数据的硬合同；静默截断会掩盖预处理损坏，并可能把不同物理时间单位的数据混入同一统计和训练过程。
- `previous_root_frame`只需要root yaw变换，独立工具能直接表达该语义并避免无意义的body二次变换。

验证：

- Dataset、VAE训练和VAE contract定向测试：24 passed。
- 全部测试：61 passed。
- 正式train split真实batch smoke：23240个样本，batch body `[32,172,265]`、root `[32,172,5]`，有效长度20–172帧。
- `python -m py_compile`覆盖Dataset、motion representation、训练数据装配、预处理器与新增测试，通过。
- `git diff --check`通过。

涉及文件：

- `datasets/humanml3d.py`
- `utils/motion_representation.py`
- `utils/training/vae/data.py`
- `tools/preprocess_humanml3d.py`
- `tests/test_vae_data_pipeline.py`
- `docs/DEVELOPMENT_LOG.md`

尚未完成的后续事项：

- 在GPU 2/3/4执行短DDP训练smoke，验证多worker读取、完整loss反传、显存和checkpoint流程。

## 2026-07-15 · BABEL_motion与同构MultiDataset落地

类型：数据集重构 / artifact构建器 / multi-dataset训练配置 / 外部数据产品 / 测试 / 文档

实际改动内容：

- 将`datasets/babel.py`从旧263D feature/token/trajectory/text万能Dataset重写为`BABELDataset`，直接消费独立`BABEL_motion` root5/body265 artifact，并与`HumanML3DDataset`输出同一个VAE batch contract。
- `HumanML3DDataset`增加source-specific record loader hook；HumanML与BABEL保留独立类名、磁盘根目录和batch source identity，但共享四帧crop、previous-root、随机yaw与artifact验证语义。
- 将HumanML-style 263D文件转换、source hash、原子NPZ写入抽为`tools/motion_artifact.py`；HumanML和BABEL构建器使用同一转换实现，避免两套root/body codec漂移。
- 新增`tools/preprocess_babel.py`：读取`BABEL_streamed/motions`与显式split映射，支持多进程、短片段过滤、missing预检、source-hash续跑和atomic split发布。默认只发布train/val，不把空`test_processed.txt`或调试`test_min_processed.txt`声明成正式test。
- 将`datasets/multi.py`重写为同构Dataset concat：删除旧feature/token/traj key union与缺失键补零collate；新版`collate_multi()`只调用统一body265 collate，并保留每条样本的`dataset`来源。
- VAE data factory支持`data.datasets`列表，并把同一split、frame范围、random-yaw与model FPS下发到各source-specific Dataset。
- 新增`configs/vae_multi.yaml`，组合`HumanML3D_motion`和`BABEL_motion`，使用独立的HumanML3D+BABEL联合statistics；`configs/vae.yaml`继续保留HumanML-only基线。
- 在共享数据目录完成BABEL全量构建和联合statistics生成，并同步更新数据/VAE设计文档与README。

改动理由：

- BABEL VAE训练只需要动作artifact，不应继续携带已删除架构的ControlNet trajectory、旧VAE token和分段文本拼装职责；LDF阶段的多文本absolute timeline应在后续condition/data协议单独实现。
- multi-dataset只有在所有子Dataset共享严格batch contract时才安全；旧万能collate会把配置或数据错误隐藏为补零/list，容易形成训练期静默mismatch。
- HumanML与BABEL都已经是HumanML-style 263D、20 FPS表示，复用一个物理转换器可以保持root5/body265完全同构；dataset origin由目录、split与build summary记录，representation metadata继续诚实标记`humanml3d-263-ik-v1`。

验证：

- 合成测试覆盖BABEL一次构建、短片段过滤、断点续跑、无伪test发布、`BABELDataset`读取及HumanML+BABEL mixed collate。
- 全部测试：63 passed；仅有受限环境无法初始化NVML的PyTorch warning。
- `python -m py_compile`覆盖HumanML/BABEL/Multi Dataset、data factory、两个预处理器、公共artifact转换器和新增测试，通过。
- `git diff --check`通过；活跃Dataset/config/training路径中不存在旧`BabelDataset`、`datasets.multi.collate_fn`或traj-key multi collate实现。
- 全量`BABEL_motion`：train 10442、val 3645，共14087个artifact，source motion缺失0；过滤少于20帧的train 178、val 81；转换1304200帧。
- 真实MultiDataset：train长度`(23240, 10442)`，总计33682；32样本batch成功输出body `[32,188,265]`与root `[32,188,5]`。
- 联合`HumanML3D_BABEL_motion_stats.npz`六组数组shape正确、全部finite、std严格为正，metadata记录两个train split及联合hash。
- 真实HumanML+BABEL双样本经过联合statistics的`BodyVAE.forward()`：frames `[2,192,265]`、posterior mu `[2,48,128]`、continuous reconstruction `[2,192,261]`、contact logits `[2,192,4]`。

涉及文件与数据产品：

- `datasets/humanml3d.py`、`datasets/babel.py`、`datasets/multi.py`、`datasets/__init__.py`
- `utils/training/vae/data.py`
- `tools/motion_artifact.py`、`tools/preprocess_humanml3d.py`、`tools/preprocess_babel.py`
- `configs/vae_multi.yaml`
- `tests/test_multi_dataset.py`、`tests/test_vae_training.py`
- `README.md`、`docs/rearchitecture/02_DATA_PIPELINE.md`、`docs/rearchitecture/02_VAE_AND_BODY_REPRESENTATION.md`
- `/data1/yuankai/text2Motion/FloodDiffusion/raw_data/BABEL_motion/`
- `/data1/yuankai/text2Motion/FloodDiffusion/raw_data/HumanML3D_BABEL_motion_stats.npz`
- `docs/DEVELOPMENT_LOG.md`

尚未完成的后续事项：

- 在GPU 2/3/4执行`configs/vae_multi.yaml`短DDP训练smoke，检查8-worker吞吐、各source采样比例、loss尺度、显存和checkpoint。
- LDF数据阶段重新设计BABEL分段文本与absolute timeline/future observation，不把旧BabelDataset文本逻辑恢复进body VAE Dataset。

## 2026-07-15 · 恢复配置内WandB认证并修复VAE logger接线

类型：训练配置修复 / logger接线 / 回归测试

实际改动内容：

- 在`configs/paths_default.yaml`恢复旧版`wandb_info.key/project/entity`配置，训练无需额外设置`WANDB_API_KEY`环境变量。
- `train_vae._create_logger()`改为直接读取`cfg.wandb_info`；key存在时写入当前进程的`WANDB_API_KEY`并创建`WandbLogger`，debug模式或缺少配置/key时才回退到Lightning本地logger。
- `configs/vae.yaml`不再维护另一套`logger.wandb`结构，避免`wandb_info`与`logger`字段不匹配导致logger被静默禁用。
- 正式配置测试增加`wandb_info`断言；新增logger构造测试，通过替换`WandbLogger`验证project、entity、run name和key接线，不访问外部网络。

改动理由：

- 此前`paths_default.yaml`提供`wandb_info`，而训练入口检查`logger.wandb`，导致`_create_logger()`直接返回`None`，训练只产生本地TensorBoard events而不上传WandB。
- 按当前项目约定恢复旧版配置内认证方式，使已有配置无需额外shell环境即可启动在线记录。

验证：

- `python -m py_compile train_vae.py tests/test_vae_training.py`通过。
- VAE训练配置与迁移守卫定向测试：18 passed。
- 全部测试：60 passed；仅有测试环境无法初始化NVML的PyTorch warning。
- `git diff --check`通过。

涉及文件：

- `configs/paths_default.yaml`
- `configs/vae.yaml`
- `train_vae.py`
- `tests/test_vae_training.py`
- `docs/DEVELOPMENT_LOG.md`

尚未完成的后续事项：

- 重新启动GPU 2/3/4训练，确认控制台出现`WandB logging enabled`及WandB run URL，并完成短DDP smoke。
## 2026-07-15 · HumanML3D/BABEL VAE流式重建评估任务

类型：评估任务实现 / 流式decoder验证 / 可视化与artifact输出 / 测试

实际改动内容：

- 新建`eval/vae/`任务包、独立配置与README；默认从HumanML3D和BABEL各自`val.txt`按原始顺序选取前10个完整样本，不经过训练crop或随机yaw。
- 评估模型从训练checkpoint严格加载EMA encoder/decoder；以deterministic posterior `mu`模拟LDF在反归一化边界后的body token，并为每个样本初始化独立`VAEDecoderState`逐token提交四帧重建。
- explicit root不交给VAE重建；original和reconstruction共享source root，从而单独衡量body tokenizer，而不混入LDF Root Stage误差。
- 每个样本额外执行同权重offline decode并检查offline/stream parity；长序列浮点累积容差冻结为`1e-4`，实际真实样本最大差异约`2.5e-5`。
- 输出结构实现为`output/<dataset>/video/{original,reconstruction}/<sample_id>.mp4`，并保留original/reconstruction NPZ、posterior mu、local-root及validity、contact logits/probability、global joints、逐样本metrics、manifest和dataset/global summary。
- 指标包含position MAE、velocity MAE、rotation geodesic error、contact accuracy/precision/recall/F1、original/reconstruction skating以及stream parity误差。
- 新增合成测试覆盖deterministic-mu流式重建、offline parity、root/body到global joints的表示恢复和用户要求的输出目录合同；`eval/vae/output`只保留`.gitignore`，正式结果不进入源码仓库。

改动理由：

- LDF运行时按token提交latent并维护VAE decoder state，VAE评估必须走相同流式decoder接口，不能只验证batched offline reconstruction。
- VAE不拥有explicit root，original/reconstruction共享root能让视频和数值差异准确对应body tokenizer质量。
- 视频之外保留physical motion、latent/local-root中间量和结构化metrics，便于后续定位视觉问题、复现实验并比较不同checkpoint。

验证：

- `python -m py_compile eval/vae/evaluate_reconstruction.py tests/test_vae_eval.py`通过。
- `pytest -q tests/test_vae_eval.py`：3 passed。
- CPU真实数据无视频smoke：HumanML3D `012698`与BABEL `11364_1`均完成EMA加载、full-clip encode、逐tokendecode、NPZ/metrics/manifest/summary写出；offline/stream最大差异分别为`2.48e-5`与`2.29e-5`。
- CPU真实MP4 smoke：上述两样本的original/reconstruction共4个视频均成功生成且文件非空。
- 全部测试：73 passed，1个测试环境无法初始化NVML的warning；`git diff --check`通过。

涉及文件：

- `eval/__init__.py`
- `eval/vae/__init__.py`
- `eval/vae/config.yaml`
- `eval/vae/evaluate_reconstruction.py`
- `eval/vae/README.md`
- `eval/vae/output/.gitignore`
- `tests/test_vae_eval.py`
- `docs/DEVELOPMENT_LOG.md`

尚未完成的后续事项：

- 按本轮范围未执行HumanML3D和BABEL各10个样本的正式GPU评估，也未向`eval/vae/output`写入正式结果。
- 正式运行后需要人工并排检查original/reconstruction视频，结合逐样本metrics确认足部、快速转身和BABEL跨数据集样本的视觉质量。
- 本任务只评估source explicit root下的VAE body reconstruction；LDF生成root与latent后的端到端质量属于后续LDF评估任务。

## 2026-07-15 · VAE direct-stream与10/10 rolling评估任务拆分

类型：评估协议拆分 / rolling window调度 / trace artifact / 配置与测试

实际改动内容：

- 将VAE重建评估拆成两个独立公共入口：`python -m eval.vae.stream`负责deterministic-mu直接逐token提交，`python -m eval.vae.rolling`负责固定active-window调度；共享EMA加载、physical metrics、视频和artifact实现。
- 配置拆为`eval/vae/stream.yaml`与`eval/vae/rolling.yaml`，结果分别隔离在`eval/vae/output/stream/`和`eval/vae/output/rolling/`，避免两种协议互相覆盖。
- rolling默认冻结为10个history slot、10个future slot和每步commit 1 token；history在commit边界前右对齐、future在边界后左对齐，cold start缺失history和sequence tail缺失future均保留为`position_id=-1`的invalid padding。
- 每次rolling只把future区第一个token提交给persistent VAE decoder，history只读且不重放进cache；窗口随后前移一个token，直至所有token各提交一次。
- rolling reconstruction NPZ新增完整trace：`timeline_position_ids`、history/future mask、window origin、history/future绝对范围、commit token以及10/10/1配置；逐样本metrics和global summary记录rolling协议与window配置。
- README更新为两个任务的独立命令、语义边界和输出路径。

改动理由：

- direct stream只验证VAE causal decoder/cache；history/future active window属于LDF提交调度。拆成两个任务才能分别定位cache错误与window indexing/rolling错误。
- rolling history不能重复送入decoder，否则persistent cache会重复累计已提交动作；固定commit boundary和显式validity使首尾窗口行为可审计。
- 两个任务使用相同deterministic mu时应产生相同body重建；差异只能来自调度bug，因此可作为rolling正确性的强验证。

验证：

- VAE eval定向测试：5 passed；覆盖10/10窗口在cold start、中间与tail的slot/position/mask语义，44帧样本11个token连续且唯一commit，并验证rolling与direct-stream输出逐值一致。
- HumanML3D/BABEL真实双入口无视频smoke通过；两任务的重建指标一致，offline/stream最大差异保持约`2.5e-5`。
- HumanML3D真实`012698`共48 token，rolling artifact trace shape为`[48,20]`，commit范围严格为`0..47`，配置记录为history/future/commit=`10/10/1`。
- 全部测试：75 passed，1个测试环境无法初始化NVML的warning；`git diff --check`通过。

涉及文件：

- `eval/vae/evaluate_reconstruction.py`
- `eval/vae/stream.py`
- `eval/vae/rolling.py`
- `eval/vae/stream.yaml`
- `eval/vae/rolling.yaml`
- `eval/vae/config.yaml`（删除）
- `eval/vae/README.md`
- `tests/test_vae_eval.py`
- `docs/DEVELOPMENT_LOG.md`

尚未完成的后续事项：

- 按本轮范围仍未执行两个任务各自HumanML3D/BABEL前10样本的正式GPU视频评估。
- 当前rolling使用ground-truth deterministic mu模拟LDF已生成future token；真实LDF denoise、三角调度、root/latent联合误差和condition update应由后续LDF端到端评估覆盖。

## 2026-07-15 · humanml265数据身份、VAE训练合同与正式EMA tokenizer修复

类型：数据合同 / statistics所有权 / VAE loss与validation / EMA checkpoint / latent artifact / GPU评估 / 外部数据产品

实际改动内容：

- 将整套HumanML263→root5/body265转换统一为唯一`MOTION_CONVERTER_VERSION="humanml265"`；恢复root/world joints、层级组合22个global rotations、20 FPS backward velocity、cold-start validity和contact复制的完整数学步骤写入转换函数注释，不再设计position/rotation/velocity子版本。
- artifact、Dataset、HumanML/BABEL构建summary和statistics统一写入并校验converter version、FPS、source representation与source SHA；artifact freshness不再只比较source hash。
- statistics从六组motion统计收敛为VAE-owned `local_root[4] + body_cont[261]`，global root统计留给LDF数据阶段独立生成。statistics记录实际train artifact manifest hash、骨架、FPS和converter；训练入口在优化前重新计算当前Dataset manifest，不匹配时抛出`VAE_STATISTICS_STALE`。
- 构建器增加转换前全量finite预检并在summary记录`invalid_nonfinite`；真实HumanML test中的`007975/M007975`两条非有限样本被显式排除，其旧converter孤儿artifact已删除，train/val集合未变化。
- skating loss改为GT contact乘预测足速，不能通过压低预测contact概率逃逸，并与cold-start velocity validity和contact validity联合mask。padding frame的continuous输出置零、contact logits置为`-20`；causal kernel只允许奇数且至少3，消除kernel-size 1的`[-0:]`cache错误。
- VAE validation在EMA权重下同时记录随机sample与deterministic `mu` reconstruction，并增加posterior sigma、mu RMS和active latent fraction诊断；PyTorch 2.6训练恢复/验证入口对项目自产可信checkpoint显式使用`weights_only=False`。
- 新增EMA checkpoint工具与`tools/export_vae_tokenizer.py`：缺少EMA立即失败，将EMA shadows覆盖encoder/decoder全部106个参数并保留checkpoint statistics buffers，校验配置statistics逐元素一致，导出约78 MB的`body-vae-tokenizer-v1`纯推理bundle及强hash。
- `tools/pretokenize_body_latents.py`改为只接受正式EMA tokenizer，采用两遍有界内存编码、`dataset/sample_id`命名空间和atomic NPZ；sample-id hash产生确定性yaw，root/body在encode前同步旋转，每个latent artifact保存yaw offset，metadata记录EMA、checkpoint、statistics、split与artifact manifest身份。
- GPU重建评估发现Ada上TF32令full-sequence与single-token Conv1d选择不同kernel、物理max-abs约`4.2e-3`；评估入口关闭TF32并启用deterministic cuDNN后恢复到约`3.6e-5`，没有放宽`1e-4` parity合同。
- 更新VAE/data/LDF边界文档和README，记录首个300k EMA tokenizer已就绪、latent artifact仍待生成。

改动理由：

- shape contract不能识别公式、source mutation或FPS变化；一个整套converter version加显式公式注释既满足用户要求的单一版本，也能防止旧artifact/statistics被静默复用。
- VAE decoder只消费local root和body，完整clip global root分布与LDF OriginEpoch/window root分布不同，放在同一statistics artifact会混淆所有权。
- predicted-contact skating存在直接优化contact probability的捷径；无效velocity/contact和padding输出若不mask会污染loss、展示与导出。
- 正式LDF tokenizer使用deterministic mu，必须使用同一EMA的encoder/decoder并直接评估mu路径；raw/EMA混用或全量latent驻留RAM均不可接受。
- body265含global rotations与velocities，缓存latent若不同时固定root yaw会产生结构性不一致；显式确定性yaw和命名空间使后续LDF能够复现同一物理样本。

验证：

- 全部测试：73 passed，1个受限环境NVML warning；`py_compile`覆盖本轮训练、转换、statistics、EMA、pretokenize、模型和评估入口；`git diff --check`通过。
- HumanML artifacts升级：train 23240、val 1450、test 4356，`humanml265`，test非有限样本2；BABEL artifacts升级：train 10442、val 3645，非有限样本0。
- HumanML与HumanML+BABEL VAE statistics均重新生成，只含local-root/body数组，并分别绑定artifact manifest SHA256。
- 从`step_299999.ckpt`成功导出`tokenizer_ema.pt`；global step 300000，EMA tensor hash `be2ccb3679b0a4dbac382c0d293898a111eec35e2d40ecd0695d37f96c6e3fa4`与训练时`ckpt_hash.txt`完全一致。
- 完整HumanML validation：deterministic-mu total `0.0051869`、sample total `0.0051965`、mu reconstruction `0.0042533`、posterior sigma mean `0.00341`、active latent fraction `1.0`。
- 正式GPU逐token评估各10条：HumanML position MAE `1.41 mm`、rotation `1.49°`、velocity MAE `0.0150 m/s`；BABEL position MAE `3.47 mm`、rotation `3.00°`、velocity MAE `0.0199 m/s`；contact F1均为`1.0`，offline/stream max-abs约`3.6e-5`。视频、motion和metrics写入ignored的`eval/vae/output/`。
- 抽查BABEL最困难样本`11995_5`中间帧的original/reconstruction并排画面，未发现坐标翻转、骨架爆炸或root错位；误差主要表现为腿部和手臂姿态的小幅偏移，与该样本较高geodesic指标一致。

涉及文件与数据产品：

- `utils/motion_representation.py`、`utils/training/vae/checkpoint.py`、`utils/training/vae/data.py`、`utils/training/vae/lightning_module.py`、`utils/training/vae/losses.py`
- `models/vae_wan_1d.py`、`models/tools/wan_vae_1d.py`、`datasets/humanml3d.py`、`train_vae.py`
- `tools/motion_artifact.py`、`tools/preprocess_humanml3d.py`、`tools/preprocess_babel.py`、`tools/compute_vae_stats.py`、`tools/export_vae_tokenizer.py`、`tools/pretokenize_body_latents.py`
- `eval/vae/evaluate_reconstruction.py`、`eval/vae/README.md`
- `tests/test_vae_data_pipeline.py`、`tests/test_vae_loss.py`、`tests/test_vae_model.py`、`tests/test_vae_training.py`、`tests/test_vae_tokenizer.py`
- `README.md`、`docs/rearchitecture/`、`docs/DEVELOPMENT_LOG.md`
- `/data1/yuankai/text2Motion/FloodDiffusion/raw_data/{HumanML3D_motion,BABEL_motion,HumanML3D_BABEL_motion_stats.npz}`
- `/data1/yuankai/text2Motion/Floodcontrol/vae/20260715_022912_vae_body265/tokenizer_ema.pt`

尚未完成的后续事项：

- 本轮300k checkpoint训练时仍使用修复前的predicted-contact skating公式；现有重建评估健康，因此保留为首版tokenizer，但若进行第二轮训练才会使用新GT-contact loss。
- 尚未比较110k–300k所有保留checkpoint；当前正式bundle来自最后一步，后续可用相同EMA-mu协议筛选是否存在更优较早checkpoint。
- 尚未生成全量namespaced、yaw-consistent latent artifacts，也未将yaw offset/root读取和latent stats接入LDF Dataset。
- 需要人工查看`eval/vae/output`中original/reconstruction视频，重点检查BABEL困难样本`11995_5`；LDF global root statistics和端到端Web/runtime仍属于后续阶段。

## 2026-07-15 · VAE评估输出统一由根级Git规则排除

类型：仓库卫生 / 评估产物隔离

实际改动内容：

- 在仓库根`.gitignore`加入`eval/vae/output/`，统一排除VAE评估生成的视频、motion NPZ、metrics、manifest、summary及后续新增子目录。
- 删除`eval/vae/output/.gitignore`占位文件；output目录不再需要任何可跟踪文件来维持，评估任务运行时按需创建。

改动理由：

- VAE评估结果体积较大且属于本地/实验产物，不应随源码提交到GitHub；根级目录规则比内部通配占位更直接，也不会因新增输出类型而漏收。

验证：

- `git check-ignore -v eval/vae/output/summary.json`命中仓库根`.gitignore`的`eval/vae/output/`规则。
- `git diff --check`通过；本轮仅修改ignore与日志，未运行模型测试。

涉及文件：

- `.gitignore`
- `eval/vae/output/.gitignore`（删除）
- `docs/DEVELOPMENT_LOG.md`

尚未完成的后续事项：

- 无；后续VAE评估可继续写入`eval/vae/output/`，产物默认不会进入Git状态。

## 2026-07-15 · LDF在线EMA编码协议冻结与VAE代码通关审计

类型：架构协议 / VAE因果接口 / 迁移守卫 / statistics校验 / 回归测试

实际改动内容：

- 将正式LDF训练路径从“读取逐样本预编码latent artifact”改为“加载physical root5/body265后，由冻结EMA encoder在线计算deterministic posterior `mu`”。训练时root、body、encoder context和previous boundary共享同一在线yaw；encoder无梯度且每个batch只执行一次，scheduled/self-forcing更新复用同一target。
- 冻结causal encoder warm-up协议：active crop必须携带完整历史感受野，当前网络精确需要`encoder_layers * 2 * (kernel_size - 1)`个历史token；真实序列起点不足时才使用零边界。`CausalBodyVAE/BodyVAE`新增只读`encoder_context_tokens`接口，避免LDF侧复制架构公式。
- 明确latent statistics仍需在VAE冻结后独立计算，但只保存`latent_mu_mean/std`及EMA/context/yaw身份，不保存正式训练逐样本latent。`tools/pretokenize_body_latents.py`降级为诊断或可选加速工具，不再是LDF训练前置条件。
- 更新VAE、数据管线、顶层模型、文档索引和README中的旧latent-cache描述；`train_ldf.py`的fail-fast信息改为准确列出尚未接线的online encoder、latent statistics、context sampler和hybrid batch。
- VAE statistics校验增加finite检查，NaN/Inf mean/std不能进入模型。新增validation即使上层配置`random_yaw: true`也保持确定性、以及encoder context长度的回归断言。

改动理由：

- 在线encode允许每个epoch在物理空间同步改变root/body朝向，消除cached latent与在线root augmentation不一致；同时避免维护大规模latent artifact及其恢复协议。
- causal encoder的输出依赖历史。如果只编码active crop，同一motion token会因window起点不同得到不同target，因此历史感受野必须成为VAE公开合同而不是LDF内部常数。
- validation必须禁止随机yaw才能比较不同checkpoint；statistics中的非有限值应在训练前明确失败，不能传播到normalize和diffusion target。

验证：

- VAE专项：`45 passed in 9.39s`。
- 全仓：`76 passed in 12.66s`。
- `py_compile`覆盖本轮修改的模型、statistics、Dataset、LDF守卫、可选pretoken工具和测试，通过。
- `git diff --check`通过；全仓文档残留搜索未发现`LATENT_GENERATION_PENDING`、正式LDF依赖yaw-consistent latent artifact或`BodyCodeStore`旧表述。

涉及文件：

- `docs/rearchitecture/02_VAE_AND_BODY_REPRESENTATION.md`
- `docs/rearchitecture/02_DATA_PIPELINE.md`
- `docs/rearchitecture/01_MODEL_ARCHITECTURE_AND_IO.md`
- `docs/rearchitecture/README.md`
- `README.md`
- `models/tools/wan_vae_1d.py`
- `models/vae_wan_1d.py`
- `utils/motion_representation.py`
- `train_ldf.py`
- `tools/pretokenize_body_latents.py`
- `tests/test_vae_model.py`
- `tests/test_vae_data_pipeline.py`
- `docs/DEVELOPMENT_LOG.md`

尚未完成的后续事项：

- 本轮只冻结协议并暴露encoder context长度，尚未实现LDF online encoder bridge、context-aware Dataset/collate、latent-statistics生成器和hybrid training batch；`train_ldf.py`继续fail-fast是预期行为。
- 正式EMA tokenizer bundle在latent statistics生成前导出；LDF接线时必须明确“EMA权重/physical stats”与后生成`latent_mu_mean/std`的加载所有权，避免bundle中的占位latent buffers覆盖真实统计。
- VAE validation已有sample/mu reconstruction、sigma mean、mu RMS和active fraction，但仍可补充sigma min/max、显式sample-mu gap以及单独的KL/token诊断；当前训练`kl`保持与FloodDiffusion一致的valid token×latent channel元素平均定义。
- MultiDataset仍采用自然concat比例，尚未记录显式sampling-policy artifact或按HumanML/BABEL分别汇报validation loss。
- decoder snapshot仍未绑定VAE checkpoint/architecture/session identity，该项留给Web/runtime事务接线阶段。
- 当前首个300k checkpoint是在GT-contact skating修复前训练的；代码通关不等于该旧权重已经获得新loss收益。
