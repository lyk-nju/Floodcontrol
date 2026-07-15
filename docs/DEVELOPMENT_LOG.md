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

## 2026-07-15 · VAE rolling评估修正为有限历史cache回放

类型：评估协议修正 / causal decoder cache验证 / artifact与指标 / 测试

实际改动内容：

- 修正此前“VAE direct-stream与10/10 rolling评估任务拆分”条目中的rolling语义：VAE decoder为causal模型，rolling不再包含future token，也不再把当前token持续提交到完整历史的persistent state。
- rolling每个commit点均创建全新的`VAEDecoderState`，最多回放当前token之前10个deterministic posterior `mu`及对应GT local-root patch，再解码当前token并只提交其4帧输出；cold start的历史slot继续右对齐并以`position_id=-1`显式padding。
- 增加两种彼此独立的对照：完整序列persistent stream作为有限历史误差reference；相同`history + current`截断窗口的offline decode作为fresh cache逐token回放的正确性oracle。
- rolling trace从`history/future`合同改为`history/current`合同，记录`timeline_position_ids`、history/current mask、窗口起止和唯一commit token；配置删除`future_tokens`，默认窗口为10 history + 1 current。
- rolling reconstruction NPZ额外保存persistent-stream continuous body和contact logits；metrics增加rolling相对完整stream的position、velocity、rotation、contact与max-abs差异，以及同窗口cache/offline parity误差。
- README和定向测试同步采用“完整encoder deterministic mu + GT local root，只截断decoder历史”的VAE专用实验定义。

改动理由：

- 原10/10实现虽然移动了可见窗口，但decoder一直保留从token 0开始的完整persistent cache，而且每次提交的仍是完全相同的`mu`，因此输出必然与direct stream相同，无法衡量真实有限窗口重建影响。
- VAE causal decoder不消费未来token；future horizon属于LDF去噪窗口，而不是VAE cache实验合同。
- 将“rolling对完整历史的预期差异”和“同窗口offline/stream应一致”拆开，才能区分有限上下文造成的模型误差与decoder cache实现错误。

验证：

- `python -m py_compile eval/vae/evaluate_reconstruction.py tests/test_vae_eval.py`通过。
- `pytest -q tests/test_vae_eval.py`：5 passed；覆盖history/current slot合同、有限历史结果区别、唯一commit和同窗口cache parity。
- CPU真实EMA checkpoint smoke通过：HumanML3D `012698`与BABEL `11364_1`均完成；HumanML样本48 token的trace shape为`[48,11]`、commit范围为`0..47`、artifact不存在future字段且包含persistent reference。
- HumanML真实样本rolling相对完整stream的max-abs为`5.0783e-3`，同一截断窗口cache/offline max-abs为`2.4796e-5`，符合“有限历史改变输出、cache回放仍正确”的预期。
- 全仓测试共76项：75 passed，1 failed；唯一失败为工作区`configs/vae.yaml`当前`trainer.devices=1`，而既有`test_formal_vae_training_config_matches_frozen_recipe`仍断言3卡，与本次VAE rolling修改无关，本轮未擅自改动训练设备配置。

涉及文件：

- `eval/vae/evaluate_reconstruction.py`
- `eval/vae/rolling.py`
- `eval/vae/rolling.yaml`
- `eval/vae/README.md`
- `tests/test_vae_eval.py`
- `docs/DEVELOPMENT_LOG.md`

尚未完成的后续事项：

- 本轮只执行了两个数据集各1个样本的CPU无视频smoke；修正后的HumanML3D/BABEL各10样本GPU视频评估尚未重跑。
- VAE rolling只验证GT local root和encoder deterministic mu下的decoder有限历史影响；LDF自身的active window、去噪、root/latent联合rolling仍应由独立LDF评估覆盖。

## 2026-07-15 · HumanML root5/body265表示接口收敛与旧motion工具清理

类型：表示接口重构 / 旧代码清理 / 命名优化 / 数据增强边界 / 测试

实际改动内容：

- 将正式HumanML 263D source adapter中的root积分恢复从`utils/motion_process.py`迁入`utils/motion_representation.py`，公共名称改为`recover_humanml263_root()`；使用输入同源的`new_zeros/zeros_like`保留device和dtype，并增加shape、长度与finite校验。
- 新增`extract_initial_root_yaw(root_motion)`，直接从explicit root5第一帧的`[cos(yaw), sin(yaw)]`恢复物理yaw；不新增重复字段，也不把随机偏移写进确定性263D预处理。
- 新增`root_body_to_global_joints()`作为root5 + body265/body261到global 22-joint positions的统一接口，并将VAE reconstruction eval从局部重复实现迁移到该公共函数。
- `motion_process.py`明确降级为旧工具兼容模块，删除无调用的legacy root4/trajectory7提取、263D root replacement、`StreamJointRecovery263`、未使用SMPL85/axis-angle恢复链及通配符quaternion导入；文件从616行缩减到218行。
- 仍有旧Web/metrics/runtime调用的263D trajectory、trajectory7和272D visualization适配器暂时保留，但命名显式化：`extract_root_trajectory_length_263()`、`convert_legacy_motion_to_joints()`及私有`_legacy_*`辅助函数；`xy_only`修正为符合实际XZ语义的`planar_only`。
- 更新`local_frame`关于HumanML canonical quaternion来源的文档引用，并增加migration guard，防止已删除的root replacement、旧通用转换名和stream recovery接口重新出现。

改动理由：

- 新版训练和生成的权威表示是root5/body265；旧`motion_process.py`同时暴露263D、trajectory7、272D和SMPL工具，会让新代码继续依赖隐式积分、维度分派和冲突的rotation约定。
- HumanML raw 263D仍是artifact重建的上游来源，因此其确定性root恢复不能直接删除，但应被明确限制在source conversion职责内。
- 首帧yaw已经存在于root5，训练随机yaw必须继续由`rotate_root_body_yaw()`同步作用于root、body positions/rotations/velocities和previous boundary；独立首帧yaw状态或预处理随机化会造成重复状态与不可复现statistics。
- 旧root recovery通过`torch.zeros(...).to(device)`丢失输入dtype，float64输入会在赋值时报错；迁移后的实现修复了该确定性缺陷。

验证：

- `py_compile`覆盖motion representation/process、local frame、VAE eval、legacy visualization、difficulty工具及相关测试，通过。
- 定向回归：`37 passed`，覆盖263D转换、float64 dtype保持、首帧yaw随机偏移、local-root旋转不变量、root/body global joints、VAE eval及migration guards。
- 全仓测试共78项：`77 passed, 1 failed`；唯一失败仍为工作区`configs/vae.yaml`当前`trainer.devices=1`，而既有冻结配置测试断言3卡，本轮未修改用户当前训练设备配置。
- `git diff --check`通过；全仓搜索只在migration guard字符串中保留被删除API名称，不存在活跃导入或调用。

涉及文件：

- `utils/motion_representation.py`
- `utils/motion_process.py`
- `utils/local_frame.py`
- `utils/visualization/video.py`
- `eval/vae/evaluate_reconstruction.py`
- `tools/summarize_difficulty.py`
- `tests/test_vae_contract.py`
- `tests/test_vae_data_pipeline.py`
- `tests/test_vae_eval.py`
- `tests/test_migration_guards.py`
- `docs/DEVELOPMENT_LOG.md`

尚未完成的后续事项：

- 旧Web debug、`datasets/generate.py`和legacy stream/traj metrics仍从raw 263D恢复trajectory；切换到HumanML3D_motion root5/body265 artifact后，可继续删除`extract_root_trajectory_263*()`与`recover_joint_positions_263()`。
- 旧`stream_runtime/runtime_update`仍使用trajectory7 view；新版LDF runtime完成root5 timeline接线后，应删除`build_physical_7d_from_5d()`及其私有delta adapter。
- legacy 272D visualization仍通过`convert_legacy_motion_to_joints()`保留；新版通用视频入口迁移为显式root5/body265或global joints输入后，可物理删除`motion_process.py`剩余内容。

## 2026-07-15 · Motion processing统一为root5/body265并物理清理旧runtime

类型：表示接口重构 / 263D离线转换隔离 / 旧runtime物理删除 / 指标与可视化迁移 / 测试

实际改动内容：

- 将`utils/motion_process.py`重写为唯一canonical physical motion模块，文件头完整记录root5/body265通道、坐标系、单位与rotation6d列约定；公共API冻结为`pack_body/unpack_body`、`rotation_to_matrix/matrix_to_rotation`、`compute_joint_velocities/build_motion`、三个`recover_*`和两个yaw旋转函数。
- 删除`utils/motion_representation.py`。VAE statistics与artifact manifest校验迁入中立`utils/motion_artifact.py`；converter/source版本常量迁入VAE/data contract，模型和Dataset不依赖离线tools。
- 新增`tools/convert_motion_263_to_265.py`，集中记录HumanML 263D与目标265D定义，并独占`recover_root_263`、world joint positions、层级global rotations、contact策略和唯一263→265转换；HumanML/BABEL artifact构建器统一调用该离线转换。
- 将Dataset、BodyVAE、loss、statistics、latent statistics、pretokenize、reconstruction eval、测试和可视化迁移到新API。`rotation_6d_to_matrix/matrix_to_rotation_6d`正式改名为`rotation_to_matrix/matrix_to_rotation`；首帧heading通过`recover_root_yaw(...)[...,0]`取得。
- `metrics.stream`改为直接接收explicit root5/body265并计算boundary与offline parity；difficulty工具改为读取转换后的artifact root5；通用video入口改为从root5/body265恢复world joints，不再按263/272维度分派。
- 物理删除无活跃调用的旧`metrics/traj.py`、`datasets/generate.py`、旧turn-update工具、`utils/inference/stream_runtime`、`utils/inference/runtime_update`和Web trajectory7诊断。Web debug preset在body-VAE runtime接线前复用既有`BLOCKED_ON_BODY_VAE`显式失败。
- 更新VAE/data架构文档、README和migration guards，明确运行时只允许root5/body265，263D只作为离线source adapter存在。

改动理由：

- 运行时同时存在263D积分恢复、trajectory7、272D分派与root5/body265会造成表示所有权不清、rotation约定冲突以及新代码继续依赖legacy shape分支。
- 263D仍是当前artifact的真实上游来源，因此需要保留可测试的离线转换，但不能被模型、Dataset或在线runtime直接消费。
- 显式root与body表示已经能直接支持joint恢复、stream指标和可视化；继续维护旧root replacement和trajectory7适配器没有架构收益。

验证：

- `python -m compileall -q datasets eval metrics models tools utils web_demo tests`通过。
- VAE/数据/评测/迁移定向回归：`50 passed`。
- 全仓`pytest -q`：`94 passed, 1 warning`；warning仅为测试环境无法初始化NVML，不影响CPU测试结果。
- `git diff --check`通过。
- 全仓Python残留搜索确认不存在活动的`motion_representation`导入、旧motion函数名、trajectory7、旧stream/runtime模块引用；263D只保留在离线转换、预处理以及对应测试中。

涉及文件：

- `utils/motion_process.py`
- `utils/motion_artifact.py`
- `utils/conditions/vae.py`
- `tools/convert_motion_263_to_265.py`
- Dataset、VAE、loss、statistics、eval、metrics、visualization及对应测试调用方
- 旧263/trajectory7 runtime、metrics和工具文件（删除）
- `README.md`、`docs/rearchitecture/02_VAE_AND_BODY_REPRESENTATION.md`、`docs/rearchitecture/02_DATA_PIPELINE.md`

尚未完成的后续事项：

- 真实LDF在线EMA encoder bridge、latent statistics与hybrid batch仍未接线，`train_ldf.py`继续按既定协议fail-fast。
- Web commit-time decoder事务仍未实现；旧trajectory7 runtime已删除，后续Web接线必须直接使用LDF Hybrid stream state、root5 timeline和body265 decode结果。

## 2026-07-15 · VAE tokenizer/statistics、causal context与skating交付修复

类型：checkpoint协议修复 / loss与评估修正 / causal streaming合同 / statistics产品 / 配置与测试

实际改动内容：

- 将正式`tokenizer_ema.pt`限定为EMA encoder/decoder参数、body/local-root physical statistics与模型身份；导出和加载均排除`latent_mean/std`，加载时先验证state hash与architecture hash，再只允许这两个post-hoc buffers缺失。`load_ema_checkpoint()`采用相同规则，不会覆盖调用方已经加载的latent statistics。
- 新增独立`tools/compute_vae_latent_stats.py`：使用冻结EMA encoder、完整因果上下文、确定性sample-hash yaw和batched GPU扫描计算`latent_stats.npz`，全程检查posterior、mean/std和normalized latent的finite与正std；可选pretoken工具同步加入fail-fast，但仍不属于正式LDF数据路径。
- VAE resume在任何state覆盖前比较body/local-root四个physical statistics buffers；新checkpoint额外记录contract、converter、FPS、motion-statistics hash和artifact-manifest hash，旧checkpoint仅在physical buffers完全一致时允许恢复。
- `L_skate`改为从预测global foot positions的相邻帧差分直接计算足速，再由GT contact加权；crop首帧、padding、无效position transition不参与。评估拆分为GT-contact position-derived、predicted-contact position-derived和GT-contact velocity-feature三个指标。
- `BodyVAE`补齐`decoder_context_tokens`和`encode_window()`；当前正式encoder/decoder上下文均为24 tokens。`VAEDecoderState`加入tokenizer hash、architecture hash、session identity与token index，step/snapshot/restore拒绝跨模型、跨会话和非连续调用。
- Physical statistics显式记录`uniform_yaw/no_yaw`，训练入口强制匹配`data.random_yaw`；Dataset提前拒绝同一namespace重复sample ID。
- `vae.yaml`与`vae_multi.yaml`统一为单卡、`strategy: auto`以及train/val/test实际batch size 128；未设置梯度累积或静默batch调整。
- 更新LDF fail-fast信息、VAE设计文档和rolling评估文档；LDF入口仍保持阻断，但现在准确说明剩余项是context-aware crop/collate、frozen online encoder调用与HybridMotion training batch。

改动理由：

- 训练checkpoint中的identity latent buffers不能覆盖VAE冻结后计算的真实latent尺度，否则LDF normalize/unnormalize会静默错误；EMA tokenizer和post-hoc latent statistics必须有独立所有权并相互绑定hash。
- 预测velocity feature与最终展示使用的position输出是独立head，使用前者计算skating不能直接约束视觉脚滑。
- LDF在线encode和有限历史decode只有在精确感受野、session隔离与连续commit边界被公共接口固定后，才能避免crop/reset分布漂移和cache串用。

验证：

- 全仓`pytest -q`：`94 passed, 1 warning`；warning为测试环境无法初始化NVML，不影响结果。
- `py_compile`覆盖VAE模型、contracts、checkpoint、training、loss、data、statistics、eval、metrics和LDF守卫，通过。
- `git diff --check`通过。
- 使用23,240条HumanML train artifacts重算physical statistics，metadata确认`yaw_policy=uniform_yaw`。
- 从`step_299999.ckpt`原子替换`tokenizer_ema.pt`，确认state中不存在`latent_mean/std`且EMA global step为300000。
- 使用GPU 2、batch 128扫描全部train motions生成独立`latent_stats.npz`；联合加载确认tokenizer hash一致、encoder/decoder context均为24、latent mean/std finite且最小std为`1.1106098`。

涉及文件：

- `models/vae_wan_1d.py`、`models/tools/wan_vae_1d.py`、`utils/conditions/vae.py`
- `utils/training/vae/{checkpoint,data,lightning_module,losses}.py`、`utils/training/lightning_module.py`
- `tools/{compute_vae_stats,compute_vae_latent_stats,pretokenize_body_latents}.py`
- `datasets/humanml3d.py`、`eval/vae/`、`metrics/stream.py`、VAE相关测试与配置
- `docs/rearchitecture/02_VAE_AND_BODY_REPRESENTATION.md`、`train_ldf.py`
- 外部数据产品：HumanML motion statistics、300k run目录中的`tokenizer_ema.pt`与`latent_stats.npz`

尚未完成的后续事项：

- 现有300k权重是在position-derived skating启用前训练的；重新导出不改变其训练收益，只有后续重新训练才会学习该loss。
- LDF context-aware crop/collate、frozen online encoder调用和HybridMotion batch尚未实现，真实`train_ldf.py`继续fail-fast。
- Web commit-time decoder事务尚未接线；新增decoder state身份合同已为该阶段提供保护边界。

## 2026-07-15 · Motion utils职责收敛与依赖方向重构

类型：模块职责重构 / root codec统一 / 离线工具重命名 / 旧几何清理 / 测试与文档

实际改动内容：

- 将`FRAMES_PER_TOKEN`、root/local-root/body维度及feature slice统一收敛到`utils/motion_process.py`；VAE/LDF contracts改为导入同一组物理常量，不再各自维护数值副本。
- 将heading单位圆投影和backward/current-heading-local root派生集中到`utils.motion_process.project_root_heading()`与`recover_local_root()`。`utils/conditions/ldf.py`只保留typed tensor contracts与condition编译，不再实现物理root codec；LDF模型改为通过唯一motion API恢复local root。
- 将artifact contract、converter/source identity与statistics schema归入`utils/motion_artifact.py`，消除中立artifact模块反向依赖VAE condition的关系；VAE condition只导入并公开这些数据身份。
- 将离线NPZ构建工具从`tools/motion_artifact.py`改名为`tools/build_motion_artifact.py`，并迁移HumanML/BABEL预处理及测试调用，避免与运行时`utils/motion_artifact.py`同名。
- 从`utils/local_frame.py`删除无调用的5D/7D canonicalization及其私有原地变换函数，只保留通用yaw和XZ坐标几何。
- 增加migration guards，锁定`conditions → motion_process → local_frame`依赖方向、唯一root codec、唯一物理常量所有权、离线builder命名和旧trajectory canonicalization不可回归。
- 更新VAE与数据架构文档，明确runtime physical motion、artifact identity/statistics、offline conversion/build各自所有权。

改动理由：

- 原先`motion_process`反向导入`conditions.ldf`来恢复local root，使底层物理表示依赖上层模型合同；同一物理维度也散落在VAE/LDF两个condition模块，容易发生协议漂移。
- 两个不同职责的`motion_artifact.py`无法从文件名判断是runtime statistics还是offline writer；显式builder命名可以避免错误导入。
- 已删除的trajectory7 runtime不再消费canonicalization函数，继续保留只会暗示旧表示仍是受支持路径。

验证：

- 定向回归：`53 passed`，覆盖LDF condition/root codec、VAE contract/data/tokenizer、MultiDataset与migration guards。
- 全仓`pytest -q`：`97 passed`。
- `python -m compileall -q datasets eval metrics models tools utils web_demo tests`通过。
- `git diff --check`通过。
- 残留搜索确认除反回归断言外，不存在活动的`derive_local_root_motion`、旧`tools.motion_artifact`或5D/7D canonicalization调用；`motion_process.py`与`motion_artifact.py`不再导入`utils.conditions`。

涉及文件：

- `utils/motion_process.py`、`utils/motion_artifact.py`、`utils/local_frame.py`
- `utils/conditions/{ldf,vae,__init__}.py`
- `models/diffusion_forcing_wan.py`
- `tools/build_motion_artifact.py`、`tools/preprocess_humanml3d.py`、`tools/preprocess_babel.py`（旧`tools/motion_artifact.py`删除）
- LDF/VAE/MultiDataset/migration相关测试
- `docs/rearchitecture/02_VAE_AND_BODY_REPRESENTATION.md`、`docs/rearchitecture/02_DATA_PIPELINE.md`

尚未完成的后续事项：

- 真实LDF context-aware crop/collate、冻结EMA encoder在线调用与HybridMotion batch仍待实现，训练入口继续fail-fast。
- Web commit-time decoder事务仍待接线；本次仅收敛底层motion与artifact职责，不扩大runtime功能范围。

## 2026-07-15 · BodyVAE生命周期拆分与简洁latent接口重构

类型：模型职责重构 / statistics与tokenizer加载协议 / decoder session事务 / 配置迁移 / 资产兼容验证

实际改动内容：

- 将`BodyVAE`公共latent语义固定为四组简洁接口：`encode/decode`与`encode_window/decode_step`只处理raw posterior latent；`tokenize/detokenize`与`tokenize_window/detokenize_step`只处理normalized latent。删除根据`normalized_latent`布尔值切换输入空间的双语义方法。
- `BodyVAE.forward()`改为只接受`VAEInput`；batch dict继续由`VAELightningModule`转换。模型构造函数不再读取NPZ、解析metadata或接受statistics路径/零散tensor/identity fallback布尔组合，只接收已验证的`VAEStatistics`与可选`LatentStatistics`对象。
- 在`utils/motion_artifact.py`实现`LatentStatistics`的shape、finite、EMA、tokenizer hash、motion-statistics hash、yaw与encoder-context校验；physical/latent identity statistics仅通过显式test helper构造。latent buffers设为non-persistent，不写入训练checkpoint或tokenizer bundle。
- 新增`utils/training/vae/factory.py`的`build_vae_for_training()`，正式VAE配置改为通过命名factory加载physical statistics并构造不带latent statistics的训练模型；删除配置中的`allow_identity_statistics`、`require_latent_statistics`与`latent_stats_path`。
- 新增中立`utils/vae_tokenizer.py`，集中EMA tokenizer bundle的hash/architecture/physical statistics验证、导出，以及`load_ema_tokenizer()`和`load_frozen_body_tokenizer()`生命周期入口。LDF/runtime loader不再依赖`utils.training`。
- `utils/training/vae/checkpoint.py`收敛为训练checkpoint/EMA提取；从旧300k训练checkpoint恢复时显式剥离历史placeholder latent buffers，同时继续在覆盖模型前校验四个physical statistics buffers。
- 底层`CausalBodyVAE`只创建和更新causal convolution cache tensor，不再理解tokenizer、architecture、session或token index。
- 新增`utils/inference/vae_decoder.py`的`VAEDecoderSession`，统一拥有正式tokenizer identity、session identity、连续token index、snapshot/restore和normalized token逐步解码；调用方不再重复传回state内已有identity/index。
- VAE reconstruction eval迁移到raw `decode_step`做网络/cache parity；通用stream decode迁移到normalized `VAEDecoderSession`。EMA导出、latent statistics和诊断pretokenize工具迁移到新的artifact/tokenizer loader。
- 更新VAE架构文档与migration guards，冻结模型、artifact、训练loader和runtime session的职责边界。

改动理由：

- 原`BodyVAE`同时承担网络计算、statistics文件加载、tokenizer身份绑定与流式事务，构造参数和方法语义会随着artifact/runtime需求持续增长。
- raw posterior latent与normalized LDF latent由同一step方法的布尔参数解释时，调用错误不会体现在类型或方法名上；固定方法语义可以在LDF接线前消除这一风险。
- runtime若依赖training checkpoint模块，会使部署路径携带无关训练生命周期；正式tokenizer必须由中立、可验证的loader一次性构造。
- session identity与snapshot是调用事务，不是causal convolution数学；将其移出模型后，底层cache更新可以独立验证，模型也不再生成不稳定的随机tokenizer identity。

验证：

- VAE/tokenizer/training/eval/migration定向回归通过：`62 passed`。
- 全仓`pytest -q`：`102 passed`。
- `python -m compileall -q datasets eval metrics models tools utils web_demo tests`通过。
- `git diff --check`通过；残留搜索确认不存在活动的旧statistics布尔参数、`bind_tokenizer_identity`、模型内stream/snapshot方法或`normalized_latent`开关，runtime/model不反向导入`utils.training.vae`。
- 使用现有300k `tokenizer_ema.pt`与`latent_stats.npz`通过`load_frozen_body_tokenizer()`实际加载：latent 128D、encoder/decoder context均为24 tokens、模型frozen/eval、global step为300000。
- 使用现有`step_299999.ckpt`和HumanML motion statistics通过训练factory与`load_ema_checkpoint()`实际加载，EMA tokenizer state hash保持`fcb19743016780db...`，确认网络参数命名与已有资产兼容。

涉及文件：

- `models/vae_wan_1d.py`、`models/tools/wan_vae_1d.py`
- `utils/motion_artifact.py`、`utils/vae_tokenizer.py`
- `utils/inference/vae_decoder.py`、`utils/inference/__init__.py`
- `utils/training/vae/{factory,checkpoint,lightning_module,__init__}.py`
- `configs/vae.yaml`、`configs/vae_multi.yaml`、`train_vae.py`
- `tools/{export_vae_tokenizer,compute_vae_latent_stats,pretokenize_body_latents}.py`
- `eval/vae/evaluate_reconstruction.py`、`metrics/stream.py`
- VAE model/tokenizer/training/eval/migration相关测试
- `docs/rearchitecture/02_VAE_AND_BODY_REPRESENTATION.md`

尚未完成的后续事项：

- 真实LDF context-aware dataset/collate尚未调用`load_frozen_body_tokenizer()`与`tokenize_window()`构造HybridMotion batch，`train_ldf.py`继续fail-fast。
- Web commit-time decode尚未持有`VAEDecoderSession`并与LDF commit事务绑定；本次完成session能力与身份保护，但未恢复Web生成。

## 2026-07-15 · Coordinate transform几何层重构

类型：模块重命名 / 坐标几何分层 / 调用迁移 / 测试与文档

实际改动内容：

- 将`utils/local_frame.py`替换为`utils/coordinate_transform.py`，以文件名明确其职责是坐标系变换而不是local-frame状态。
- 新几何层仅公开angle、yaw matrix、heading direction、XZ点world/local变换和XZ向量world/local旋转；点变换包含平移，向量变换只包含旋转，并支持batch-prefix anchor在时间维广播。
- `utils.motion_process.recover_local_root()`改为调用`rotate_vectors_world_to_local()`派生current-heading-local速度；全局yaw增强的旋转矩阵改为调用统一`yaw_to_matrix()`，消除motion层重复几何公式。
- `utils.inference.timeline`切换到`rotate_vectors_local_to_world()`，保留timeline对同一底层坐标协议的复用。
- 将HumanML263 recovered quaternion的半角、符号解释迁入`tools/convert_motion_263_to_265.py`；运行时坐标模块不再理解263D source convention。
- 更新VAE/body representation文档和migration guard，锁定`conditions → motion_process → coordinate_transform`依赖方向，并确认旧`local_frame.py`及trajectory-specific canonicalization不会回归。

改动理由：

- `local_frame`容易被理解为某种persistent local状态，而模块实际承担的是无表示所有权的坐标变换；`coordinate_transform`能准确表达其可被motion codec、runtime timeline和后续route compiler共同依赖的底层角色。
- root5/local-root属于`motion_process`语义，HumanML quaternion属于离线source adapter语义；将两者从纯几何层分离可避免底层工具重新绑定具体动作表示。
- 位置与速度若共用含糊的XZ变换接口，容易误给速度添加anchor translation；拆分point transform和vector rotation后接口直接约束正确几何语义。

验证：

- 定向回归：`49 passed`，覆盖coordinate transform、VAE/root contract、LDF condition、migration guards和VAE data pipeline。
- 全仓`pytest -q`：`102 passed`。
- `py_compile`覆盖coordinate transform、motion process、timeline、HumanML converter和新增测试，通过。
- `git diff --check`通过。
- 残留搜索确认不存在活动的`utils.local_frame`导入；HumanML quaternion解释只存在于离线转换工具。

涉及文件：

- `utils/coordinate_transform.py`（新增）、`utils/local_frame.py`（删除）
- `utils/motion_process.py`、`utils/inference/timeline.py`
- `tools/convert_motion_263_to_265.py`
- `tests/test_coordinate_transform.py`、`tests/test_migration_guards.py`
- `docs/rearchitecture/02_VAE_AND_BODY_REPRESENTATION.md`

尚未完成的后续事项：

- `RootTimeline`本身仍包含旧body-recovery累积入口；本次只迁移其坐标依赖，后续runtime阶段需改为直接消费LDF committed physical root5。
- route compiler尚未接线；未来world route到window-local typed observations应直接复用本次point/vector接口，不再增加另一套坐标公式。

## 2026-07-15 · 删除旧traj batch辅助接口

类型：旧接口物理删除 / 迁移守卫

实际改动内容：

- 删除无活动调用的`utils/traj_batch.py`，一并移除旧`root_to_traj_feats()`的`[x,z,cos,sin]`隐式轨迹视图和未接线的root XZ高斯平滑入口。
- 在migration guard中固定该文件不得恢复，避免新版typed root observations重新依赖无mask、无坐标所有权的旧轨迹特征。

改动理由：

- 新版LDF使用root5 value与逐feature mask表达位置、heading等typed observations；旧4D打包接口会错误地把路径位置和推导heading绑定成同一种约束。
- 当前Dataset、VAE、LDF、runtime和离线数据工具均不调用该文件；用户确认本阶段不保留或迁移其中的route smoothing与heading推导能力。

验证：

- `tests/test_migration_guards.py`：`18 passed`。
- 全仓残留搜索除反回归断言外不存在`traj_batch`、`smooth_root_xz`或`root_to_traj_feats`。
- `git diff --check`通过。

涉及文件：

- `utils/traj_batch.py`（删除）
- `tests/test_migration_guards.py`

尚未完成的后续事项：

- 若未来route compiler需要路径平滑或由路径推导heading，应基于typed root observation与显式valid mask重新设计，不恢复本次删除的接口。

## 2026-07-15 · 统一strict-4 token/frame时间协议

类型：公共时间合同重构 / 调用迁移 / mask语义拆分 / 测试与文档

实际改动内容：

- 将`utils/token_frame.py`重写为唯一的四帧一token协议所有者，只保留`FRAMES_PER_TOKEN=4`，删除可由调用方传入其他factor的旧接口和所有负索引静默clamp行为。
- 新增严格的frame/token count、index、slice和commit boundary换算；非整数、负数和非四帧对齐输入均fail-fast。仅离线预处理可以显式调用`aligned_frame_floor()`截去不完整尾patch，模型、Dataset与runtime使用exact conversion。
- 将frame mask拆成两种不可互换的规约：完整motion validity使用四帧AND，稀疏observation existence使用四帧OR；prefix token mask出现`True/False/True`空洞时明确报错。
- 将motion codec、VAE/LDF contracts与模型、HumanML Dataset、VAE loss、timeline、预处理、latent statistics/tokenization和VAE eval迁移到统一接口，删除调用方自行`/4`、`*4`、`%4`推断时间边界的关键路径。
- 更新VAE设计文档与migration guard，锁定`token_frame.py`拥有时间合同、`motion_process.py`拥有root5/body265物理表示的职责边界。

改动理由：

- 旧接口允许每个调用方传入`frames_per_token`，会让固定strict-4架构在运行时产生多个互不兼容的时间协议。
- 旧frame count采用向上取整、负索引夹到零，会把不完整patch或非法window边界静默解释成有效token；这与VAE每个token必须完整对应四帧的模型合同冲突。
- padding有效性要求四帧全部有效，而稀疏root observation只要求patch内存在任一观测；使用同一个OR helper会让含padding的motion token被错误计入loss或posterior统计。

验证：

- token/frame、migration、LDF condition、VAE contract/model/loss/data定向回归：`70 passed`。
- 全仓`pytest -q`：`106 passed`。
- `py_compile`覆盖本次修改的token contract、motion/condition、Dataset、模型、loss、预处理、统计与eval文件，通过。
- 残留搜索确认旧`token_start_frame`、ceil token count、通用`frames_to_token_mask`等接口已无活动引用，且仅`utils/token_frame.py`直接定义`FRAMES_PER_TOKEN`。

涉及文件：

- `utils/token_frame.py`、`utils/motion_process.py`
- `utils/conditions/{vae,ldf}.py`、`utils/inference/{timeline,geometry}.py`
- `datasets/humanml3d.py`
- `models/{vae_wan_1d,diffusion_forcing_wan}.py`、`models/tools/wan_vae_1d.py`
- `utils/training/vae/losses.py`
- `tools/{preprocess_humanml3d,preprocess_babel,build_motion_artifact,compute_vae_latent_stats,pretokenize_body_latents}.py`
- `eval/vae/evaluate_reconstruction.py`
- `tests/test_token_frame_contract.py`、`tests/test_migration_guards.py`
- `docs/rearchitecture/02_VAE_AND_BODY_REPRESENTATION.md`

尚未完成的后续事项：

- LDF真实训练与Web commit runtime尚未接线；它们后续必须直接复用本时间合同，不得重新引入可配置frames-per-token或首帧特殊token。
- 当前`RootTimeline`仍是旧runtime遗留结构；本次只迁移其frame/token换算，后续route/window compiler重构时再审视其业务接口。

## 2026-07-15 · 删除孤立弧长模块并整理初始化接口

类型：旧模块物理删除 / 公共API命名迁移 / 代码风格与测试

实际改动内容：

- 删除没有任何活动调用和测试的`utils/path_arclength.py`；新版route仍通过`utils/inference/geometry.py`中的实际调用链执行时间采样与polyline弧长重采样。
- 重写`utils/initialize.py`的模块说明、类型标注、常量、配置override解析、动态target校验、run snapshot和分布式timestamp注释，同时保留原配置合并、动态构造与run初始化行为。
- 将含糊公共名统一迁移为行为导向命名：`Config → ProjectConfig`、`instantiate → instantiate_target`、`get_function → resolve_function`、`save_config_and_codes → save_run_snapshot`、`print_model_size → log_model_parameters`、`check_state_dict → log_state_dict_summary`、`get_shared_run_time → get_shared_run_timestamp`。
- `log_state_dict_summary()`现在分别报告unexpected keys、missing parameters和missing buffers，并统一通过rank-zero logger输出；动态target缺少模块限定、目标不存在或目标不可调用时给出明确错误。
- 同步迁移VAE训练入口、通用Lightning模块、VAE data factory、multi Dataset、T2M metric与migration测试；不保留旧名兼容别名。

改动理由：

- `path_arclength.py`属于未接线的旧手绘路径pipeline，并与当前`inference/geometry.py`的route几何能力重叠；保留两套弧长实现会让后续route compiler难以判断协议所有者。
- 原`initialize.py`混用了`get/print/check`等无法表达副作用的名称，调用处看不出是动态导入、rank-zero日志还是完整源码快照。
- 初始化模块是当前配置、Dataset、模型、optimizer、scheduler和训练run的活动依赖，不能像无调用旧模块一样直接删除；本次通过明确命名和边界改善可维护性而不拆散其现有bootstrap职责。

验证：

- initialize、migration、VAE training与multi Dataset定向回归：`31 passed`。
- 全仓`pytest -q`：`101 passed`。
- 修改文件`py_compile`通过，`git diff --check`通过。
- 残留搜索确认不存在活动的旧initialize函数名；`path_arclength`只剩migration guard中的反回归路径断言。

涉及文件：

- `utils/path_arclength.py`（删除）
- `utils/initialize.py`
- `train_vae.py`
- `utils/training/lightning_module.py`、`utils/training/vae/data.py`
- `datasets/multi.py`、`metrics/t2m.py`
- `tests/test_initialize.py`、`tests/test_migration_guards.py`

尚未完成的后续事项：

- `initialize.py`目前仍同时承载配置、factory与run bootstrap；若未来训练入口数量明显增加，可再拆为`config/factory/training experiment`模块，但当前不为纯目录美观增加额外迁移层。
- route compiler若未来确实需要Douglas-Peucker简化、归一化arc-s或degenerate-path mask，应在typed route合同下重新实现并补测试，不恢复已删除的孤立模块。

## 2026-07-15 · Visualization收敛为HumanML22轻量渲染

类型：可视化架构清理 / 旧实现物理删除 / VAE eval迁移 / 测试

实际改动内容：

- 将`utils/visualization/skeleton.py`收敛为不可变`HUMANML22_CHAINS`与统一chain color常量，不再混合骨架定义、Pyrender scene、相机、mesh和视频编码。
- 新增`utils/visualization/motion_video.py`，公开`render_joint_video([F,22,3])`与`render_motion_video(root5, body265)`两个固定语义入口；前者只负责固定正交视角投影和轻量ImageIO/PIL rasterization，后者通过canonical `recover_joint_positions()`恢复world joints。
- 删除`utils/visualization/video.py`及无人调用的目录批渲染、FFprobe/FFmpeg composite、标题文字条功能；同时从旧`skeleton.py`删除已损坏的Pyrender 3D分支、root-only trajectory renderer和随机MP4 demo。
- VAE reconstruction eval改为直接调用`render_joint_video()`，original与reconstruction继续共享explicit root恢复出的world trajectory与同一相机拟合协议。
- 新renderer严格校验frame/joint shape、finite值、FPS、偶数画面尺寸和绘制参数，不再静默补齐或裁剪trajectory/mask；输出目录自动创建，writer在编码异常时也通过`finally`关闭。
- `utils.visualization`改为轻量、无副作用导出；导入时不再加载Pyrender、Trimesh或Matplotlib。

改动理由：

- 当前VAE eval真实使用的只有HumanML22拓扑与轻量renderer，而旧3D路径调用`create_skeleton_trimesh()`时传入其签名不接受的`bone_colors`，实际不可运行。
- 旧模块在设置`PYOPENGL_PLATFORM`前已经导入Pyrender，并让任何简单视频评测都承担OpenGL与Trimesh依赖；这会放大headless训练/评测环境的不稳定性。
- 目录遍历、artifact布局和多视频comparison属于具体eval任务，不应由通用motion renderer拥有；root5/body265恢复与joint rasterization分层后接口所有权更清楚。

验证：

- visualization、VAE eval与migration定向回归在实现完成时：`34 passed`；随后当前工作区Dataset接口发生外部并行变化后，visualization与VAE eval复验：`15 passed`。
- 使用ImageIO/FFmpeg真实写入两帧480×480 MP4烟测通过，文件大小5229 bytes并在验证后删除；同一进程确认导入`utils.visualization`未加载`pyrender`、`trimesh`或`matplotlib`。
- 修改文件`py_compile`通过，`git diff --check`通过；残留搜索确认生产代码不存在旧Pyrender/composite/root-only renderer调用。
- 全仓`pytest -q`未完成：收集阶段因本轮范围外、在验证期间同时发生的Dataset接口变化而出现两个ImportError——`datasets.humanml3d.collate_humanml3d`与`datasets.multi.collate_multi`当前不存在。本轮未修改或回滚这些并行改动。

涉及文件：

- `utils/visualization/skeleton.py`
- `utils/visualization/motion_video.py`（新增）
- `utils/visualization/video.py`（删除）
- `utils/visualization/__init__.py`
- `eval/vae/evaluate_reconstruction.py`
- `tests/test_visualization.py`（新增）

尚未完成的后续事项：

- 若后续评测需要original/reconstruction横向拼接，应在`eval/vae`内新增任务级comparison工具和明确的FFmpeg依赖测试，不恢复通用visualization中的旧250行composite实现。
- 全仓测试需要等待Dataset/collate并行重构稳定后重跑；当前visualization自身与其VAE eval调用链已经独立验证。

## 2026-07-15 · 删除旧多阶段step semantics

类型：旧训练模块物理删除 / checkpoint步数修正 / 公共导出清理

实际改动内容：

- 删除`utils/training/step_semantics.py`与`utils/training/module_step.py`，移除`StepSemantics`、`CheckpointStepInfo`、resume offset、runtime max-step重写、scheduler step重写和Lightning module adapter。
- 清理`utils/training/__init__.py`中的全部旧step API导出，不保留兼容别名。
- `BasicLightningModule.training_step()`直接使用Lightning `global_step + 1`记录`ckpt_absolute_step`，语义固定为当前batch完成后累计的optimizer steps。
- 增加migration guard，固定两个文件和旧API不得回归；训练测试同时断言checkpoint metric不再经过`ckpt_step_info()`且明确包含completed-step修正。

改动理由：

- 旧实现服务于FloodNet中“resume后重新计数的self-forcing第二阶段”，依赖未在Floodcontrol赋值的`_resume_step_offset`和当前配置不存在的`self_forcing.enabled`。
- 当前唯一活动调用最终只把`global_step`包装成checkpoint metric，其他resume/runtime/scheduler函数均无调用和专项测试；Lightning已经负责恢复global step、optimizer、scheduler与loop state。
- metric在`training_step()`读取的是本batch执行前的global step；旧逻辑导致完成300000步的历史checkpoint文件名为`step_299999.ckpt`。直接记录`global_step + 1`后，未来300000步run将保存为`step_300000.ckpt`。
- 后续LDF self-forcing应由global-step驱动的显式schedule实现；若最终采用独立第二阶段Trainer，再设计写入配置/checkpoint的`TrainingPhase`，而不是恢复隐藏offset推断。

验证：

- training与migration定向回归：`29 passed`。
- 修改文件`py_compile`通过，`git diff --check`通过；残留搜索仅在反回归测试中出现旧模块/API名称。
- 全仓`pytest -q`完成执行：`110 passed, 4 failed`。四项失败均位于本轮未修改的并行LDF/VAE encoder-window重构：三项`tests/test_ldf_data.py`期望旧`body_with_context/context_frame_valid_mask`合同，一项`tests/test_vae_model.py`仍以Python int调用当前要求tensor的`context_token_count`；本轮未越界修改这些并行问题。

涉及文件：

- `utils/training/step_semantics.py`（删除）
- `utils/training/module_step.py`（删除）
- `utils/training/lightning_module.py`
- `utils/training/__init__.py`
- `tests/test_migration_guards.py`
- `tests/test_vae_training.py`

尚未完成的后续事项：

- 当前checkpoint callback仍使用`ckpt_absolute_step`作为monitor以保留最近20个checkpoint；后续若切换到Lightning内建`{step}`命名，应同时验证`save_top_k`排序和resume命名，不能只替换filename模板。
- 上述四项全仓失败需要在正在进行的LDF context-window协议重构中统一测试与实现，和本次step模块删除无依赖关系。

## 2026-07-15 · 简化BodyVAE模型、checkpoint与显式cache边界

类型：VAE架构简化 / 强协议删除 / checkpoint与runtime迁移 / 测试与文档

实际改动内容：

- 将`BodyVAE`收缩为以计算为中心的模型：构造函数直接读取普通motion/latent statistics NPZ，只保留shape、finite、positive-std检查；公共方法继续使用`encode/decode`表示raw latent、`tokenize/detokenize`表示normalized deterministic mu。
- 删除模型中的architecture/tokenizer hash、statistics metadata和tokenizer identity。physical statistics仍是persistent buffers；训练后计算的latent mean/std保持non-persistent，因此训练checkpoint加载不会覆盖配置中的latent statistics。
- 将`VAEDecoderState`简化为只含causal convolution caches；删除session UUID、token index、模型身份和snapshot事务。`BodyVAE.init_decoder_state()`、`decode_step()`与`detokenize_step()`显式接收并返回state，不恢复module-global cache或`first_chunk`开关。
- 将VAE权重加载统一为`load_vae_checkpoint(model, checkpoint_path, use_ema=True)`：训练、评测、未来LDF和runtime直接读取Lightning训练checkpoint，默认应用EMA shadows、校验四个physical statistics buffer并冻结模型；旧checkpoint中的placeholder latent buffers在加载前移除。
- 物理statistics工具改为只写四组mean/std数组；latent statistics工具改为直接从配置构建模型、从训练checkpoint加载EMA encoder并只写`mean/std`，继续保留posterior finite检查和完整train split扫描，不生成逐样本latent cache。
- 删除`utils/vae_tokenizer.py`、`utils/training/vae/factory.py`、`utils/inference/vae_decoder.py`、`tools/export_vae_tokenizer.py`和`tools/pretokenize_body_latents.py`。配置直接target `models.vae_wan_1d.BodyVAE`，评测与stream metrics迁移到模型的显式state接口。
- 删除`utils/motion_artifact.py`中的`VAEStatistics/LatentStatistics`强metadata对象；保留离线motion转换本身需要的contract/converter与确定性yaw辅助逻辑。
- 更新VAE/data设计文档和README，明确EMA训练checkpoint是唯一权重来源、statistics是普通NPZ、cache只含计算状态；测试使用独立helper生成临时identity arrays，不向生产模型重新加入identity fallback。

改动理由：

- tokenizer bundle、artifact hash、yaw/context metadata和随机session身份没有改变VAE数学计算，却将模型、数据产品和runtime生命周期耦合在一起，显著增加研究迭代成本。
- 直接加载训练checkpoint既避免旧代码在多个入口重复实现EMA逻辑，也不需要额外维护一份bundle；latent normalization通过non-persistent buffer与checkpoint权重自然分离。
- 显式cache仍能避免旧版隐藏module cache造成的并发污染；去掉与卷积计算无关的事务字段后，state可以由调用方按普通研究代码方式复制和持有。

验证：

- VAE定向回归：`72 passed`。
- 全仓`pytest -q`：`101 passed`；仅出现环境无法初始化NVML的PyTorch warning，不影响CPU测试结果。
- `py_compile`覆盖BodyVAE、condition、checkpoint、训练data/module、statistics工具、VAE eval、stream metrics和新测试，通过。
- `git diff --check`通过；活动代码与当前设计文档中不存在`vae_tokenizer`、`VAEDecoderSession`、training factory、tokenizer/architecture hash或`tokenizer_ema.pt`依赖。
- 使用现有`step_299999.ckpt`、HumanML motion statistics和现有latent statistics实际构建并加载正式128D模型成功：EMA模型为eval/frozen，encoder与decoder causal context均为24 tokens。已有300k checkpoint参数及外部数据文件未被修改。

涉及文件：

- `models/vae_wan_1d.py`、`utils/conditions/vae.py`
- `utils/training/vae/{checkpoint,data,lightning_module,__init__}.py`
- `utils/motion_artifact.py`、`metrics/stream.py`、`eval/vae/evaluate_reconstruction.py`
- `configs/{vae,vae_multi}.yaml`
- `tools/{compute_vae_stats,compute_vae_latent_stats}.py`
- 删除：`utils/vae_tokenizer.py`、`utils/training/vae/factory.py`、`utils/inference/vae_decoder.py`、`tools/export_vae_tokenizer.py`、`tools/pretokenize_body_latents.py`
- `tests/{__init__,vae_helpers,test_vae_model,test_vae_tokenizer,test_vae_loss,test_vae_contract,test_vae_eval,test_vae_training,test_vae_data_pipeline,test_migration_guards}.py`
- `README.md`、`docs/rearchitecture/{02_DATA_PIPELINE,02_VAE_AND_BODY_REPRESENTATION}.md`

尚未完成的后续事项：

- LDF在线encoder context sampler与HybridMotion batch尚未接线；后续直接构造带motion/latent stats的`BodyVAE`并调用公共checkpoint loader，不再增加tokenizer bundle层。
- Web commit-time decode尚未接线；接线时由每条stream直接持有`VAEDecoderState`，cache归属由runtime对象生命周期保证。
- 现有外部`tokenizer_ema.pt`不再被代码使用；本次没有删除或改写运行目录中的历史文件。

## 2026-07-15 · 增加root5/body265到HumanML263的T2M评测适配器

类型：评测表示转换 / round-trip验证 / 真实T2M embedding对比

实际改动内容：

- 新增`metrics/humanml.py`的`convert_root5_body265_to_humanml263()`，将显式root5与physical body265恢复为标准HumanML263：root半角forward transition、heading-local root displacement、root height、heading-canonical non-root positions、21-joint local rotation6d、heading-local joint displacement和contacts。
- 适配器直接从22个global rotation矩阵按HumanML22父子关系恢复21个local rotations；positions与velocities使用HumanML quaternion方向转换，并自动消除任意全局translation/yaw对评测表示的影响。
- 明确标准HumanML一行拥有到下一pose的forward transition：`F`帧physical pose严格产生`F-1`行263D。`tail="drop"`用于严格round-trip；`tail="approximate"`通过最后一次观测transition补齐长度，供与固定长度baseline做正式T2M横向比较时显式选择。
- 新增`tools/compare_humanml_adapter.py`，在原始HumanML263数据上执行263 → root5/body265 → 263，分别统计root/position/rotation/velocity/contact误差，并使用与FloodDiffusion相同的T2M movement/motion encoder、mean/std计算embedding cosine、L2与FID漂移。
- 新增round-trip、batch/approximate-tail和全局yaw不变性测试；VAE设计文档增加标准T2M评测边界，明确不为body265另训不可横向比较的FID encoder。

改动理由：

- FloodDiffusion的HumanML evaluator固定消费263D并在去掉4维contacts后将259D送入预训练movement encoder；直接将body265送入或重训新encoder都会失去与既有FID/R-Precision结果的可比性。
- root5/body265保存physical absolute motion，而HumanML263保存heading-canonical pose与forward transition；显式适配器能将内部生成表示与公开评测表示隔离，并量化转换本身带来的偏差。
- 最后一条forward transition依赖未来pose，不能从有限physical clip严格恢复；因此接口显式区分数学验证用drop与等长baseline比较用approximate，不静默混合两种语义。

验证：

- 适配器定向与相邻数据/migration测试：`30 passed`。
- 全仓`pytest -q`最终复跑：`110 passed`；仅有环境无法初始化NVML的PyTorch warning。
- `py_compile`覆盖适配器、比较工具和测试，通过；`git diff --check`通过。
- 使用HumanML3D val split前128条真实`new_joint_vecs`及正式预训练T2M evaluator实测：
  - `tail="drop"`：embedding cosine mean `0.99999994`，embedding L2 mean `0.00121323`，FID经非负数值截断为`0.0`；root/position/rotation/contact MAE分别约`1.43e-8/7.21e-9/4.33e-8/0`。
  - exact-drop velocity MAE约`4.76e-6`，最大`0.06277`；检查显示偏差集中于少量原始263样本中velocity通道与其position forward difference本身不一致，而非root/pose逆变换漂移。
  - `tail="approximate"`：embedding cosine mean `0.99999815`、L2 mean `0.00811251`、FID `1.4939e-4`，确认尾transition近似对等长T2M比较的影响很小。
  - 将drop结果直接与未裁剪完整reference比较时，因T2M内部`length // 4`少一个movement token，embedding L2增至`0.06132`、FID为`0.0025899`；因此固定长度baseline横向比较应显式使用approximate tail，其适配偏差更小且长度一致。

涉及文件：

- `metrics/humanml.py`
- `tools/compare_humanml_adapter.py`
- `tests/test_humanml_adapter.py`
- `docs/rearchitecture/02_VAE_AND_BODY_REPRESENTATION.md`
- `docs/DEVELOPMENT_LOG.md`

尚未完成的后续事项：

- 新版LDF真实生成评测尚未恢复；接线时严格round-trip测试使用drop，与既有固定长度baseline做标准指标比较时显式使用approximate tail，并继续复用FloodDiffusion原始T2M checkpoint与mean/std。
- 正式结果需同时记录tail策略；不得把drop与approximate产生的不同有效长度混在同一组FID统计中，也不为此修改模型内部strict-4时间合同。

## 2026-07-15 · root5/body265 Dataset与训练数据层重构

类型：Dataset边界重构 / VAE与LDF collator实现 / artifact协议简化 / statistics与evaluation迁移 / 测试与文档

实际改动内容：

- 将`HumanML3DDataset`与`BABELDataset`改为两个互不继承的source Dataset；各自解析split、motion和`caption#tokens#start#end`文本，统一返回完整未裁剪的`dataset/name/root_motion/body_motion/body_feature_valid_mask/text_data` sample。Dataset不再执行crop、yaw、translation rebase、previous-root推导或padding；旧NPZ中的version/hash/FPS等多余metadata直接忽略。
- `MultiDataset`收敛为子Dataset实例化与`ConcatDataset`组合，不再提供collate。VAE配置删除Dataset collate入口，并分别配置HumanML3D/BABEL的独立`text_path`，现有motion NPZ与原始texts可直接组合使用。
- 在`utils/training/vae/data.py`实现`VAEWindowCollator`：train执行四帧对齐的随机长度/起点crop、同步全局yaw与XZ translation rebase，validation固定取确定性前缀且关闭随机增强；collator统一推导previous-root、frame/feature mask并完成不同长度batch padding，保持BodyVAE训练batch字段。
- 新增`utils/training/ldf/data.py`与package导出，实现简单四帧对齐active window、固定`encoder_context_tokens × 4`左侧body context、clip起点不足时左零填充及context mask、active root/body/previous-root与padding mask。同一时间区间多caption在train随机选一个、validation取第一个；与active window相交后输出裁剪过的相对token区间。本轮未恢复`train_ldf.py`，也未加入OriginEpoch、future observation或history/generation noise。
- 离线预处理NPZ改为只写`root_motion/body_motion/body_feature_valid_mask`三个字段；HumanML3D/BABEL构建器发布过滤后的split TXT并复制原始texts，不再写per-sample source SHA256、converter/contract/FPS metadata或额外build summary文件。已有完整target仍可被resumable构建跳过。
- physical statistics与latent statistics改为遍历Dataset full sample；前者保留四点yaw quadrature，后者使用显式seed的普通`torch.Generator`生成yaw，不再使用sample identity hash。VAE reconstruction evaluation直接实例化Dataset，也支持显式传入单Dataset或`MultiDataset`进行评测。
- 删除运行时`utils/motion_artifact.py`及condition/model中的artifact contract常量与`BodyVAE.contract_version`；保留root/body维度于`motion_process.py`、四帧时间常量于`token_frame.py`、263→265转换于`tools/`。没有恢复已删除的`utils/vae_tokenizer.py`。
- 重写Dataset/VAE data测试并新增LDF data测试，覆盖旧metadata兼容、完整motion/text、BABEL独立性、Multi source identity、VAE随机/确定性crop与padding、LDF context/文本备选，以及单Dataset/Multi statistics和reconstruction smoke。更新数据架构、VAE表示文档与README，明确Dataset和training data的新所有权边界。

改动理由：

- source Dataset应只回答“磁盘上的一个完整样本是什么”，训练任务的窗口、增强、边界与batch形状不应反向污染HumanML3D/BABEL加载接口。
- root5/body265 schema已经固定后，runtime hash/version血缘和重复manifest校验没有改变模型数学语义，却让模型、Dataset、statistics、evaluation与离线converter互相耦合；最小NPZ与普通seed更适合当前研究迭代。
- VAE与LDF对同一完整motion需要不同视图：VAE需要随机重建crop，LDF还需要冻结encoder的左历史与文本token timeline。把两者放在各自training data模块后，MultiDataset可以稳定复用，也为后续LDF trainer恢复保留清晰接点。

验证：

- `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m py_compile datasets/humanml3d.py datasets/babel.py datasets/multi.py utils/training/vae/data.py utils/training/ldf/data.py tools/build_motion_artifact.py tools/preprocess_humanml3d.py tools/preprocess_babel.py tools/compute_vae_stats.py tools/compute_vae_latent_stats.py eval/vae/evaluate_reconstruction.py`通过。
- Dataset/VAE/LDF data定向回归：`16 passed`。
- 全仓`/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests -q`：`110 passed`。
- 使用现有共享数据实际读取HumanML3D val首样本`[192,5]/[192,265]`与3条文本、BABEL val首样本`[52,5]/[52,265]`与2条文本，确认旧NPZ和独立原始text路径兼容；仅出现Matplotlib cache目录回退到`/tmp`的环境提示。
- `git diff --check`通过；`rg`确认活动代码中不存在`utils.motion_artifact`、`collate_humanml3d`或`collate_multi`引用。

涉及文件：

- `datasets/{humanml3d,babel,multi,__init__}.py`
- `utils/training/vae/{data,__init__}.py`
- `utils/training/ldf/{data,__init__}.py`（新增）
- `tools/{build_motion_artifact,preprocess_humanml3d,preprocess_babel,compute_vae_stats,compute_vae_latent_stats}.py`
- `eval/vae/evaluate_reconstruction.py`
- `models/vae_wan_1d.py`、`utils/conditions/{vae,__init__}.py`
- `configs/{vae,vae_multi}.yaml`
- `tests/{test_vae_data_pipeline,test_multi_dataset,test_ldf_data,test_vae_training,test_migration_guards}.py`
- `README.md`、`docs/rearchitecture/{02_DATA_PIPELINE,02_VAE_AND_BODY_REPRESENTATION}.md`
- 删除：`utils/motion_artifact.py`

尚未完成的后续事项：

- `train_ldf.py`仍保持migration guard；后续需要把当前LDF data batch接到冻结EMA `BodyVAE.encode_window/tokenize_window`、latent normalization与`HybridMotion`训练输入，本轮没有越过该边界。
- OriginEpoch、future observation、history/generation noise和更强的epoch/DDP随机可复现策略仍待真实LDF训练协议确认；不应提前塞回Dataset。
- 现有外部旧NPZ继续可读；若需要用最小schema重新发布全量数据，应显式清理/选择新的output目录后运行预处理，本轮未改写共享数据目录。

## 2026-07-15 · HumanML3D Dataset回归旧版代码组织风格

类型：Dataset可读性重构 / 注释与方法边界整理 / 回归测试

实际改动内容：

- 参考旧版`FloodDiffusion/datasets/humanml3d.py`的阅读顺序，将新版`HumanML3DDataset`组织为`__init__ → _load_file_list → load_motion/load_text → __getitem__ → _process`；恢复`file_list`、`motion_path`、`text_path`和`self.dataset`这些直观成员命名，并使用旧版熟悉的motion/text分块注释。
- `_load_file_list()`只构建包含source identity和文件路径的轻量索引，不把全量body265数组预加载到内存；`_process()`在取样时懒加载完整motion、解析全部caption并组装统一sample。
- `load_motion()`只读取root5/body265/feature-valid三个字段，继续忽略旧NPZ metadata；`load_text()`保留HumanML3D的`0#0`整段caption语义并把秒区间转换为frame区间。
- 保持上一轮冻结的数据职责不变：Dataset不随机选择caption、不crop、不做yaw/rebase、不推导previous-root、不padding；这些处理仍属于VAE/LDF training collator。BABEL继续是独立Dataset，不恢复继承关系。
- 测试增加轻量索引断言，确保`self.dataset`只保存`dataset/name/motion_path/text_path`，公开sample合同没有变化。

改动理由：

- 旧版最好的部分是一个类内即可顺序理解“索引、加载、处理、输出”的数据生命周期；此前新版将record、motion和text拆成多个module-level helper，虽然短但阅读时需要来回跳转，source Dataset的主线不够突出。
- 直接照搬旧版的全量数组预加载、Dataset内随机裁剪和随机caption会破坏当前已确认的Dataset/training data边界；本次只继承其代码组织与注释方式，并保留body265所需的懒加载。

验证：

- HumanML3D/Multi定向回归：`11 passed`；包含旧metadata兼容、完整motion/text、轻量索引、VAE collator、Multi/statistics/evaluation smoke。
- 本次重构后的全仓`pytest tests -q`：`110 passed`。
- `datasets/humanml3d.py`的`py_compile`通过，`git diff --check`通过。
- 使用现有HumanML3D val split实际读取1450条索引中的首样本`012698`：root `[192,5]`、body `[192,265]`、3条caption；仅出现Matplotlib cache回退到`/tmp`的环境提示。

涉及文件：

- `datasets/humanml3d.py`
- `tests/test_vae_data_pipeline.py`

尚未完成的后续事项：

- BABEL当前已保持source独立和相同公开sample语义；若后续也希望统一阅读风格，可按同一方法骨架单独整理，但不应通过继承HumanML3D来减少代码。
- 本轮没有改变公开Dataset参数或训练配置，不需要重建现有HumanML3D_motion数据。

## 2026-07-15 · HumanML3D_motion文本自包含与VAE/LDF统一数据根目录

类型：数据迁移 / Dataset配置统一 / VAE与LDF真实数据smoke test

实际改动内容：

- 明确`HumanML3D_motion/artifacts/<name>.npz`并非只保存body265，而是保存`root_motion [F,5]`、`body_motion [F,265]`和`body_feature_valid_mask [F,265]`三个训练张量；现存旧metadata继续由Dataset忽略，未为本次迁移重写约2.9万个motion artifact。
- 将原始`HumanML3D/texts/*.txt`复制到`HumanML3D_motion/texts/`，使处理后数据根目录同时拥有motion、split和caption，不再要求HumanML配置跨目录读取原始数据集文本。
- `configs/vae.yaml`及`configs/vae_multi.yaml`中的HumanML `text_path`统一改为相对路径`texts`；`HumanML3DDataset`相对于split/data根目录解析该路径。BABEL配置本轮未改动。
- 更新数据架构文档，冻结自包含目录协议：NPZ只负责数值motion，TXT继续负责`caption#tokens#start#end`文本，文本不重复嵌入每个NPZ。VAE collator消费同一sample的motion字段并忽略文本；LDF collator消费相同motion以及解析后的`text_data`。
- 更新配置测试，锁定HumanML单数据集与MultiDataset均使用相对`texts`路径。

改动理由：

- VAE和LDF应共享同一个“完整样本”Dataset，区别只存在于各自training collator：VAE不需要caption，LDF需要caption timeline；不应为两个任务复制motion或维护两套Dataset。
- 将split、artifacts和texts放在同一个处理后数据根目录，可以整体移动、发布和配置数据集，避免motion来自`HumanML3D_motion`、文本却隐式来自`HumanML3D`的跨目录依赖。
- 文本保持独立TXT可复用原始HumanML3D标注格式，并避免同一caption被重复写入motion NPZ；Dataset负责在读取时统一组装`text_data`。

数据迁移结果：

- 运行`tools.preprocess_humanml3d`迁移文本并重新发布过滤后的split；已有兼容artifact全部跳过：`converted=0`、`skipped=29046`、`copied_texts=29046`。
- 发布后的split大小为train `23240`、val `1450`、test `4356`；过滤不足20帧样本train `144`、val `10`、test `26`，另过滤test非有限样本`2`。
- 当前共享数据目录结构为`HumanML3D_motion/{train.txt,val.txt,test.txt,artifacts/,texts/,motion_stats.npz}`。

验证：

- 使用真实val数据通过统一`HumanML3DDataset`读取首样本`012698`：root `[192,5]`、body `[192,265]`、caption `3`条。
- 同一sample经VAE collator得到root `[1,40,5]`、body `[1,40,265]`；经LDF collator得到active root `[1,40,5]`、`body_with_context [1,136,265]`和1条active text，确认24-token/96-frame encoder context与40-frame active window对齐。
- 全仓`/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest -q`：`110 passed`；`git diff --check`通过。

涉及文件与数据：

- `configs/{vae,vae_multi}.yaml`
- `tests/test_vae_training.py`
- `docs/rearchitecture/02_DATA_PIPELINE.md`
- `/data1/yuankai/text2Motion/FloodDiffusion/raw_data/HumanML3D_motion/texts/`及过滤后的split TXT

尚未完成的后续事项：

- `BABEL_motion`目前仍通过配置读取`BABEL_streamed/texts`；若需要与HumanML一致的可移植数据根目录，应单独迁移BABEL文本并改为相对`texts`。
- LDF数据合同和collator已能消费统一sample，但`train_ldf.py`仍保持migration guard，需在冻结EMA VAE在线encode接线完成后恢复训练入口。

## 2026-07-15 · 消除Dataset一次性module helper

类型：Dataset接口简化 / 私有依赖清理 / BABEL风格对齐

实际改动内容：

- 删除`datasets/humanml3d.py`中只为类内流程服务的`_resolve_text_root()`与`_load_motion()`两个module-level函数；相对text目录解析直接放入`_load_file_list()`，NPZ读取与基础tensor合同检查直接放入`HumanML3DDataset.load_motion()`。
- `HumanML3DDataset.load_motion()`不再转调第二层私有函数，代码从类入口即可顺序阅读；旧metadata忽略、完整motion懒加载和公开sample字段保持不变。
- 为消除BABEL对HumanML3D私有实现的反向依赖，将`BABELDataset`同步整理为`_load_file_list/load_motion/load_text/_process`类内流程；BABEL不再导入HumanML3D的私有helper，也没有通过继承复用source逻辑。
- 没有引入新的common/base Dataset文件，避免为了两段简单source IO再次增加间接层。

改动理由：

- 只被类方法转调一次的module helper没有形成可复用抽象，反而让一次NPZ读取跨两个函数跳转；路径解析只有几行，也更适合留在建立样本索引的上下文中。
- BABEL直接导入HumanML3D的下划线函数虽然没有类继承，但仍形成隐藏耦合；两个source Dataset各自拥有少量明确IO逻辑比共享私有实现更容易修改和阅读。

验证：

- Dataset/VAE/LDF data定向回归：`24 passed`。
- 全仓`pytest tests -q`：`110 passed`。
- `datasets/humanml3d.py`与`datasets/babel.py`的`py_compile`通过，`git diff --check`通过。
- `rg`确认`datasets/`中不存在`_resolve_text_root`、module-level `_load_motion`或从HumanML3D导入私有符号的代码。

涉及文件：

- `datasets/humanml3d.py`
- `datasets/babel.py`

尚未完成的后续事项：

- 两个Dataset当前仍保留必要的字段、shape、四帧对齐与finite检查；若后续继续精简，应单独决定哪些错误由离线预处理保证，避免在“减少函数”任务中顺带改变坏数据的失败位置。

## 2026-07-15 · BABEL与MultiDataset风格整理及Generate职责说明

类型：Dataset可读性优化 / Multi组合逻辑简化 / 数据架构文档

实际改动内容：

- 将`BABELDataset`进一步对齐当前`HumanML3DDataset`的代码组织与注释：保留`file_list/motion_path/text_path/self.dataset`轻量索引、`load_motion/load_text/_process`类内生命周期、旧NPZ metadata忽略说明和完整分段caption输出；补充`_process()`职责注释。
- 保持BABEL source语义：同时间区间的多个caption全部留在`text_data`，Dataset不随机选择、不裁剪；后续caption备选策略仍由LDF collator处理。
- 简化`MultiDataset`：显式保存`split`，逐项只向source Dataset传递该项的target、对应split路径、motion/text路径与FPS，然后由`ConcatDataset`管理长度和索引；`dataset_lengths`直接读取各子Dataset长度，不再从累计边界反推。
- Multi不提供collate、crop、padding或训练增强；不同任务继续使用VAE/LDF各自training data collator。
- 在数据架构文档新增“为什么不保留`datasets/generate.py`”：明确旧`GenerateDataset`只是硬编码prompt/时长并创建全零feature/token的test请求生成器，不是拥有真实root5/body265与valid mask的数据源；未来固定prompt批量生成应归具体eval/inference request builder，而不是训练Dataset。
- 测试补充BABEL轻量索引与Multi split断言；现有migration guard继续确保`datasets/generate.py`不返回。

改动理由：

- BABEL与HumanML3D共享公开sample合同，但应各自在一个文件内清楚表达source IO和文本语义；统一阅读顺序比共享私有helper或继承更直接。
- 旧版Multi最重要的是子配置隔离，旧通用collate和config复制不适合当前任务专属collator边界；直接参数传递与`ConcatDataset`已足够。
- 全零feature/token不能代表“待生成动作”的ground truth。将这种占位对象放进Dataset会污染statistics、reconstruction和LDF有效性mask，混淆数据事实与生成请求。

验证：

- Dataset/Multi/VAE/LDF/migration定向回归：`43 passed`。
- 全仓`pytest tests -q`：`110 passed`。
- `datasets/babel.py`与`datasets/multi.py`的`py_compile`通过，`git diff --check`通过。
- 使用正式`configs/vae_multi.yaml`实际构建val MultiDataset：HumanML3D 1450条、BABEL 3645条、合计5095条；两侧首样本分别为body `[192,265]`/3条文本与`[52,265]`/2条文本。仅出现Matplotlib cache回退到`/tmp`的环境提示。
- 活动Python/YAML代码中不存在`GenerateDataset`或`datasets.generate`引用。

涉及文件：

- `datasets/humanml3d.py`
- `datasets/babel.py`
- `datasets/multi.py`
- `tests/test_multi_dataset.py`
- `docs/rearchitecture/02_DATA_PIPELINE.md`

尚未完成的后续事项：

- 若未来需要可复现的固定prompt生成集，应在具体eval目录定义typed generation request及其timeline，不恢复包含零动作占位的通用Dataset。
- 本轮没有改变公开root5/body265 sample合同、配置路径或共享数据文件，不需要重新预处理数据。

## 2026-07-15 · 纯VAE文本IO与训练基类清理

类型：训练配置收敛 / Lightning基类简化 / 无效接口删除

实际改动内容：

- `configs/vae.yaml`将`text_path`显式设为`null`；`configs/vae_multi.yaml`中的HumanML3D和BABEL子配置也都设为`null`。Dataset公开sample合同不变，但纯VAE训练不再打开和解析未消费的caption TXT。
- 删除两份VAE配置中未接通训练入口的`test_steps`、`test_meta_paths`和`test_batch_size`；正式重建测试继续由独立`eval/vae/`入口负责。
- 将`BasicLightningModule`收敛为模型实例化、optimizer/scheduler、EMA、checkpoint和train/validation loss五项职责；删除旧LDF遗留的多validation-loader test路由、空metric/test/render hooks、tokenizer环境变量副作用、`FLOODNET_DEBUG`追踪和`ckpt_hash.txt`写入。
- 统一缓存实际可训练参数供optimizer和EMA使用，并在fit/validation开始时显式移动EMA；train/val日志改用batch中的真实batch size，直接记录loss tensor，不再逐项调用`.item()`触发GPU同步。
- 保留现有训练checkpoint的`state_dict + ema_state`格式以及物理statistics恢复检查；新增配置和源码回归断言，防止无效test接口与debug/hash副作用回流。
- 数据架构文档明确：需要文本的任务设置`text_path: texts`，纯VAE设置`null`。

改动理由：

- VAE只重建root/body motion，加载caption会带来重复文件IO，却不参与collator、模型或loss；是否读取文本应由任务配置表达，而不是改变统一Dataset。
- 当前训练基类只有VAE使用，旧LDF时期为inline test、render和状态hash保留的空扩展点已经没有调用方；删除这些分支可使正式训练控制流从入口直接读到train/validation语义。
- 使用配置batch size会错误加权最后一个不足整批的batch，且`.item()`会造成不必要的设备同步；日志应使用运行时batch和原始tensor。

验证：

- Python `py_compile`通过：`utils/training/lightning_module.py`、`tests/test_vae_training.py`。
- 最终工作区VAE配置、Dataset、Multi、checkpoint定向回归：`25 passed`。
- 使用真实HumanML3D数据完成CPU Lightning单步train/validation、`last.ckpt`保存、全新module恢复和再次validation；确认VAE sample的`text_data=[]`、EMA checkpoint恢复成功。
- 最终工作区全仓`/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests -q`：`113 passed`。
- `git diff --check`通过。

涉及文件：

- `configs/{vae,vae_multi}.yaml`
- `utils/training/lightning_module.py`
- `tests/test_vae_training.py`
- `docs/rearchitecture/02_DATA_PIPELINE.md`
- `docs/DEVELOPMENT_LOG.md`

尚未完成的后续事项：

- `train_vae.py`的`train: false`分支仍沿用`test_ckpt`命名并允许空checkpoint；若继续保留该入口，应改成明确的validation checkpoint接口并在为空时失败，或完全交给`eval/vae/`。
- `configs/paths_default.yaml`中的WandB凭据管理不属于本次两个改动项，仍应另行迁移为环境变量并轮换现有key。

## 2026-07-15 · BABEL文本自包含与Dataset固定source identity

类型：Dataset公共metadata修正 / BABEL共享数据迁移 / 配置与测试

实际改动内容：

- `HumanML3DDataset`和`BABELDataset`不再以split文件父目录名构造`sample["dataset"]`，分别固定返回`HumanML3D`和`BABEL`；duplicate identity同步使用固定source名称，同一source通过多个split路径重复提供相同sample ID时继续fail-fast。
- `configs/vae_multi.yaml`中的BABEL `text_path`改为相对目录`texts`，不再引用`BABEL_streamed/texts`；保留HumanML在纯VAE配置中的`text_path: null`，需要caption的LDF数据入口可对两个source显式使用`texts`。
- 使用现有resumable `tools.preprocess_babel`重新发布共享`BABEL_motion`：现有14087个root5/body265 artifact全部通过检查并跳过重算，将正式train/val涉及的14087个caption原子复制到`BABEL_motion/texts/`，重新写出相同过滤规则下的train/val split；未删除或修改原始`BABEL_streamed/texts`。
- 测试增加任意目录名下HumanML/BABEL仍返回固定source identity的回归，更新Multi/VAE batch与reconstruction sample的metadata期望，并验证BABEL预处理产物包含自有text。
- 数据架构文档明确两个处理后dataset root均可自包含split、artifacts和texts，且公开source identity不随目录移动或重命名变化。已有`eval/vae/output`历史JSON未重写，其中旧目录派生标签作为历史运行结果保留。

改动理由：

- 数据目录名是部署路径，不是数据源语义；将它写入batch会使同一数据在复制、挂载或重命名后获得不同metadata，影响Multi source分组和评估记录的稳定性。
- `BABEL_motion`已经拥有正式motion与split，却跨目录读取caption，不便整体移动或发布；复制对应文本后，其结构与`HumanML3D_motion`同构，LDF可以仅通过一个dataset root获得motion和文本。

数据迁移与验证：

- BABEL迁移结果：`converted=0`、`skipped=14087`、`copied_texts=14087`；train `10442`、val `3645`，不足20帧过滤train `178`、val `81`，missing与nonfinite均为0。
- 逐项检查train/val：重复ID 0、缺失artifact 0、缺失text 0；两split合计14087个唯一ID，磁盘上artifact与text均为14087个。
- 真实val smoke：HumanML首样本返回`HumanML3D`、root/body `[192,5]/[192,265]`和3条caption；BABEL首样本返回`BABEL`、`[52,5]/[52,265]`和2条caption。两者均通过VAE 40-frame collator和LDF `96 context + 40 active` collator，batch identity保持固定。
- Dataset/VAE/LDF定向测试：`35 passed`；最终全仓`pytest -q`：`113 passed, 1 warning`，warning仅为测试环境无法初始化NVML。
- 相关Python文件`py_compile`通过，活动代码/配置/测试中不存在目录名推导source identity或`BABEL_streamed/texts`引用，`git diff --check`通过。

涉及文件与数据：

- `datasets/{humanml3d,babel}.py`
- `configs/vae_multi.yaml`
- `tests/{test_vae_data_pipeline,test_multi_dataset,test_vae_eval,test_vae_training}.py`
- `docs/rearchitecture/02_DATA_PIPELINE.md`
- `/data1/yuankai/text2Motion/FloodDiffusion/raw_data/BABEL_motion/{texts,train.txt,val.txt}`

尚未完成的后续事项：

- `train_ldf.py`仍保持migration guard；未来LDF配置应为HumanML和BABEL都显式设置`text_path: texts`并接入现有`LDFWindowCollator`，本轮没有提前恢复trainer。
- 旧评估artifact中的`HumanML3D_motion/BABEL_motion`标签不会迁移；比较新旧评估时应将其视为历史metadata名称差异，而不是数据或模型变化。

## 2026-07-15 · LDF真实causal context与逐样本window encode

类型：LDF数据合同修正 / VAE在线编码接口 / 因果数值回归

实际改动内容：

- `LDFWindowCollator`删除cold-start假左零context与冗余`context_frame_valid_mask`。每个样本只携带`min(window_start_token, encoder_context_tokens)`个真实历史token，批内统一排列为`[真实context | active window | 右padding]`。
- `context_token_count`由全batch共享标量改为`long [B]`，逐样本记录active window在encoder输入中的真实offset；active root/body及其frame/feature mask继续作为独立右padding张量，不引入第二套motion mask或Root命名。
- `BodyVAE.encode_window()`与`tokenize_window()`改为接受逐样本context count。接口验证四帧patch内frame validity一致、有效encoder token为连续左前缀、context不超过encoder感受野和样本有效长度，并保证每个样本至少保留一个active token。
- encoder仍只执行一次batch前向；随后按各样本context offset gather active `mu/logvar`，统一右补零。`tokenize_window()`在latent归一化后再次清零active padding，避免非零latent mean将padding变成数值非零。
- `train_ldf.py`的migration guard更新为已完成real-context collator与window encode边界；真实训练仍明确阻塞在frozen EMA online encode调用和HybridMotion Lightning batch尚未接线。
- 数据与VAE设计文档同步冻结真实历史、逐样本offset和右侧batch padding协议。

改动理由：

- 显式补在序列左侧的零token不是causal convolution的“不存在历史”。这些token会经过input projection、bias、normalization和残差块，产生非零隐藏状态并污染active posterior；真实序列起点必须直接由网络自身的causal左边界处理。
- batch内样本的window起点不同，真实可用历史长度天然不同。共享固定offset无法同时表达cold start、partial context和full context，必须使用逐样本count提取active posterior。
- latent padding在归一化后可能因非零mean/std变为非零，因此padding语义必须在normalized tokenizer出口再次显式恢复。

验证：

- LDF collator覆盖cold start无假前缀、partial/full真实历史、混合context与active长度、逐样本count、右侧唯一padding、previous root和文本裁剪语义。
- VAE encoder新增cold-start、partial-context和full-context三类数值parity；window active `mu/logvar`均与full-clip对应token在`atol=1e-6`内一致。
- 混合context batch验证逐样本gather、posterior右padding为零，以及使用非identity latent statistics后normalized `mu` padding仍为零。
- LDF/VAE/migration/training定向回归：`46 passed`；最终全仓`pytest tests -q`：`117 passed`。
- 相关Python文件`py_compile`通过，`git diff --check`通过；活动代码中不再输出独立`context_frame_valid_mask`或使用标量`context_token_count`。

涉及文件：

- `models/vae_wan_1d.py`
- `utils/training/ldf/data.py`
- `tests/{test_ldf_data,test_vae_model,test_migration_guards}.py`
- `train_ldf.py`
- `docs/rearchitecture/{02_DATA_PIPELINE,02_VAE_AND_BODY_REPRESENTATION}.md`

尚未完成的后续事项：

- 本轮只完成LDF data batch与VAE tokenizer边界；frozen EMA encoder的`no_grad`在线调用、HybridMotion组装、root/latent noise与真实LDF Lightning module仍未实现，训练入口继续fail-fast。

## 2026-07-15 · 新版Hybrid inference核心与原子commit事务

类型：在线推理架构重构 / 条件编译 / LDF流式接口 / VAE commit接线 / 测试与文档

实际改动内容：

- 物理删除没有真实LDF编译能力的`ConditionManager`、混合旧reference语义的`route_condition.py`、宽松`text_condition.py`以及从body recovery反推world root的`RootTimeline`。
- 将`utils/inference/geometry.py`重写为严格、无状态的world XZ route几何；新增`route.py`，以time-parameterized `RoutePlan`明确区分world/relative输入参考与hold/release终点行为。relative route只在update时解析一次，正式状态只保存world事实。
- 新增`TextTimeline`和CPU派生`TextEmbeddingCache`；文本区间使用absolute token半开区间，允许token-aligned prompt update且不把embedding cache纳入persistent generation state。
- 新增feature-masked `RootObservationTimeline`和`InferenceConditionCompiler`：每step按当前`window_origin/commit_index`重新采样20 FPS route、执行world到model translation、feature-wise observation覆盖、root statistics归一化、committed mask清空和future constraint packing；`CompiledCondition`记录window与三类revision，拒绝rolling后复用时间错位条件。
- 新增batch-1 `InferenceSession`、`InferenceConfig`、session级`GuidanceConfig`、`GeneratedMotionChunk`与`InferenceSnapshot`。一次step先获得candidate LDF state与committed Hybrid token，再从physical world root派生backward local-root并调用`BodyVAE.detokenize_step()`；所有候选结果、heading和shape通过后才一次性交换LDF/VAE/origin/boundary state。
- LDF公开`normalize_root()/denormalize_root()`，所有generate/stream入口显式接收per-call CFG scales，避免共享模型被某个session修改；新增scheduler-aware `rebase_stream_state()`，window roll后分别按`1-beta`平移clean/partial noisy root，pure noise、latent和RNG保持不变。
- 修复LDF此前只投影`LDFPrediction.clean_root_motion`、却可能提交非单位heading state的问题：offline generate结束时投影全部clean root，stream在每个commit边界投影新clean token并将同一值写入persistent history。inference只验证root5 manifold，不进行模型外root replacement。
- `03_STREAMING_ACTIVE_WINDOW.md`从开放问题改为当前已实现合同，明确状态所有权、时间/坐标协议、route/text update、future约束、原子commit、snapshot、control latency和Web未完成边界。

改动理由：

- 新版LDF已经唯一拥有Hybrid active window、三角调度和rolling，BodyVAE唯一拥有causal decoder cache；继续保留旧root recovery、第二套timeline或未执行的delay/blend字段会重新制造FloodNet式重复状态所有权。
- world route必须在origin变化后重新编译，future constraint长度必须与motion RoPE/beta/VAE长度解耦，condition必须携带window stamp才能避免shape相同但absolute timeline错位。
- LDF与VAE都是out-of-place state transition，runtime可以在不预先复制整个state的情况下实现失败不提交；per-session CFG和独立decoder state允许多个Web session安全共享eval模型。
- root5的cos/sin单位圆是生成状态合同，不应只存在于Body Stage的临时clean-root view，也不能留给inference在commit后偷偷修正。

验证：

- inference/LDF/VAE/migration定向回归：`59 passed`；新增测试覆盖严格route时间、relative route一次性world解析、text cache、route/root observation编译、past mask、future position、scheduler-aware rebase、四帧commit、snapshot确定性、decoder失败回滚、roll后origin更新和session CFG不修改共享模型。
- 使用未patch的真实tiny Root/Body Transformer和tiny BodyVAE执行CPU端到端一步session smoke：成功输出root `[1,4,5]`、continuous body `[1,4,261]`，committed heading norm约`0.9999999`。
- 最终全仓`/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests -q`：`134 passed`。
- inference与LDF相关Python文件`py_compile`通过；`git diff --check`通过；活动Python不存在旧`ConditionManager/RootTimeline/route_condition/text_condition`引用。

涉及文件：

- `models/diffusion_forcing_wan.py`
- `utils/inference/{__init__,geometry,route,text,condition,session}.py`
- 删除`utils/inference/{condition_manager,route_condition,text_condition,timeline}.py`
- `tests/{test_inference,test_ldf_stream,test_migration_guards}.py`
- `docs/rearchitecture/03_STREAMING_ACTIVE_WINDOW.md`

尚未完成的后续事项：

- 正式LDF checkpoint、UMT5 encoder loader、Web session锁/队列、输出收集与视频编码尚未接线；现有Web fail-fast保持不变。
- route update首版只在当前commit立即生效；如果实验需要scheduled route，必须实现真正的route timeline及分段feature compilation，不恢复旧的占位delay/blend协议。
- active band中途改变route/text的质量仍需要真实checkpoint评测，并与后续scheduled training/self-forcing共同验证control latency。
- 当前snapshot是同一已加载模型上的in-memory协议；跨进程持久化和checkpoint identity绑定留给部署阶段。

## 2026-07-15 · 统一VAE→LDF训练合同与正式128D配置

类型：LDF输入合同 / EMA tokenizer桥接 / statistics工具 / 正式配置 / 测试与文档

实际改动内容：

- `LDFInput`新增与`previous_root_frame`成对的`previous_root_valid_mask [B]`。全batch cold start使用`None/None`，随机crop混合batch使用physical boundary tensor与逐样本bool mask；Root Stage派生local-root时将mask直接传给统一backward codec。
- LDF forward、CFG和stream step input全部贯通boundary validity。同步stream在boundary存在时构造全true mask；训练batch可以同时正确表达cold-start与真实preceding-root样本。
- 新增`utils/training/ldf/lightning_module.py`。`LDFLightningModule`从正式训练checkpoint加载并冻结EMA BodyVAE，在`no_grad`下调用`tokenize_window()`生成normalized deterministic μ；physical active root按LDF root statistics归一化并reshape为`[B,T,4,5]`，两者共享active token mask后构造`HybridMotion`。
- LDF的`latent_dim`与`local_root_mean/std`直接由VAE实例注入，不在LDF配置维护第二份数值；初始化时验证latent statistics已加载、local-root statistics一致且VAE始终保持eval/frozen。normalized root与latent padding均显式清零。
- `BasicLightningModule`增加可选prebuilt model入口，使LDF训练模块能够先解析VAE/statistics再构造LDF；既有VAE自动实例化路径保持不变，optimizer与EMA仍只跟踪LDF可训练参数。
- 新增`tools/compute_ldf_root_stats.py`，按照20–200帧四帧对齐random window、首帧XZ rebase、随机全局yaw和显式seed计算train-split physical root5 mean/std。工具只拥有global-root statistics，local-root继续复用VAE motion statistics。
- 使用HumanML3D正式train split 23240条样本、每样本一个seeded window生成`HumanML3D_motion/root_stats.npz`。得到`root_mean=[-0.0016468, 0.9169331, -0.0018975, -0.0002812, -0.0030483]`、`root_std=[0.6728486, 0.1535048, 0.6822162, 0.7087607, 0.7054423]`。
- `configs/ldf.yaml`由8D identity-statistics model-core配置切换为正式128D合同配置：引用300k VAE EMA checkpoint、latent/motion/root statistics、24-token encoder context与8+8层Root/Body模型。注入字段不再出现在`model.params`。
- `train_ldf.py`的守卫更新为`BLOCKED_ON_LDF_TRAINING`：VAE/LDF合同已闭合，但noise/beta、text/constraint condition和root/latent v-predict loss尚未连接，因此真实训练仍不会误启动。
- README与三份重构文档同步记录逐样本boundary、唯一EMA bridge、root statistics policy和新的实际阻塞位置。

改动理由：

- `previous_root_frame`只有batch级optional无法表达同一batch中不同crop起点；没有逐样本mask会把cold-start占位boundary误当作真实历史，导致第一帧local yaw/velocity有效性错误。
- physical body到normalized latent的转换只属于LDF训练层。将EMA加载、deterministic μ和root normalization集中在一个Lightning边界，可以避免Dataset依赖模型，也避免LDF模型读取physical body265。
- latent维度和local-root statistics由decoder/tokenizer共同定义，必须以VAE为单一事实来源；global root则是LDF独有生成变量，应按其实际window/rebase/yaw分布独立统计。
- root与latent normalization会把数值零变成非零，故所有batch padding必须在normalization后再次由同一个token mask清零。

验证：

- LDF合同、data、forward、CFG、stream、training bridge、statistics和migration定向测试：`48 passed`。
- 最终工作区全仓`PY=... ./scripts/run_pytest.sh tests -q`：`134 passed, 1 warning`；warning仅为测试环境无法初始化NVML。
- 相关Python文件`py_compile`通过，`git diff --check`通过。
- 正式`configs/ldf.yaml`成功加载真实300k EMA checkpoint、motion/latent/root statistics并构造`LDF`：trainable参数`227,929,079`、latent维度128、encoder context 24、latent statistics ready。
- 两条真实HumanML3D train sample经random crop/yaw、真实context collator和正式EMA bridge得到root `[2,10,4,5]`与latent `[2,10,128]`，有效token数分别为7和10；`HybridMotion.validate()`通过。
- 单元测试覆盖mixed boundary validity、VAE/LDF local statistics共享、EMA frozen/eval、root/latent token对齐、非identity normalization、右padding归零以及active partial-token拒绝。

涉及文件与数据：

- `utils/conditions/ldf.py`
- `models/diffusion_forcing_wan.py`
- `utils/training/{lightning_module,ldf/lightning_module,ldf/__init__}.py`
- `tools/compute_ldf_root_stats.py`
- `configs/ldf.yaml`
- `train_ldf.py`
- `tests/{test_ldf_conditions,test_ldf_forward,test_ldf_cfg,test_ldf_training,test_ldf_statistics,test_migration_guards}.py`
- `README.md`
- `docs/rearchitecture/{01_MODEL_ARCHITECTURE_AND_IO,02_DATA_PIPELINE,02_VAE_AND_BODY_REPRESENTATION}.md`
- `/data1/yuankai/text2Motion/FloodDiffusion/raw_data/HumanML3D_motion/root_stats.npz`

尚未完成的后续事项：

- `LDFLightningModule._step()`仍明确失败；下一阶段需要设计并实现root/latent v-predict noise target、history/generation beta、文本/约束condition编译、loss与optimizer step后才能解除`train_ldf.py`守卫。
- Web/runtime尚未接入commit-time VAE decoder，继续保持自身的`BLOCKED_ON_BODY_VAE`守卫。

## 2026-07-15 · 收紧VAE→LDF tokenizer contract

改动类型：模型公共接口 / 训练bridge / 配置单一事实来源 / statistics状态校正 / 测试与文档

实际改动内容：

- `BodyVAE`删除LDF侧不需要的公共`encode_window()`及其一次性私有包装，只保留`tokenize_window()`作为context+active body到normalized deterministic μ的唯一窗口接口。该接口直接验证逐样本long context count、四帧patch内一致的frame validity、有效token左前缀、encoder causal context上限和至少一个active token，并按逐样本offset gather后清零active padding。
- `LDFLightningModule._create_clean_motion()`除批内最大shape外，新增逐样本token计数校验：`encoder有效token数 - context token数`必须逐项等于active root有效token数，拒绝最大长度相同但样本间root/body有效长度交叉错位的batch。
- bridge不再直接调用通用`normalize_features()`，统一通过LDF公共`normalize_root()`进入global-root归一化合同。
- `configs/ldf.yaml`删除手写`data.encoder_context_tokens: 24`；encoder context长度只由已加载VAE的只读属性提供。当前`root_stats.npz`在配置、工具说明、README与数据/VAE文档中明确标为provisional，而非正式训练分布资产。
- contract文档将raw posterior入口限定为完整VAE训练`encode()`，LDF窗口入口限定为`tokenize_window()`；同时记录正式H/G/F/C窗口与rebase冻结后必须通过共享训练sampler重新生成root statistics。
- 新增/改写回归测试，覆盖cold-start、partial/full context、mixed batch gather、窗口padding归零、partial encoder token拒绝、公共接口收敛、配置无context重复值，以及逐样本root/body token错位拒绝。

改动理由：

- LDF只消费normalized deterministic μ，公开窗口posterior既扩大接口又会让零填充的`logvar`具有可采样歧义；单一`tokenize_window()`更符合实际所有权。
- tensor的批内最大shape不能证明逐样本时间轴一致。若两个样本有效长度互换，原bridge会静默把错误latent mask套到root上，因此必须比较每个样本的active token数。
- encoder context由VAE架构决定，配置复制固定数值会在VAE层数或kernel变化后悄悄失配。
- root statistics依赖尚未冻结的训练窗口、OriginEpoch和rebase策略；当前简单crop统计只能验证bridge，不能提前声明为正式训练统计。

验证：

- 相关Python文件`py_compile`通过。
- VAE tokenizer、LDF bridge/data、migration和frame/token定向回归：`49 passed, 1 warning`。
- 全仓`PY=/home/yuankai/.conda/envs/flooddiffusion/bin/python ./scripts/run_pytest.sh tests -q`：`136 passed, 1 warning`；warning仅为测试环境无法初始化NVML。
- `rg`确认活动代码与设计文档不再调用`encode_window()`，正式LDF配置不再保存`encoder_context_tokens`副本；`git diff --check`通过。

涉及文件：

- `models/vae_wan_1d.py`
- `utils/training/ldf/lightning_module.py`
- `configs/ldf.yaml`
- `tools/compute_ldf_root_stats.py`
- `tests/{test_vae_model,test_ldf_training,test_migration_guards}.py`
- `README.md`
- `docs/rearchitecture/{02_DATA_PIPELINE,02_VAE_AND_BODY_REPRESENTATION}.md`
- `docs/DEVELOPMENT_LOG.md`

尚未完成的后续事项：

- 正式H/G/F/C训练窗口、OriginEpoch、history/generation状态与self-forcing尚未实现；下一阶段应先冻结共享window contract，再让collator与root-statistics共同消费它。
- 正式窗口/rebase/yaw分布冻结后必须重新生成`root_stats.npz`，当前provisional资产不得作为最终训练统计结论。
- `LDFLightningModule._step()`与`train_ldf.py`守卫保持不变；noise/beta、condition、v-predict loss和optimizer训练接线仍未完成。

## 2026-07-15 · Web runtime迁移到Hybrid InferenceSession与四帧chunk

改动类型：Web推理runtime / HTTP协议 / 前端播放 / 并发与生命周期 / 迁移清理 / 测试与文档

实际改动内容：

- 将Web生成的唯一权威状态切换为每会话一个`InferenceSession`。新增`WebSession`，使用同一把session锁串行化`generate_step()`、text/route/CFG更新，并使用进程级execution lock保护共享eval-mode LDF/BodyVAE执行。
- 新增`WebMotionChunk`与bounded `MotionChunkBuffer`。每个队列元素严格对应一个latent token和四帧world root/body/joints/contact probability；队列满时对producer施加backpressure，不再使用`deque(maxlen=...)`静默丢帧。
- 新增完整`WebRuntime` facade，负责lazy `ModelBundle`、单活跃浏览器会话、force takeover、pause/resume/reset、消费超时回收和worker异常状态。接管时先停止旧worker再更换session；停止超时不会释放活跃所有权或启动第二个共享模型worker。
- `ModelBundle`只保存共享LDF、BodyVAE、text encoder和device，不再保存per-session runtime或旧stream kernel。模型加载守卫从过时的`BLOCKED_ON_BODY_VAE`推进为`BLOCKED_ON_LDF_CHECKPOINT`，准确说明当前只缺正式LDF checkpoint/root statistics/checkpoint loader合同。
- 重写Flask app factory、typed Web配置、请求schema与REST API。route统一为typed XZ、world/relative-to-actor、hold/release；初始route在worker启动前写入session，因此从token 0生效。route更新不再携带horizon/delay/blend或`replace_future`，future constraint tokens只属于session config。
- HTTP传输由20 FPS单帧轮询改为每次长轮询一个四帧chunk；浏览器收到后在本地按20 FPS播放，并通过`session_epoch/token_index/frame_index`拒绝跨reset旧数据。
- 重构前端为`api_client.js`、`renderer.js`、`route_editor.js`和精简`main.js`。保留22-joint Three.js骨架、相机跟随、root trail、Shift拖拽world route、text/route在线更新、pause/resume和force takeover；删除root feedback、smoothing、route delay、旧Root proposal/LDF payload诊断等重复架构控件。
- 修复`skeleton.js`单个trail点时透明度除零以及bone geometry未释放；重写CSS并删除原文件中的重复/失效规则。
- 物理删除`ModelManager`、旧bootstrap、`TrajectoryController`、单帧`FrameBuffer`、generic `GenerationWorker`、重复`GenerationState`和分离`SessionService`。有价值的后台生产、单活跃会话与消费监控能力已迁入新所有权结构，而非简单丢弃。
- `server.sh`改为PID范围内的graceful SIGTERM，超时才终止该PID；Flask退出时显式调用`WebRuntime.shutdown()`，不再无PID时强杀所有`python app.py`进程。
- README、stream配置及模型/VAE/active-window/LDF实现文档同步更新为“Web runtime已接线、正式模型加载等待LDF checkpoint”的实际状态。

改动理由：

- LDF active window、route/text revision和BodyVAE decoder state已经由`InferenceSession`统一拥有；Web继续维护第二套trajectory/pending/root-feedback状态会重新产生旧FloodNet的状态分裂和竞态。
- 新版最小提交事务是`1 token = 4 frames`。按单帧排队和HTTP传输会拆散commit边界，且bounded deque静默淘汰会造成不可诊断的动画跳帧；chunk级backpressure才能保持模型与展示时间轴一致。
- `InferenceSession`明确不是线程安全对象。生成worker、HTTP route/text更新和force takeover必须在Web边界串行化，同时多个会话共享模型时还需要独立于session状态的GPU execution lock。
- window、future horizon和denoise schedule属于模型/session创建合同；让每次route update修改这些值会再次把motion length、condition horizon和runtime控制混为同一接口。

验证：

- `bash -n web_demo/server.sh web_demo/start.sh`通过。
- 所有`web_demo/static/js/*.js`均通过`node --check`。
- `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m compileall -q web_demo`通过。
- Web runtime与migration定向回归：`27 passed`。覆盖严格四帧payload、chunk backpressure无丢弃、text/route更新、relative route一次性world解析、pause/resume/reset、单活跃冲突、初始route从token 0生效、REST chunk协议及真实loader HTTP 503守卫。
- 最终全仓`/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests -q`：`142 passed, 1 warning`；warning仅为测试环境无法初始化NVML。
- `git diff --check`通过；活动Web代码不再引用旧`ModelManager/TrajectoryController/FrameBuffer/SessionService`或root-feedback/delay/replace-future协议。

涉及文件：

- `web_demo/{app.py,config.py,README.md,__init__.py}`
- `web_demo/api/{routes,schemas,responses}.py`
- `web_demo/runtime/{contracts,chunk_buffer,model_bundle,model_loader,web_session,web_runtime}.py`
- 删除`web_demo/{bootstrap,model_manager}.py`、`web_demo/runtime/{frame_buffer,generation_worker,state,trajectory_controller}.py`与`web_demo/services/`
- `web_demo/static/{css/style.css,js/api_client.js,js/main.js,js/renderer.js,js/route_editor.js,js/skeleton.js}`
- `web_demo/templates/index.html`
- `web_demo/server.sh`
- `configs/stream.yaml`
- `tests/{test_web_runtime,test_migration_guards}.py`
- `README.md`与`docs/rearchitecture/{README,01_MODEL_ARCHITECTURE_AND_IO,02_VAE_AND_BODY_REPRESENTATION,03_STREAMING_ACTIVE_WINDOW,06_LDF_IMPLEMENTATION_DESIGN}.md`

尚未完成的后续事项：

- 正式LDF训练、冻结root statistics、UMT5 text encoder加载与LDF checkpoint schema尚未完成，因此真实`POST /api/sessions`当前按设计返回`BLOCKED_ON_LDF_CHECKPOINT` HTTP 503；本轮未伪装真实动作生成已经可用。
- 正式checkpoint冻结后需要实现`load_model_bundle()`，校验LDF/VAE FPS、latent维度、statistics与checkpoint identity，再进行真实GPU长时stream、route/text突变质量和浏览器人工可视化测试。
- 跨进程snapshot、生成视频持久化和多活跃用户调度不在第一版Web runtime范围内；当前明确采用单活跃会话与进程内状态。

## 2026-07-15 · LDF固定噪声Active Band与Detached Self-Forcing内核

改动类型：LDF训练数据边界 / flow代数 / self-forcing策略 / statistics / 配置 / 测试与文档

实际改动内容：

- 将`LDFWindowCollator`收敛为`LDFSpanCollator`。它只选择batch共享的40–200帧physical source span、各样本独立crop、真实causal VAE context、previous root、source token坐标、cold-start标志和token prompt timeline，不再做translation rebase、H/A/frontier划分、noise或rollout。
- true cold start按batch以10%概率采样并强制`H=0/source_start=0/context=0/previous invalid`；continuation要求`H>=1`且允许mid-clip crop。random yaw仍同步作用于root/body/context/previous root。
- 新增`flow.py`，以一个连续公式构造完整span的history beta=0、5-token active beta和frontier beta=1；新增fixed absolute-token noise混合、`v*=x0-epsilon`、self-forcing低误差clean recovery与full-gradient auxiliary recovery的明确分离接口。
- 新增`batch.py`，始终构造固定S的`LDFInput`。history与active+pure-noise frontier共同组成完整有效attention前缀，`generation_mask`覆盖active和frontier，独立`loss_mask`只覆盖5-token active band；timeline使用source absolute token坐标，RoPE以当前active首token为0。
- 新增`self_forcing.py`的immutable`LDFWindowPlan`与mutable`SelfForcingState`。每个rollout一次性冻结per-sample phase、root/body Gaussian noise和初始H translation anchor；前K-1步保持model train mode但关闭autograd，只用projected clean root与`x_beta+beta*v` body替换最左active token，最终一步保留梯度。
- 实现K=2/3/5（0%/40%/70%）curriculum和可配置K=1 replay，默认概率为20%/10%/10%；K=1 teacher baseline和pure-SF消融均可由调用方显式选择。
- `tools/compute_ldf_root_stats.py`改为复用10% true cold start、随机初始H和rollout固定anchor分布；CLI最短span改为40帧，但不会自动覆盖已有root statistics资产。
- `configs/ldf.yaml`加入cold-start与self-forcing实验默认值，最短span改为40帧。`train_ldf.py`继续fail-fast，但错误信息已推进为“active-band kernel ready，text/condition与Lightning optimization未接线”。
- 数据、训练方法、训练配置、LDF实现和README文档同步记录physical span、fixed noise、active/frontier、固定坐标frame与剩余trainer边界。

改动理由：

- 同一absolute token若在K步间重采Gaussian noise，会跳到另一条diffusion path；固定noise后active右移只改变beta，才能匹配persistent noisy frontier。
- pure-noise frontier属于当前LDF motion state并必须参与self-attention，不能与span外condition-only future goal混为一类，也不能被generation validity mask裁掉。
- mid-clip `H=0`会让EMA causal VAE target隐含不可见历史，并让Root/Body boundary信息不对称；第一版cold start因此只表示真实sample起点。
- self-forcing只需要模拟已提交history误差，无需把完整runtime commit/roll/rebase状态机复制到训练；固定初始anchor也避免把OriginEpoch事务混入K-step kernel。
- `x_beta+beta*v_pred`把最左active token的replacement误差从`delta`缩小到`beta*delta`，比`v_pred+epsilon`更适合低beta stage boundary。

验证：

- 相关Python文件`py_compile`通过。
- Dataset/span/flow/self-forcing/VAE bridge/statistics/migration定向测试：`41 passed`。
- 真实集成覆盖`HumanML3DDataset fixture -> LDFSpanCollator -> frozen EMA VAE -> K-step kernel -> loss.backward()`；验证VAE无梯度、前序replacement无`grad_fn`、最终Root/Body Transformer均有梯度。
- 全仓`/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests -q`：`155 passed`。
- `git diff --check`通过；`rg`确认无活动`LDFWindowCollator`引用，且训练contract明确以`generation_mask = ~history_mask`包含frontier。

涉及文件：

- `utils/training/ldf/{data,flow,batch,self_forcing,__init__}.py`
- `utils/training/ldf/lightning_module.py`
- `tools/compute_ldf_root_stats.py`
- `configs/ldf.yaml`
- `train_ldf.py`
- `tests/{test_ldf_data,test_ldf_self_forcing,test_ldf_training,test_ldf_statistics,test_migration_guards}.py`
- `README.md`
- `docs/rearchitecture/{02_DATA_PIPELINE,04_TRAINING_METHOD,05_TRAINING_CONFIG,06_LDF_IMPLEMENTATION_DESIGN}.md`
- `docs/DEVELOPMENT_LOG.md`

尚未完成的后续事项：

- `LDFLightningModule._step()`与`train_ldf.py`守卫仍保留；下一阶段需要接入UMT5/token prompt embedding、span内root/body condition与dropout、root/body loss权重、optimizer/EMA和validation loop。
- 当前root statistics工具语义已正确，但正式`root_stats.npz`尚未重算或覆盖；启动真实训练前必须显式生成并校验统计资产。
- span外`F_goal`、空间约束采样、decoded/FK/contact auxiliary loss与history corruption不在本轮范围。
## 2026-07-15 · FloodDiffusion局部文本条件与LDF teacher-training接线

改动类型：LDF文本协议 / Transformer attention / 三角可见窗口 / future-root packing / Lightning训练 / 配置与测试

实际改动内容：

- `LDFSpanCollator`把Dataset的caption区间编译为固定长度`prompt_timeline[B][S]`。HumanML3D从覆盖当前span的caption备选中选择一条并在所有token重复；BABEL按四帧对齐区间为每个token分配prompt，并对未对齐或重叠区间fail-fast。
- LDF data loader新增任务局部`MinimumFrameDataset`，在组batch前排除短于最小source span的样本。实际扫描`BABEL_motion`的14087个artifact发现2282个少于40帧，因此不能依赖collator在长训中随机遇到后才失败。
- 修复true cold-start window plan在未显式指定H时仍可能随机采到`H>0`的问题；cold-start现在无条件固定`H=0`，continuation才采样正history。
- 删除Transformer中“先把每条T5 sequence求均值，再让所有motion token共同注意整条prompt timeline”的实现。Root/Body现在使用严格token-aligned cross-attention：每个motion query只读取自己的T5 sequence；sample-aligned空文本仍可广播用于CFG。
- 文本padding改为batch内实际最大长度（上限128），重复prompt的T5 tensor在projection前按identity去重；runtime `TextEmbeddingCache`对同一device上的重复文本复用同一个tensor。
- 三角scheduler的persistent state继续保存固定pure-noise frontier，但`LDFInput.generation_mask`只标记本步可见/更新的active前缀。尚未触及的beta=1 tail不进入self-attention，也不会因预先编译future prompt而泄漏条件。
- Root Stage将`[visible motion prefix | valid future-root tokens]`逐样本紧凑打包；future token不再要求整个固定motion window都有效，文本query只映射回visible motion，future-root query不读取文本。
- `time_embedding_scale`正式作用于beta sinusoidal embedding；LDF默认T5长度由512收敛为论文设置的128。
- `LDFLightningModule`完成冻结EMA VAE在线`tokenize_window()`、冻结UMT5 batch唯一文本编码、sample级text dropout、固定噪声window plan、可选detached self-forcing和root/body flow-v loss接线。VAE与T5不进入LDF optimizer、EMA或checkpoint。
- `train_ldf.py`由迁移守卫改为薄Lightning入口，并在root/motion/latent statistics、VAE/T5 checkpoint或tokenizer缺失时fail-fast。由于现有root statistics早于当前fixed-span anchor sampler，HumanML和mixed配置均保持`root_statistics_required`状态；只有显式重算并切换为`training_ready`后入口才允许长训。新增`configs/ldf_multi.yaml`描述HumanML+BABEL从头训练设置。
- `tools/compute_ldf_root_stats.py`新增`--config`入口，可直接复用HumanML或HumanML+BABEL配置的数据mixture、短样本过滤、span、chunk和cold-start参数，并默认写入该配置声明的`root_stats_path`。
- 公共`create_window_condition()`可以把sample-major absolute token text timeline切到当前window并用对应null prompt右填充。

改动理由：

- FloodDiffusion的流式局部性要求当前motion只依赖当前可见motion前缀和对应时刻的prompt。把future caption合并成共享文本序列，或用pure-noise tail扩大有效`seq_len`，都会产生训练时不明显、在线prompt切换时暴露的条件泄漏。
- future-root是只读condition token，不是latent motion。把它紧接visible prefix打包，可以同时保留non-causal Root Transformer和明确token contract，而不重新引入FlexTraj专用mask。
- HumanML的重复caption与BABEL的分段caption最终都归一成同一个`B*T`模型接口，使Root/Body、训练与runtime不需要数据集特判。
- VAE deterministic μ、text encoding、diffusion noise和loss分别保留在各自所有者，LDF checkpoint只携带实际需要优化和部署的LDF权重。

验证：

- 新增token-aligned cross-attention隔离测试：只修改token 1 prompt不会改变token 0的cross-attention输出，非motion query输出严格为零。
- 新增HumanML prompt重复、BABEL区间编译、pure-noise frontier隐藏、partial-visible future-root紧凑打包和完整在线VAE→LDF loss/backward smoke test。
- 真实`configs/ldf_multi.yaml` train Dataset构造通过：保留31184条满足40帧source span的样本，在组batch前排除2498条短样本；其中BABEL短样本为2282条。
- 相关Python文件`py_compile`通过，`git diff --check`通过。
- 全仓`/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests -q`：`159 passed, 1 warning`；warning仅为测试环境无法初始化NVML。

涉及文件：

- `models/diffusion_forcing_wan.py`
- `models/tools/wan_model.py`
- `utils/conditions/ldf.py`
- `utils/inference/text.py`
- `utils/training/ldf/{data,batch,self_forcing,lightning_module,__init__}.py`
- `train_ldf.py`
- `tools/compute_ldf_root_stats.py`
- `configs/{ldf,ldf_multi}.yaml`
- `tests/{test_ldf_data,test_ldf_forward,test_ldf_stream,test_ldf_self_forcing,test_ldf_training,test_wan_model,test_migration_guards}.py`
- `README.md`
- `docs/rearchitecture/{01_MODEL_ARCHITECTURE_AND_IO,02_DATA_PIPELINE,03_STREAMING_ACTIVE_WINDOW,04_TRAINING_METHOD,05_TRAINING_CONFIG,06_LDF_IMPLEMENTATION_DESIGN}.md`

尚未完成的后续事项：

- 当前teacher baseline只训练文本条件与结构化root/body生成；span内root observation、span外future-root constraint的训练采样/dropout和constraint-CFG校准仍需单独设计，不应由本轮静默假设。
- `configs/ldf_multi.yaml`需要先基于实际HumanML+BABEL训练mixture生成`HumanML3D_BABEL_root_stats.npz`；本轮没有伪造该statistics，也没有启动正式长训。
- Web继续等待首个正式LDF checkpoint与loader合同；本轮未解除`BLOCKED_ON_LDF_CHECKPOINT`。

## 2026-07-15 · LDF文本、验证、恢复训练与self-forcing可靠性修复

改动类型：LDF数据编译 / 文本特征加载 / validation / self-forcing curriculum / checkpoint resume / 可复现采样 / 性能与测试

实际改动内容：

- BABEL prompt timeline不再要求原始caption边界四帧对齐。每个四帧motion token按实际frame overlap选择caption；重叠相同时优先更短区间，再按原始稳定顺序决定。现有同区间多caption仍保持train随机、validation取首条的语义。
- 新增`LengthBucketBatchSampler`，按clip长度分桶后组成训练batch，避免一个短clip把同batch全部长clip限制为短span。sampler通过epoch/sample seed把crop、caption选择和yaw增强变成可恢复的确定性采样；Lightning侧的phase、Gaussian noise、text dropout和teacher replay由`global_step + rank`显式generator控制。
- Validation由单一cold-start loader拆为固定`teacher_cold`和`teacher_continuation` probe；启用self-forcing时增加固定self-forcing probe。每个probe显式固定H、K和seed，因此phase与root/body noise可以重复，指标按probe分别记录。
- self-forcing curriculum改为相对fine-tune阶段的`phase_start_step/phase_steps`进度，不再用包含teacher baseline的`global_step / max_steps`。
- 新增轻量`TextEmbeddingLookup`。正式LDF训练只加载`tools/pretokenize_t5_text.py`生成的CPU caption table，不再把约11GB UMT5-XXL放到LDF训练GPU；预编码和加载两端都检查shape与finite，缺失caption立即失败。
- LDF checkpoint新增训练边界metadata；resume在覆盖当前模型前比较root/local-root buffers、VAE physical/latent statistics以及VAE/statistics/text配置路径。旧checkpoint缺少metadata时只在四个LDF统计buffer完全一致后允许加载并给出warning。
- `LDFCondition.validate()`按tensor identity去重finite扫描，保留逐项shape/dtype检查，避免HumanML重复prompt和self-forcing多步反复扫描同一个大tensor。
- 修复训练kernel与既有设计文档的矛盾：pure-noise frontier继续保存在固定`HybridMotion`和fixed noise中，但`generation_mask`只暴露当前active band。当前forward只看到`history + active`，不可见future prompt不能通过non-causal self-attention提前传播。
- 文档将文本语义准确表述为`direct token-aligned cross-attention`：每个motion query直接读取自身prompt，但已注入的文本信息可在后续层通过可见motion self-attention参与动作过渡，不宣称严格文本隔离。
- HumanML和HumanML+BABEL配置增加文本embedding路径、length bucket、确定性validation和self-forcing phase字段；训练入口要求预编码文本文件，不再要求在线文本编码器进入训练进程。

改动理由：

- 任意frame文本标注属于Dataset事实，四帧token所有权应由LDF编译器根据重叠确定；即使当前14087个`BABEL_motion`文本文件全部恰好对齐，也不应把该偶然数据性质写成会崩溃的公共合同。
- 随机validation无法可靠比较checkpoint，只测cold start也无法代表流式continuation；固定多probe将覆盖范围和数值可比性同时写死。
- 完整resume会在constructor校验后覆盖模型buffers，因此统计比较必须发生在`load_state_dict()`之前。VAE latent坐标和文本特征坐标同样属于恢复训练边界，但不应重新进入LDF模型计算代码。
- UMT5预编码不改变逐tokencross-attention语义，只移除固定语料上的重复11GB encoder加载和每batch重复计算。
- self-forcing fine-tune从baseline checkpoint继续global step时，curriculum必须相对fine-tune阶段推进，否则会跳过预定的K=2阶段。

验证：

- 新增任意frame BABEL overlap、tie-break、length bucket、seeded augmentation、DataLoader seeded index、多validation loader、phase-relative progress、确定性noise、resume统计保护、VAE统计保护和文本lookup fail-fast测试。
- LDF定向测试：`74 passed`。
- 全仓`/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests -q`：`170 passed`。
- 修改Python文件`py_compile`通过。
- `git diff --check`通过。

涉及文件：

- `utils/training/ldf/{data,text,batch,self_forcing,lightning_module,__init__}.py`
- `utils/conditions/ldf.py`
- `tools/pretokenize_t5_text.py`
- `train_ldf.py`
- `configs/{ldf,ldf_multi}.yaml`
- `tests/{test_ldf_data,test_ldf_text,test_ldf_training,test_ldf_self_forcing,test_migration_guards}.py`
- `docs/rearchitecture/{01_MODEL_ARCHITECTURE_AND_IO,02_DATA_PIPELINE,03_STREAMING_ACTIVE_WINDOW,04_TRAINING_METHOD,05_TRAINING_CONFIG,06_LDF_IMPLEMENTATION_DESIGN}.md`
- `docs/DEVELOPMENT_LOG.md`

尚未完成的后续事项：

- `root_stats.npz`和两个T5 embedding table尚未在正式数据目录生成；配置继续保持`root_statistics_required`，因此本轮没有启动真实LDF长训。
- span内root observation、span外future-root constraint训练采样/dropout及constraint-CFG校准仍需单独设计。
- Web继续等待首个正式LDF checkpoint；本轮未解除`BLOCKED_ON_LDF_CHECKPOINT`。

## 2026-07-16 · LDF数据恢复、continuation验证与文本表身份修复

改动类型：LDF DataLoader resume / deterministic augmentation / validation coverage / text embedding artifact / checkpoint contract / 配置与测试

实际改动内容：

- `LengthBucketBatchSampler`不再在`__iter__()`内隐式递增epoch；Lightning通过既有`batch_sampler.sampler.set_epoch()`成为唯一epoch所有者。同一epoch重复构造iterator会得到相同bucket顺序和sample augmentation seed。
- 新增单卡`ResumableDataLoader`，向Lightning `CombinedLoader`保存`epoch + yielded_batches`。游标以真正交给训练循环的batch为准，不读取会被multi-worker prefetch提前推进的sampler内部位置；resume时重建相同epoch顺序并跳过已消费batch，随后从精确next batch继续。
- batch augmentation seed从可碰撞的加权和改为有序64-bit混合；batch内seed排列和内容都会改变crop、caption、yaw及后续随机流的batch seed。
- 修复正式validation中短于`max_frames`的clip会把整段作为span、导致所谓continuation仍从token 0开始的问题。新增`validation.continuation_span_frames=40`，continuation/self-forcing使用至少多一个真实token的独立数据视图，并按稳定sample index在一个loader中轮换early/middle/late source位置。
- 离线T5预编码表新增由encoder/tokenizer身份、caption、shape/dtype和embedding内容共同生成的`content_id`。`TextEmbeddingLookup`通过mmap加载时只检查metadata、shape和dtype，finite检查延迟到caption第一次lookup，避免启动时扫描整个表；LDF checkpoint保存并在resume前比较`content_id`，同路径内容被替换也会失败。
- pure-noise frontier可见性保持不变：persistent state继续保存frontier，Transformer仍只读取history+active；该项与原版FloodDiffusion prefix输入一致，不作为bug修改。

改动理由：

- sampler epoch本身已经由当前Lightning恢复，真正缺口是epoch中途的batch cursor；将状态放在Lightning实际会保存的DataLoader边界，才能避免只实现一个无人调用的sampler `state_dict()`。
- validation必须保证source start大于零，才能测到真实continuation；仅把collator标记为`cold_start=False`并不能在span占满整段clip时创造历史。
- 文本表路径相同不代表内容相同，但每次训练启动重新hash数GB文件会破坏mmap的价值；离线生成一次内容身份、训练时直接比较是更明确且更便宜的边界。
- mmap tensor的`detach()`不会复制完整embedding storage，实际启动成本来自逐tensor eager finite扫描，因此只移动检查时机，不引入offset-table重构。

验证：

- 新增sampler epoch所有权、旧加权和碰撞、Lightning stateful loader识别、精确next-batch恢复、新epoch重置、early/middle/late continuation、文本表缺失身份、lazy non-finite、内容变化和同路径resume拒绝测试。
- 全仓`MPLCONFIGDIR=/tmp/matplotlib /home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests -q`：`177 passed, 1 warning`；warning仅为测试环境无法初始化NVML。
- 修改Python文件`py_compile`通过，`git diff --check`通过。

涉及文件：

- `utils/training/ldf/{data,text,lightning_module}.py`
- `tools/pretokenize_t5_text.py`
- `configs/{ldf,ldf_multi}.yaml`
- `tests/{test_ldf_data,test_ldf_text,test_ldf_training,test_migration_guards}.py`
- `docs/rearchitecture/{02_DATA_PIPELINE,04_TRAINING_METHOD,05_TRAINING_CONFIG}.md`
- `docs/DEVELOPMENT_LOG.md`

尚未完成的后续事项：

- 现有旧格式T5 embedding文件没有`content_id`，正式LDF训练前必须用更新后的`tools/pretokenize_t5_text.py`重新生成；本轮没有在正式数据目录运行UMT5或修改外部数据。
- 当前正式训练配置固定单卡；若未来重新启用DDP，需要设计并验证每个rank独立的loader cursor，不能直接把单卡恢复状态当成多卡合同。
- frontier是否额外暴露一个`beta=1`边界token可作为单独消融，但全pure-noise frontier不会改成默认可见。

## 2026-07-16 · LDF loss职责迁移

改动类型：训练代码结构整理 / import迁移 / 文档与测试

实际改动内容：

- 新增`utils/training/ldf/losses.py`，集中保存`compute_velocity_loss()`。
- `batch.py`只保留physical anchor、`LDFTrainingStep`、target和mask构造，不再负责prediction reduction。
- Lightning module、包级导出与LDF训练/self-forcing测试统一从`losses.py`导入loss函数。
- root/body normalized flow-v MSE、active-band mask、配置权重和日志键完全不变，没有新增decoded/FK/contact/skate辅助项。

改动理由：

- batch合同与loss reduction属于不同职责；在正式训练前拆开可以让后续loss消融只修改`losses.py`，不污染noise/window/input构造。
- 本轮只整理所有权，不把文件重构伪装成训练目标变化。

验证：

- LDF training/self-forcing定向测试：`19 passed`。
- 全仓`MPLCONFIGDIR=/tmp/matplotlib /home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests -q`：`177 passed, 1 warning`；warning仅为测试环境无法初始化NVML。
- 修改Python文件`py_compile`通过，`git diff --check`通过。

涉及文件：

- `utils/training/ldf/{batch,losses,lightning_module,__init__}.py`
- `tests/{test_ldf_training,test_ldf_self_forcing}.py`
- `docs/rearchitecture/{04_TRAINING_METHOD,06_LDF_IMPLEMENTATION_DESIGN}.md`
- `docs/DEVELOPMENT_LOG.md`

尚未完成的后续事项：

- 当前loss仍是首轮teacher baseline的root/body flow-v MSE；是否加入任何辅助loss必须基于baseline结果单独消融，本轮未提前引入。

## 2026-07-16 · VAE重建评测按模型隔离并评估164325 checkpoint

改动类型：VAE eval输出协议 / 配置 / 测试 / 正式重建评测

实际改动内容：

- VAE stream与rolling配置切换到`20260715_164325_vae_body265/last.ckpt`的EMA权重，并显式声明稳定模型身份`vae_body265_20260715_164325`。
- 评测输出从`output/<task>/<dataset>/...`改为`output/<task>/<dataset>/<model_id>/...`；dataset manifest、summary、逐样本指标、motion和视频均按checkpoint隔离，不再被后续模型评测覆盖。
- task聚合summary改为`output/<task>/summaries/<model_id>.json`，并在task/dataset summary和checkpoint metadata中记录`model_id`。
- `model_id`必须是非空单层目录名，禁止通过路径分隔符逃出预期输出层级；README和布局测试同步更新。
- 在GPU 2实际运行stream与`history_tokens=10, commit_tokens=1` rolling：HumanML3D和BABEL各取validation前10个样本，每个task/dataset均生成10个原始视频、10个重建视频、20个motion NPZ和10个逐样本指标。评测产物继续由`.gitignore`排除。

改动理由：

- VAE checkpoint对比必须保留各模型独立的可视化、数值指标与manifest；仅替换配置中的checkpoint路径而复用dataset目录会静默覆盖旧实验，无法进行后续HumanML-only与multi VAE横向比较。
- 显式`model_id`避免checkpoint被移动或命名顺序不一致时改变已发布实验目录，同时让配置承担模型身份所有权。

验证：

- `python -m py_compile eval/vae/evaluate_reconstruction.py`通过。
- 定向测试`tests/test_vae_eval.py tests/test_multi_dataset.py`：`12 passed`。
- 全仓`MPLCONFIGDIR=/tmp/matplotlib /home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests -q`：`177 passed, 1 warning`；warning仅为测试沙箱无法初始化NVML。
- `git diff --check`通过。
- 正式stream评测通过offline/token-stream parity；mean max-abs为HumanML3D `3.35e-5`、BABEL `3.03e-5`。HumanML3D平均position MAE为`0.001674 m`、rotation geodesic为`1.4454 deg`；BABEL分别为`0.003944 m`和`2.9407 deg`。
- 正式rolling评测通过每个有限窗口的offline/cache parity；mean max-abs为HumanML3D `3.20e-5`、BABEL `3.05e-5`。相对persistent stream，HumanML3D平均position差为`1.24e-5 m`、rotation差为`0.01136 deg`；BABEL分别为`2.35e-6 m`和`0.00645 deg`。

涉及文件：

- `eval/vae/evaluate_reconstruction.py`
- `eval/vae/{stream,rolling}.yaml`
- `eval/vae/README.md`
- `tests/test_vae_eval.py`
- `tests/test_multi_dataset.py`
- `docs/DEVELOPMENT_LOG.md`

尚未完成的后续事项：

- 当前164325 checkpoint只在HumanML3D训练；BABEL前10个样本的平均重建误差约为HumanML3D的两倍，且`11995_5`明显更困难。是否训练multi VAE应在扩大分层validation样本后决定，不能只根据这20个可视化样本作最终结论。
- 本轮没有删除旧版未分模型的本地eval产物；它们不受Git跟踪，也不会影响新的model-scoped目录。若需要释放空间，可在确认旧结果无保留价值后单独清理。

## 2026-07-16 · 164325 VAE latent statistics与LDF配置切换

改动类型：正式数据扫描 / VAE-LDF checkpoint边界 / LDF配置

实际改动内容：

- 使用`20260715_164325_vae_body265/last.ckpt`的EMA encoder和deterministic posterior `mu`重新扫描HumanML3D完整train split，共处理`23240/23240`个样本；按当前统计工具的固定`yaw_seed=0`顺序采样全局yaw增强。
- 将128D latent mean/std原子写入`/data1/yuankai/text2Motion/Floodcontrol/vae/20260715_164325_vae_body265/latent_stats.npz`。当前协议只保存`mean/std`，checkpoint与statistics配对由配置所有，与`02_VAE_AND_BODY_REPRESENTATION.md`一致。
- `configs/ldf.yaml`和`configs/ldf_multi.yaml`同时从022912 VAE切换到164325的`last.ckpt`与新生成的`latent_stats.npz`；两条LDF训练线现在共享同一HumanML-trained VAE latent坐标系。

改动理由：

- latent statistics属于具体冻结EMA encoder的输出分布，不能继续沿用022912模型生成的mean/std；checkpoint和latent normalization必须作为同一不可拆分配置边界迁移。
- `ldf_multi`本轮按明确要求继续使用HumanML训练得到的共享VAE，因此也必须使用该VAE在HumanML train上生成的statistics，不能混用旧模型或另一个数据分布的统计量。

验证：

- 正式扫描成功完成`23240/23240`个HumanML train样本并写入目标NPZ。
- 新NPZ SHA256为`6a30d2f41787ad5548fe91df52d62add12ab80319d75f863986d006c067349d2`；`mean/std`形状均为`[128]`、dtype为float32、全部finite，std全部为正且范围为`[1.05748, 2.67038]`。
- 分别从`configs/ldf.yaml`和`configs/ldf_multi.yaml`构造`BodyVAE`并加载164325 `last.ckpt`的EMA权重成功；两者均确认`latent_statistics_ready=True`且模型处于冻结eval状态。
- 全仓`MPLCONFIGDIR=/tmp/matplotlib /home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests -q`：`177 passed, 1 warning`；warning仅为测试沙箱无法初始化NVML。
- `git diff --check`通过。

涉及文件：

- `configs/ldf.yaml`
- `configs/ldf_multi.yaml`
- `docs/DEVELOPMENT_LOG.md`
- 外部生成产物：`/data1/yuankai/text2Motion/Floodcontrol/vae/20260715_164325_vae_body265/latent_stats.npz`

尚未完成的后续事项：

- `ldf_multi`当前有意使用HumanML latent statistics；若以后训练真正的HumanML+BABEL multi VAE，必须用那个新EMA encoder重新扫描约定的数据mixture并再次整体切换checkpoint/statistics，不能复用本文件。
- 本轮只切换VAE及latent statistics，没有启动正式LDF训练，也没有改动root statistics或T5 embedding artifacts。

## 2026-07-16 · 生成HumanML3D与BABEL训练所需T5文本表

改动类型：离线文本预编码工具修复 / 正式数据产物生成 / 配置接线验证

实际改动内容：

- 修复`tools/pretokenize_t5_text.py`仍从旧版`model.params`读取UMT5 checkpoint、tokenizer和文本长度的问题；当前优先读取正式配置的`text_encoder.*`，仅为旧配置保留回退读取。
- 预编码启动前显式检查checkpoint文件、tokenizer目录，并要求`text_encoder.text_len`与LDF模型的`model.params.text_len`一致，避免缺失路径继续运行到模型加载阶段才失败。
- 使用GPU 2生成HumanML3D文本表，共47,420条caption（包含空文本），保存到`HumanML3D_motion/t5_text_embeddings.pt`。
- 以HumanML3D表为基础，通过`--reuse-existing`补充3,632条BABEL caption，生成供multi配置使用的HumanML3D+BABEL联合表，共51,052条，保存到`HumanML3D_BABEL_t5_text_embeddings.pt`；没有额外制造当前训练配置不消费的BABEL-only表。
- 两个正式配置解析后的`text_embeddings_path`均已指向上述实际文件。

改动理由：

- 当前LDF训练只消费HumanML3D单数据表或HumanML3D+BABEL联合表，按配置边界生成两份产物可以避免重复存储HumanML embedding和无用途的第三份表。
- 预编码脚本若继续读取旧字段会得到`None`路径；在长时间UMT5编码前完成路径和长度fail-fast，能防止生成错误或不兼容的embedding表。

验证：

- HumanML3D表通过正式`TextEmbeddingLookup(expected_dim=4096, expected_text_len=128)`加载：47,420条，空文本shape为`[1,4096]`、dtype为`bfloat16`，`content_id=c517945625635128a03b0bf16241df0dc73999c1`，文件大小6,927,794,453 bytes。
- 联合表通过相同正式loader加载：51,052条，空文本shape为`[1,4096]`、dtype为`bfloat16`，`content_id=1ba8101404f15b2388dc90e3b4474f51ea8b1bcf`，文件大小7,080,869,033 bytes。
- `configs/ldf.yaml`和`configs/ldf_multi.yaml`经项目`load_config()`解析后，目标文件均存在且大小与上述结果一致。
- `tests/test_ldf_text.py tests/test_migration_guards.py`：`27 passed`。
- `python -m py_compile tools/pretokenize_t5_text.py`通过；`git diff --check`通过。

涉及文件：

- `tools/pretokenize_t5_text.py`
- `docs/DEVELOPMENT_LOG.md`
- 外部数据产物：`/data1/yuankai/text2Motion/FloodDiffusion/raw_data/HumanML3D_motion/t5_text_embeddings.pt`
- 外部数据产物：`/data1/yuankai/text2Motion/FloodDiffusion/raw_data/HumanML3D_BABEL_t5_text_embeddings.pt`

尚未完成的后续事项：

- 本轮只补齐文本embedding表；正式LDF启动仍需检查并生成配置要求的root statistics等剩余训练前产物。
- 本轮未运行全仓pytest；工具修复由两次真实UMT5生成、正式loader加载和27项相关测试覆盖。

## 2026-07-16 · 生成canonical HumanML3D LDF root statistics

改动类型：正式训练数据产物 / normalization协议冻结 / LDF配置与文档

实际改动内容：

- 使用当前`tools.compute_ldf_root_stats`的fixed-span/anchor采样器遍历22,418个满足最短长度的HumanML3D训练样本，每个样本按固定seed采一个40–200帧四帧对齐窗口，并复用10% cold start、随机初始H、history末帧XZ anchor和均匀random-yaw训练分布。
- 先生成并验证临时NPZ，再通过同目录pending文件原子替换`HumanML3D_motion/root_stats.npz`，没有在统计过程中直接覆盖正式文件。
- 冻结HumanML3D root statistics为canonical normalization：`ldf.yaml`和`ldf_multi.yaml`共同引用同一文件。加入BABEL时只切换Dataset和联合T5表，不再重算或切换root尺度，从而保持从HumanML checkpoint继续训练时的数值语义。
- 两份正式配置的VAE checkpoint、motion/latent/root statistics和T5表前置路径均通过训练入口校验后，将状态从`root_statistics_required`切换为`training_ready`；本轮没有实际启动训练。
- 数据与训练配置文档同步移除“HumanML+BABEL单独计算root stats”的旧协议，并增加配置测试锁定共享路径。

改动理由：

- global root的统计分布必须匹配LDF实际fixed-span translation anchor与random-yaw处理；旧文件早于当前采样协议，不能仅凭同名文件存在继续使用。
- 原版FloodDiffusion让HumanML与BABEL共享同一VAE latent尺度；Floodcontrol同样保持一套canonical root/local/latent normalization，避免数据mixture或fine-tune阶段改变模型输入输出含义。

验证：

- 正式文件只含`root_mean/root_std [5] float32`，所有值finite且std严格为正；SHA256为`97460eab919e02569502ce84cabce1670a3677dd9031e87e4aa36449132a72cd`。
- `root_mean=[0.0006808192, 0.91795015, -0.0039776308, 0.0038573092, 0.0005940247]`。
- `root_std=[0.56822836, 0.15350504, 0.56570762, 0.70656031, 0.70764202]`。
- 将两份配置以`training_ready`送入`train_ldf._validate_training_config()`，全部前置文件检查通过且解析到同一root stats路径。
- `tests/test_ldf_statistics.py tests/test_migration_guards.py`：`25 passed`。
- `git diff --check`通过。

涉及文件：

- `configs/{ldf,ldf_multi}.yaml`
- `docs/rearchitecture/{02_DATA_PIPELINE,05_TRAINING_CONFIG}.md`
- `tests/test_migration_guards.py`
- `docs/DEVELOPMENT_LOG.md`
- 外部数据产物：`/data1/yuankai/text2Motion/FloodDiffusion/raw_data/HumanML3D_motion/root_stats.npz`

尚未完成的后续事项：

- 本轮未启动HumanML或HumanML+BABEL正式LDF训练；配置现已解除statistics迁移守卫，可在最终训练超参数和运行资源确认后启动。
- 需要在首轮BABEL联合训练中监控normalized root的极值与标准差，确认canonical HumanML尺度没有产生异常outlier；监控结果不应通过训练中途切换stats来修复。
- 本轮未运行全仓pytest；统计/迁移定向测试、两份真实配置前置校验与产物数值校验已经完成。

## 2026-07-16 · LDF active/future XZ轨迹条件训练接线

改动类型：核心训练协议修复 / Root Stage轨迹控制 / constraint CFG训练分布

实际改动内容：

- 新增`utils/training/ldf/conditioning.py`，从translation-anchored、random-yaw增强后的normalized clean root编译训练条件。当前active band只暴露root feature 0/2，即每帧XZ；root y与heading不进入当前轨迹条件。
- 将active band之后最多`future_root_lookahead_tokens=20`个真实token编译为future XZ condition，携带absolute timeline position并按每个sample的有效span自然缩短。短样本尾部不使用假零未来。
- `LDFLightningModule`现在对每个sample独立采样text dropout和constraint dropout；constraint决定在一个self-forcing rollout内保持不变，所有step复用同一决定。正式配置的两者均为0.1。
- Root Stage继续使用branch-local只读masked input view；persistent noisy state、clean prediction和history不被hard replace。Body Stage不直接读取raw active/future XZ，只读取唯一Root Stage结果派生的local root与heading。
- future root value在进入`future_projection`前按feature mask清零，阻止未观测的root y/cos/sin通过数值张量泄漏。`LDFCondition`同时要求`future_valid_mask`与实际含约束的future token严格一致。
- `train_ldf.py`新增正式训练守卫：text/constraint dropout必须位于`[0,1]`，`future_root_lookahead_tokens`必须为正；没有XZ lookahead的text-only配置不能标记为`training_ready`。
- 两份LDF配置均加入`constraint_dropout_probability: 0.1`与`future_root_lookahead_tokens: 20`。README及训练方法、训练配置、模型实现文档同步记录新的真实训练路径。

改动理由：

- 将root提升为LDF内部结构化生成变量，只解决了“谁生成root”与“Body怎样依赖root”，并不会自动让模型学会服从用户XZ轨迹。没有把active/future trajectory condition送入训练，Root Transformer仍只是text-to-root生成器，不是轨迹控制器。
- 此前实现把model-core、condition schema和training bridge拆成阶段后，错误地把“模型已经能接收constraint”当成“轨迹控制已经完成”，并在root statistics就绪后过早将配置标成`training_ready`。验收条件也只覆盖了condition dataclass/CFG公式，没有要求“训练batch真实出现constraint”和“改变XZ会改变Root预测”。这是里程碑定义与验收标准的架构失误，不是ARDY方法不需要该条件。
- Floodcontrol参考ARDY的核心目标正是让root/trajectory成为模型内部生成与条件建模的一部分，因此active XZ、future lookahead和独立constraint dropout必须属于首个teacher baseline，而不能推迟到可选的后续finetune。

验证：

- condition测试确认active mask只覆盖当前active band的XZ，future mask只覆盖后续lookahead的XZ，constraint dropout按sample生效且validation不丢条件。
- self-forcing测试确认active/future范围随step同步右移，constraint keep/drop决定在rollout内保持不变。
- 模型回归分别确认改变active XZ和改变future XZ都会改变Root Stage预测；未masked的future y/heading数值不会进入projection。
- 训练bridge smoke test确认真实`_step()`同时构造active XZ与future XZ，loss可反传LDF且冻结VAE无梯度。
- 两份正式配置均通过训练入口前置校验。
- `MPLCONFIGDIR=/tmp/matplotlib /home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests -q`：`186 passed, 1 warning`；warning仅为测试环境无法初始化NVML。
- 全仓Python文件`py_compile`通过，`git diff --check`通过。

涉及文件：

- `utils/training/ldf/{conditioning,lightning_module,__init__}.py`
- `utils/conditions/ldf.py`
- `models/diffusion_forcing_wan.py`
- `train_ldf.py`
- `configs/{ldf,ldf_multi}.yaml`
- `tests/{test_ldf_training,test_ldf_self_forcing,test_ldf_forward,test_ldf_conditions,test_migration_guards}.py`
- `README.md`
- `docs/rearchitecture/{04_TRAINING_METHOD,05_TRAINING_CONFIG,06_LDF_IMPLEMENTATION_DESIGN}.md`
- `docs/DEVELOPMENT_LOG.md`

尚未完成的后续事项：

- 本轮完成的是监督训练bridge，不代表已经训练出可用轨迹控制checkpoint；正式LDF长训尚未启动。
- 当前条件是dense active XZ与最多4秒future XZ；稀疏waypoint、heading约束、lookahead长度和dropout概率需要在保持同一condition合同下单独消融。
- Web/runtime仍等待正式LDF checkpoint loader和真实route timeline到`LDFCondition`的端到端控制验证，不能把合成forward敏感性测试等同于在线控制质量。

## 2026-07-16 · LDF dense trajectory、sparse waypoint与future goal混合采样

改动类型：轨迹控制训练分布 / 稀疏条件编译 / self-forcing约束生命周期

实际改动内容：

- 将LDF训练约束从单一dense active/future XZ扩展为三种按sample采样的模式：50% dense trajectory、25% sparse waypoints、25% single future goal。所有模式只观察root x/z，不暴露root y或heading。
- sparse waypoint在`[B,T,4,5]`的frame维随机选择1–4个帧，选择一个token不会自动暴露其中全部四帧；future goal清空首个active band内的约束，只选择一个严格位于其后的future frame。短span没有真实future frame时明确退化为一个active waypoint，不静默变成无约束样本。
- 每个batch先采样一次absolute XZ mask，再在整个self-forcing rollout内复用。当前step仅编译落入active band的帧；其后的稀疏约束按absolute timeline position压紧打包为future tokens。窗口右移时同一个goal会从future条件自然迁入active条件，不重新采样。
- 保留独立sample-level constraint dropout，并在mode采样后清空整份absolute mask；因此无约束样本只由CFG dropout产生，不与短样本fallback混淆。
- `train_ldf.py`新增mode概率和值域校验：三种概率均位于`[0,1]`且和为1，`max_waypoints`必须为正。HumanML和HumanML+BABEL配置统一冻结为`0.5/0.25/0.25`与最多4个waypoints。
- README和LDF训练方法、配置、实现文档同步记录frame-level sparse mask、packed future token和persistent constraint plan语义。

改动理由：

- dense path following只训练“沿完整轨迹走”，不能充分覆盖ARDY式“给一个稀疏waypoint或远期goal，让模型自己补全中间动作”的使用方式。将三种条件密度放进同一teacher baseline，才能让Root Stage同时学习路径跟随、稀疏插值和目标导向规划。
- 约束必须绑定absolute motion frame并在rollout内持久化；若每个self-forcing step重新采样，模型看到的任务会随窗口变化，单一goal也无法从future token连续迁移到active observation。
- constraint dropout与条件稀疏度承担不同职责：前者为CFG提供无约束分支，后者定义控制任务。显式分离可以避免把“没有future可采”错误解释成CFG无条件样本。

验证：

- 新增测试覆盖三种强制采样模式、frame-level XZ配对、1–4 waypoint上限、无future fallback、不同batch样本的稀疏future压紧打包和padding。
- self-forcing回归确认dense范围随active窗口移动，并确认同一个single future goal在下一step保持absolute identity并迁入active mask。
- 配置测试确认两份正式配置共享相同采样合同，并对概率和不为1、`max_waypoints<=0`执行fail-fast。
- 定向LDF条件/forward/self-forcing/训练/迁移测试：`70 passed`。
- 全仓`MPLCONFIGDIR=/tmp/matplotlib /home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests -q`：`191 passed, 1 warning`；warning仅为测试环境无法初始化NVML。
- 全仓Python文件`py_compile`通过；`git diff --check`通过。

涉及文件：

- `utils/training/ldf/{conditioning,lightning_module,__init__}.py`
- `train_ldf.py`
- `configs/{ldf,ldf_multi}.yaml`
- `tests/{test_ldf_training,test_ldf_self_forcing,test_migration_guards}.py`
- `README.md`
- `docs/rearchitecture/{04_TRAINING_METHOD,05_TRAINING_CONFIG,06_LDF_IMPLEMENTATION_DESIGN}.md`
- `docs/DEVELOPMENT_LOG.md`

尚未完成的后续事项：

- 本轮沿用root/body flow-v监督，没有同时增加masked root constraint-adherence或ARDY式额外goal loss；应先观察混合采样baseline的goal error，再将该loss作为独立消融引入。
- 50%/25%/25%、最多4个waypoints和20-token lookahead是首轮训练默认值，不是已经验证的最优超参数。
- 尚未启动正式LDF训练，也尚未用真实在线route验证稀疏goal控制质量。
