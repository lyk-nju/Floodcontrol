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
