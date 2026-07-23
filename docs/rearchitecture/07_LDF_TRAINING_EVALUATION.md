# 07 LDF训练期生成评测

状态：`IMPLEMENTED`

本文只说明训练过程中如何观察完整生成质量，不重新定义训练loss。普通validation只保留一个确定性的continuation probe，记录`val/loss_total`、`val/loss_root`和`val/loss_body`三条可横向比较的曲线。cold-start、persistent commit和长程一致性不能由ideal bridge loss代替，因此统一交给较低频率的真实stream runner测量。T2M评测保持独立，不与控制指标混合。

## 两种生成模式

两种模式共用同一套`InferenceSession`、三角调度、Root/Body forward、constraint CFG、dense XZ route compiler和VAE逐token解码。它们只在LDF窗口是否滚动上有区别：

- `stream`：创建覆盖完整目标序列的固定buffer，从第一个token依次解析到最后一个token；不移动`window_origin`，用于观察完整三角调度在固定序列上的生成质量。
- `rolling`：使用固定大小在线窗口；每次commit执行与stream相同的model-origin rebase，达到roll边界后再执行纯buffer roll和噪声补充，用于模拟部署时的长期rollout。

两种模式都一次只commit一个hybrid token，并立即通过持久VAE decoder state解码四帧。`stream`不是一次性offline denoise，`rolling`也不是事后切片模拟。

## Dense XZ评测

首轮只评测dense XZ，不向模型暴露GT的root y或yaw：

```text
target root XZ at every frame
    -> world route timeline
    -> current-window XZ condition
    -> 当前microstep真实可见motion末端之后的future XZ lookahead
    -> constraint CFG
```

每个commit先从`commit_index + 1`开始编译覆盖尚未可见active位置和远期轨迹的候选superset；每个denoise microstep按当前`history_mask | generation_mask`动态把future起点推进到真实visible-motion末端，并限制为配置的最大horizon。route/text仍然每commit只编译一次。固定buffer和rolling window在同一commit/microstep看到相同absolute future语义。

每条完整stream rollout都计算详细JSON；W&B只公开以下短名核心指标：

```text
val/cold/root_deg       首个commit四帧的Root/GT角度均值
val/cold/root_anti      首个commit四帧的近180°比例
val/cold/body_deg       首个commit四帧的Body/GT Body角度
val/cold/feet_deg       首个commit四帧的feet/GT feet角度

val/roll/root_deg       完整序列Root/GT角度均值
val/roll/root_p95       完整序列Root/GT角度P95
val/roll/root_anti      完整序列Root近180°比例
val/roll/body_deg       完整序列Body/GT Body角度
val/roll/feet_deg       完整序列feet/GT feet角度
val/roll/body_rel       Body相对Root关系的生成/GT误差
val/roll/feet_rel       feet相对Root关系的生成/GT误差
val/roll/feet_rev       生成新增、但GT不存在的反向脚比例
val/roll/ade            dense XZ逐帧平均误差（米）
val/roll/fde            dense XZ末帧误差（米）
```

`body_rel/feet_rel`比较相对角度之差，不强迫Body或脚始终平行Root，因此倒退、侧步和转身只要与GT关系一致就不会被误判。详细JSON仍保留max error、foot skating、边界跳变、GT轨迹切线和旧的完整朝向分解，便于离线定位，但这些量不再占据W&B曲线列表。

固定样本`000021`与`001168`始终采用真实`H=0` stream、dense GT XZ、joint CFG=3以及seed `4321/4322/4323`生成完整clip。每个样本只记录八条跨seed汇总：

```text
val/case/<id>/cold_mean
val/case/<id>/cold_max
val/case/<id>/root_mean
val/case/<id>/root_max
val/case/<id>/body_rel_mean
val/case/<id>/body_rel_max
val/case/<id>/feet_rel_mean
val/case/<id>/feet_rel_max
```

因此非T2M部分总计33条标量曲线：3条基础loss、14条cold/rollout指标和
2个固定样本各8条指标。T2M继续使用既有指标集合，不计入这33条。

## T2M评测

HumanML3D生成结果通过唯一`Root5/Body259 -> HumanML263`converter进入已有T2M evaluator，记录FID、Matching Score、R-Precision以及Diversity。转换先移除初始XZ与初始heading，再恢复HumanML IK parent-local rotation；joint velocity从恢复后的world positions用forward difference重算。T2M读取`data.val_meta_paths`并遍历完整HumanML3D validation split，不提供容易被误当成正式FID的`max_samples`捷径。

T2M评测不提供轨迹条件；dense XZ评测和T2M评测分别回答“控制是否准确”和“无轨迹提示时动作分布是否合理”，不能把二者混为同一指标。正式配置统一使用`cfg_mode: joint`与`cfg_scale_joint: 3.0`：该值来自step-160000 EMA在完整1450条HumanML3D validation样本上的固定seed stream sweep，离散最优出现在约3.125，正式默认取更简单且相邻表现接近的3.0。remote/local/multi不再覆盖这一语义。

每次T2M评测完成后，rank 0必须在训练控制台打印本轮样本数、生成模式、CFG模式，以及实际计算出的FID、Matching Score、R-Precision和Diversity；相同数值继续写入评测`summary.json`和Lightning/WandB日志。其他DDP rank不重复打印。

## 调度与配置

正式配置入口：

```yaml
validation:
  generation:
    enabled: true
    run_at_start: true
    steps: 5000
    modes: [stream]
    num_runs: 3
    num_denoise_steps: 10
    max_horizon_token: 10
    rolling:
      window_tokens: 50

  dense_xz:
    enabled: true
    probe: dense_xz
    standard_cases: ["000021", "001168"]
    video_yaw_degrees: [0, 90, 180]

  t2m:
    enabled: false
    steps: 10000
    cfg_mode: joint
```

生成轨迹probe不从validation split中隐式取“前N条”，而是按FloodNet格式由`data.test_probe_meta_paths.dense_xz`指定TXT。该TXT是总体rollout指标的来源；`standard_cases`必须包含于该TXT。所有样本使用相同的三组paired source seed参与数值聚合，固定样本三组run都额外渲染完整视频；run 0仍保留原始、`+90°`和`+180°`的yaw等变性视频。T2M仍独立使用完整validation split，计算、命名和输出合同不受本轮日志收口影响。

三组yaw视频采用paired-noise合同：Root5与dense world-XZ route同步旋转，Body259/raw latent source保持逐元素相同；文本、seed和denoise设置保持一致。因此三组差异用于诊断模型的yaw等变性，而不是比较三个无关随机生成。`run_at_start: true`使训练入口在`trainer.fit()`之前显式执行一次`trainer.validate(..., ckpt_path=resume_ckpt)`；该轮validation使用加载后的checkpoint/EMA并额外生成固定视频，目录标签为`fit_start`。

heavy evaluation在非sanity validation且step命中周期时执行，并支持单卡或DDP。所有rank都进入同一个EMA与collective作用域，但generation sample按全局稳定编号round-robin分片，每条motion只由一个rank生成。当前实验只启用`stream`；将modes改为`[stream, rolling]`即可同时比较两条runtime路径。运行前后每个rank分别恢复自己的Python、NumPy、Torch和当前CUDA device RNG状态。EMA参数在整轮validation开始时只交换一次，普通loss probe与inline evaluation共享同一EMA scope；epoch-end或异常路径恢复训练权重并释放临时副本，不改变optimizer、scheduler、LDF stream state或VAE decoder state。EMA shadow的device迁移和checkpoint加载显式退出Lightning validation的`inference_mode`，避免fit前validation把长期EMA storage变成后续无法原地更新的inference tensor。

DDP聚合遵守以下边界：

- 训练length-bucket batch sampler已经按global rank切分等长step；普通continuation validation也按rank无重复分片，并继续用`sync_dist=True`归约三项loss。生成评测不依赖validation DataLoader迭代，而是在相同完整Dataset上使用自己的稳定sample shard。
- dense XZ的每卡sample artifact使用互不重叠的sample ID直接写入共享输出目录；各rank只汇总轻量record和video path，只有rank 0写全局`summary.json`并记录W&B。
- T2M evaluator在每张卡上编码本地motion/text shard；CPU embedding通过collective汇总，并按全局validation sample index恢复原始顺序后计算FID、Matching Score、R-Precision和Diversity。这样world size不会改变R-Precision分组或随机diversity采样输入顺序。
- 没有分到sample的rank仍参与collective，不能提前返回；每个mode结束后使用barrier，避免下一轮evaluation与上一轮artifact聚合交叉。
- 多机DDP要求`save_dir`位于所有节点可见的共享文件系统。单机多卡没有额外目录要求。

完整生成评测不是`LDFLightningModule`的内部成员。`train_ldf.py`仅在`generation.enabled`且至少一个生成指标启用时惰性导入`eval/ldf_training.py::LDFEvaluationCallback`；callback再组合heavy runner。Lightning module始终只依赖训练kernel，因此关闭生成评测时不会加载T2M metrics、视频渲染或`utils.inference`。callback的epoch-end hook在所有rank上复用validation已激活的EMA参数，随后module hook统一恢复训练权重；只有最终summary与logger写入受rank-zero约束。

## 输出目录

输出沿用FloodNet便于按artifact类型浏览的结构：

```text
<run_dir>/<dataset>/
├── text/<probe>/<step>/
├── token/<probe>/<step>/
├── feature/<probe>/<step>/
├── traj_xz/<probe>/<step>/
├── traj_mask/<probe>/<step>/
├── frames/<probe>/<step>/
├── metrics/<probe>/<step>/
├── video/<probe>/<step>/
└── composite/<probe>/<step>/
```

`probe`为`dense_xz_stream`、`dense_xz_rolling`、`t2m_stream`或`t2m_rolling`。固定yaw视频使用`000021_yaw_000deg.mp4`、`000021_yaw_090deg.mp4`等名称；视频失败只产生warning，数值artifact与summary仍然保留。

对于dense XZ probe，人物和轨迹共享同一个固定正交相机，轨迹直接画在人物脚下的世界地面上，而不是放入独立俯视面板：

- `video/`输出单画面的生成动作；目标/条件轨迹为红色，生成root已走过的轨迹为蓝色，并随人物逐帧延伸。
- `composite/`输出“GT场景 + 生成场景”两栏；两栏使用相同的目标轨迹和固定相机语义，生成侧额外显示蓝色实际轨迹。

历史checkpoint目录中的旧视频不会被自动重写；重新运行对应generation evaluation后才会采用上述布局。

## 当前边界

- 当前只实现dense XZ视频/控制指标，没有稀疏waypoint或goal专用生成probe。
- T2M训练期默认关闭；启用时遍历完整HumanML3D validation split。最终论文结果仍应使用独立固定evaluation protocol和重复随机运行。
- 生成目前逐样本运行，优先保证与真实runtime语义一致；后续若做批量加速，必须证明与逐session结果一致。
- BABEL可参与dense XZ评测，但T2M evaluator只用于HumanML3D。
- DDP只并行不同evaluation sample；单个`InferenceSession`仍按token顺序生成，不改变stream/rolling的数值语义。
