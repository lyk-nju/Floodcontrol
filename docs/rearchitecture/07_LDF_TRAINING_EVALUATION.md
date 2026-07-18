# 07 LDF训练期生成评测

状态：`IMPLEMENTED`

本文只说明训练过程中如何观察完整生成质量，不重新定义训练loss。普通validation仍负责`teacher_cold`、`teacher_continuation`和self-forcing probe的确定性loss；生成评测由独立runner按较低频率执行。

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
    -> active-band末端之后的future XZ lookahead
    -> constraint CFG
```

future lookahead从`commit_index + chunk_size`开始，不从物理window末端开始。因此固定buffer和rolling window在同一commit时刻看到相同的future horizon。

每个样本记录ADE、FDE、time-aligned MSE、20/50cm失败率、分段/前缀MSE、path arclength误差、Chamfer误差、轨迹jitter、foot skating和token边界跳变。对前若干样本输出GT动作、生成动作和俯视XZ轨迹的并排视频。

## T2M评测

HumanML3D生成结果通过唯一`root5/body265 -> HumanML263`converter进入已有T2M evaluator，记录FID、Matching Score、R-Precision以及Diversity。正式训练与FloodNet一致，T2M读取`data.val_meta_paths`并遍历完整HumanML3D validation split，不再提供容易被误当成正式FID的`max_samples`捷径。

T2M评测不提供轨迹条件；dense XZ评测和T2M评测分别回答“控制是否准确”和“无轨迹提示时动作分布是否合理”，不能把二者混为同一指标。

## 调度与配置

正式配置入口：

```yaml
validation:
  generation:
    enabled: true
    steps: 5000
    modes: [stream]
    num_runs: 3
    video_samples: 4
    num_denoise_steps: 10
    max_horizon_token: 10
    rolling:
      window_tokens: 40

  dense_xz:
    enabled: true
    probe: dense_xz
    segment_frames: 20

  t2m:
    enabled: true
    steps: 10000
```

生成视频/轨迹probe不从validation split中隐式取“前N条”，而是按FloodNet格式由`data.test_probe_meta_paths.dense_xz`指定小型TXT；当前文件包含8条固定HumanML3D测试样本。T2M仍独立使用完整validation split。

heavy evaluation在非sanity validation且step命中周期时执行，并支持单卡或DDP。所有rank都进入同一个EMA与collective作用域，但generation sample按全局稳定编号round-robin分片，每条motion只由一个rank生成。当前实验只启用`stream`；将modes改为`[stream, rolling]`即可同时比较两条runtime路径。运行前后每个rank分别恢复自己的Python、NumPy、Torch和当前CUDA device RNG状态。EMA参数在整轮validation开始时只交换一次，普通loss probe与inline evaluation共享同一EMA scope；epoch-end或异常路径恢复训练权重并释放临时副本，不改变optimizer、scheduler、LDF stream state或VAE decoder state。

DDP聚合遵守以下边界：

- 训练length-bucket batch sampler已经按global rank切分等长step；普通teacher/self-forcing validation也按rank无重复分片，并继续用`sync_dist=True`归约loss。生成评测不依赖validation DataLoader迭代，而是在相同完整Dataset上使用自己的稳定sample shard。
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

`probe`为`dense_xz_stream`、`dense_xz_rolling`、`t2m_stream`或`t2m_rolling`。视频失败只产生warning；数值artifact与summary仍然保留。

对于dense XZ probe，人物和轨迹共享同一个固定正交相机，轨迹直接画在人物脚下的世界地面上，而不是放入独立俯视面板：

- `video/`输出单画面的生成动作；目标/条件轨迹为红色，生成root已走过的轨迹为蓝色，并随人物逐帧延伸。
- `composite/`输出“GT场景 + 生成场景”两栏；两栏使用相同的目标轨迹和固定相机语义，生成侧额外显示蓝色实际轨迹。

历史checkpoint目录中的旧视频不会被自动重写；重新运行对应generation evaluation后才会采用上述布局。

## 当前边界

- 当前只实现dense XZ视频/控制指标，没有稀疏waypoint或goal专用生成probe。
- T2M训练期默认遍历完整HumanML3D validation split；它可用于checkpoint趋势比较，最终论文结果仍应使用独立固定evaluation protocol和重复随机运行。
- 生成目前逐样本运行，优先保证与真实runtime语义一致；后续若做批量加速，必须证明与逐session结果一致。
- BABEL可参与dense XZ评测，但T2M evaluator只用于HumanML3D。
- DDP只并行不同evaluation sample；单个`InferenceSession`仍按token顺序生成，不改变stream/rolling的数值语义。
