# VAE 与 Root5 / Body259 表示

## 1. 唯一物理合同

每个20 FPS物理帧拆成：

```text
root_motion [F,5]
  [0:3] xyz
  [3:5] cos(yaw), sin(yaw)

body_motion [F,259]
  [0:63]    21个非Root关节的root-heading-local位置
  [63:189]  21个非Root关节的heading-frame cumulative IK rotation6d
  [189:255] 22个关节的current-heading-local backward velocity
  [255:259] 4维foot contact
```

`Root5`是世界XYZ平移和绝对heading yaw的唯一所有者，不包含root pitch/roll。
设`R_t`为root-heading-frame到世界坐标的主动yaw旋转：

```text
p_body(t,j) = R_t^T (p_world(t,j) - p_world(t,root))
p_world(t,j) = p_world(t,root) + R_t p_body(t,j)
v_body(t,j) = R_t^T (p_world(t,j) - p_world(t-1,j)) * 20
```

位置减去完整root XYZ，因此Body不携带绝对高度。速度使用当前帧heading、backward difference和m/s单位；cold-start第一帧速度置零且mask为false。Body中的root joint velocity只是重建/诊断信息，最终root运动始终由Root5决定。

世界平移或统一世界yaw只改变Root5；Body259保持数值不变。这是数据合同，不是近似增强目标。VAE/LDF collator的translation rebase和random yaw因此都只能修改Root5，不能旋转Body block。

## 2. Rotation gauge

当前HumanML与BABEL源都是HumanML-style 263D。rotation来自HumanML IK，不是原生SMPL rotation。明确区分：

```text
C_t = HumanML world-to-heading canonical rotation
R_t = physical root-heading-to-world yaw rotation
R_t = C_t^T
```

由HumanML parent-local IK rotation `L_j`恢复累计矩阵：

```text
A_0 = C_t
A_j = A_parent(j) L_j
B_j = R_t^T A_j, j=1..21
```

Body259保存`B_j`的矩阵前两列6D，名称固定为`heading-frame cumulative IK rotation`。它不是parent-local rotation。恢复累计世界/gauge矩阵时使用`A_j = R_t B_j`；评测回写parent-local rotation时必须重新按骨架父子关系求相对矩阵。

累计rotation较冗余，父子预测可能不完全一致。VAE验收必须同时报告direct-position、rotation-FK和两者差异；HumanML源本身已有非零position/FK误差，所以比较的是相对源基线，而不是强制绝对零。

关节顺序、parent表、raw offset方向及左右脚索引只由`utils/motion_process.py`定义。rotation6D始终保存矩阵前两列，禁止行展开。

## 3. VAE边界

VAE只编码Body259，root不进入encoder：

```text
encode()            physical Body259 -> raw posterior(mu, logvar)
decode()            raw latent + physical local-root4 -> physical Body259
tokenize()          physical Body259 -> deterministic raw mu
tokenize_window()   real context + active Body259 -> active raw mu
detokenize()        raw mu + physical local-root4 -> Body259
detokenize_step()   one raw mu token + explicit cache -> four physical frames
```

训练`forward()`使用posterior sample；LDF、评测和runtime固定使用raw deterministic `mu`。不生成`latent_mean/std`，不做latent whitening。

Decoder condition由Root5派生：

```text
local_root4 = [yaw_rate, current-heading-local vx, vz, root_y]
```

差分使用当前帧减前一帧。中间crop携带真实previous-root；真实序列起点的yaw rate和XZ velocity置零且无效，root height仍有效。

VAE只保留四组physical statistics buffer：

```text
body_cont_mean/std  [255]
local_root_mean/std [4]
```

contacts保持0/1。statistics仅从HumanML train split计算；HumanML+BABEL VAE也复用同一组HumanML statistics。训练checkpoint自包含四个buffer；LDF/runtime通过公共loader加载EMA encoder+decoder，不读取统计NPZ。

## 4. Token与causal合同

- `FRAMES_PER_TOKEN=4`，输入帧数必须整除4；预处理只丢弃不足四帧的尾部。
- encoder/decoder均为token轴causal convolution。
- 当前正式结构公开`encoder_context_tokens=decoder_context_tokens=24`，即96帧。
- `tokenize_window()`输入布局为`[真实context | active | 右padding]`，逐样本`context_token_count[B]`裁掉真实context；序列起点允许context不足，不插入假历史。
- decoder cache由调用方显式持有，module内部没有隐藏session状态。
- full encode与context-window encode、full decode与逐token stream decode必须数值一致。

## 5. Loss与诊断

基础重建分别对position、rotation、velocity做masked normalized SmoothL1，并对contact做BCE。position-derived skating只约束接触脚的世界位置速度，不代表脚方向或FK一致性。

正式训练持续记录posterior `mu` mean/RMS/channel std、logvar、KL、active latent fraction和sigma。VAE评测至少报告：

- Body local reconstruction；
- GT Root5下的world MPJPE；
- rotation geodesic；
- source FK/direct基线、reconstruction FK/direct和FK/target误差；
- ankle-toe方向误差、反向比例和长度误差；
- contact accuracy、position-derived skating；
- offline/stream parity。

FK/geodesic/position-consistency/velocity-consistency可作为独立消融；启用FK loss必须提供与冻结HumanML22 parent/offset合同一致的skeleton offsets。

## 6. LDF接口

LDF只依赖EMA VAE的：

```text
latent_dim
encoder_context_tokens
tokenize_window(...)
detokenize(...)
init_decoder_state(...)
detokenize_step(...)
```

LDF的Body状态就是raw posterior `mu`。LDF checkpoint不保存VAE参数；启动时独立加载并冻结VAE，且VAE不进入LDF optimizer或EMA。

Body259改变了position、rotation和velocity坐标系，因此旧Body265 VAE/LDF checkpoint不兼容。新VAE必须从头训练，新LDF必须基于新VAE从头训练；shape不符时直接失败。

## 7. HumanML263评测边界

标准T2M evaluator仍消费HumanML263。转换前必须移除序列初始XZ与初始heading：

```text
R_rel(t) = R_abs(0)^T R_abs(t)
C_HML(t) = R_rel(t)^T
A_canon(t,j) = R_rel(t) B(t,j)
```

对root直接子关节：`L_j = C_HML^T A_canon,j`；其他关节：`L_j = A_canon,parent^T A_canon,j`。直接把绝对Root5 heading写入HumanML rotation会在第一层关节产生双重yaw。

RIC XZ来自Body local position，RIC Y为`body_local_y + root_y`。root transition从Root5相邻帧计算；evaluator joint velocity从恢复后的世界position用forward difference重算，不使用Body中的backward velocity。`F`个物理pose严格产生`F-1`个HumanML rows。

正式发布门槛包括263 round-trip、0/45/90/180度及随机yaw不变性、第一层髋/脊柱rotation测试、T2M feature漂移和GT self-FID基线。

## 8. Root5 / Body259 表征实验记录

本节记录的是**表示与评测适配器**的验证，不是VAE重构质量或LDF生成质量。所有
T2M指标均使用HumanML官方movement/motion evaluator；FID比较同一批真实motion在
转换前后的embedding分布。

### 8.1 单样本数值闭环

真实HumanML样本`000021`执行：

```text
HumanML263
-> physical Root5 + heading-local Body259
-> canonical HumanML263
```

结果：

| 字段 | 最大绝对误差 |
|---|---:|
| non-root position | `2.98e-7` |
| HumanML root block | `6.03e-7` |
| parent-local rotation6d | `6.11e-7` |

contacts逐元素不变。velocity没有要求逐元素严格闭环：逆转换按恢复后的world position
重新计算官方forward velocity，而不是复制源263D中的冗余velocity。该策略保证评测
velocity与恢复后的几何一致。

### 8.2 32样本早期T2M冒烟

在32个真实HumanML test样本上，使用严格`tail=drop`合同进行转换：

| 指标 | 结果 |
|---|---:|
| motion embedding cosine | `0.9999986887` |
| motion embedding L2 | `0.00379194` |
| round-trip FID | `0.0001703281` |

该实验用于在全量验证前检查Root第一层rotation回写、初始heading移除和evaluator输入
长度。后续1450样本实验是更强的正式结果。

### 8.3 HumanML完整validation round-trip

协议：HumanML3D val split全部1450条动作；每条`F`帧HumanML263恢复为`F`个physical
pose，再按严格合同输出`F-1`行HumanML263，与源序列`source[:-1]`比较。结果保存于
本机gitignored的`debug/results/body259_rotation_fid_val1450.json`，关键数值在此文档
固化。

特征误差：

| block | MAE | 最大绝对误差 |
|---|---:|---:|
| root | `1.3774e-8` | `8.4564e-7` |
| positions | `8.2745e-9` | `5.0664e-7` |
| rotations | `4.9190e-8` | `1.0431e-6` |
| velocities | `6.1006e-6` | `0.47579` |
| contacts | `0` | `0` |

T2M embedding结果：

| 指标 | 结果 |
|---|---:|
| cosine mean | `0.9999991655` |
| L2 mean | `0.00205983` |
| FID | `8.28769e-6` |

velocity的较大单点最大误差来自少量源HumanML263样本的velocity通道本身与position
forward difference不一致；其整体MAE和T2M embedding drift仍很小。position、root、
rotation和contact闭环均处于float32数值误差量级，因此没有发现系统性的坐标符号、
第一层yaw或rotation6D行列错误。

### 8.4 全局yaw不变性与旋转FID

对同一1450条val motion，在Root5/Body259空间施加0°、45°、90°、180°全局yaw，
以及seed `1234`的逐样本`Uniform[0,360°)` yaw。变换只旋转Root5的XZ和heading，
Body259逐元素保持不变；随后统一回写canonical HumanML263。

| yaw | 相对未旋转round-trip FID | 相对原始source FID | 全feature最大绝对误差 |
|---|---:|---:|---:|
| 0° | `5.52e-10` | `8.28637e-6` | `8.05e-7` |
| 45° | `2.13e-10` | `8.28750e-6` | `1.42e-6` |
| 90° | `0` | `8.28718e-6` | `1.88e-6` |
| 180° | `2.21e-10` | `8.28710e-6` | `1.49e-6` |
| random | `3.79e-10` | `8.28846e-6` | `1.91e-6` |

所有旋转相对未旋转round-trip的embedding cosine mean均为`1.0`；FID的`1e-10`
量级变化属于协方差矩阵平方根的数值噪声。不同yaw相对原始source的FID稳定在
`8.29e-6`附近，说明表示和逆转换不会随世界朝向产生可观测分布漂移。随机yaw实际
覆盖`0.23°`至`359.91°`，均值`180.79°`。

### 8.5 尾帧策略对照

HumanML每一行包含从当前pose到下一pose的root transition，所以`F`个physical pose
只能严格恢复`F-1`行。两种诊断策略结果为：

| 策略 | 比较对象 | embedding cosine | embedding L2 | FID |
|---|---|---:|---:|---:|
| exact drop | `source[:-1]` | `0.99999917` | `0.00206` | `8.29e-6` |
| approximate tail | 完整`source` | `0.99999124` | `0.01091` | `2.88e-4` |
| drop vs full length | 完整`source` | `0.99989188` | `0.07645` | `3.78e-3` |

因此正式evaluator adapter固定使用exact drop。approximate tail只用于说明“伪造最后
一个transition”虽可近似工作，但不是无损合同；把短一行的结果直接与full-length
embedding比较还会混入长度变化，不应解释为representation误差。

### 8.6 已证明的结论与尚未证明的事项

上述实验支持：

- Root5/Body259与HumanML263之间的position、root、rotation和contact转换数值闭合；
- Body259对统一世界yaw保持不变，Root5独占世界XZ和absolute heading；
- HumanML evaluator回写没有第一层root yaw双计数或固定角度相关漂移；
- exact-drop adapter本身不会对后续T2M FID构成有意义的误差下限。

上述实验**不**证明：

- 新VAE能够以低MPJPE重构Body259；
- VAE预测的cumulative rotation与direct position天然一致；
- LDF能够生成正确动作、脚部几何或轨迹；
- HumanML IK rotation等同于原生SMPL rotation。

当前合成/单元测试已经锁定world position恢复、`A=RB`、rotation正交性、FK helper、
translation/yaw不变性和current-heading backward velocity。真实新VAE训练时另行记录
`world_mpjpe_m`、`source_fk_direct_mpjpe_m`、`reconstruction_fk_direct_mpjpe_m`和
`reconstruction_fk_target_mpjpe_m`。只有重建FK指标明显高于source FK/direct基线，
才说明应增加FK或position-consistency loss；不能要求HumanML源的IK gauge基线为零。

### 8.7 复现实验

```bash
python tools/compare_humanml_adapter.py \
  --split /path/to/raw_data/HumanML3D/val.txt \
  --motion-root /path/to/raw_data/HumanML3D/new_joint_vecs \
  --deps /path/to/deps \
  --samples 1450 \
  --device cuda:0 \
  --yaw-degrees 0 45 90 180 \
  --random-yaw \
  --seed 1234 \
  --output debug/results/body259_rotation_fid_val1450.json
```
