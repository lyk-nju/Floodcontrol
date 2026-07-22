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
