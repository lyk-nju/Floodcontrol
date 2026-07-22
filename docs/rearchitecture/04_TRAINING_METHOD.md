# LDF训练方法

状态：`ROOT_X0 / BODY_VELOCITY / PERSISTENT_COLD IMPLEMENTED`

## 1. 训练状态

```text
HybridMotion.root_motion   = physical root5
HybridMotion.latent_motion = raw deterministic EMA-VAE mu
```

Root和Body共用FloodDiffusion的per-token三角`beta`生命周期，但原生预测类型独立：

```text
RootTransformer = physical x0
BodyTransformer = raw-latent velocity
```

代码同时允许Root/Body各自选择`x0`或`velocity`用于消融，不接受`vel`别名。网络原生输出先完成CFG组合，再解释为统一的`clean_motion`和`solver_velocity`。Euler solver只读取`solver_velocity`；Root始终额外构造投影到heading单位圆的physical clean view，用它派生local root供Body Stage和VAE decoder使用。

Root solver固定在token生命周期内运输到raw x0，只在commit写入persistent state时强制heading单位圆。Root→Body边界始终使用临时投影后的physical clean view，因此Body不会接收非单位heading。persistent-cold各phase持续观察`root_heading_angle_degrees`、`root_heading_raw_norm`、`root_heading_raw_norm_p10`和`root_heading_low_norm_ratio`。

## 2. 窗口与课程

- parent最多50 tokens/200 frames，短序列保留自然长度；
- active chunk固定5 tokens；
- 每个sample独立采样history start，普通batch要求`H>=1`；
- true cold明确使用真实序列起点与`H=0`；
- future horizon逐样本在合法的`0..45`内采样；
- K课程在global step `0/100k/200k/300k`进入`1/2/3/5`；
- cold-start replay从第0步到训练结束占全部global batch的10%。

K=1是同一solver入口的ideal训练，不存在另一套“普通训练”实现。K>1从runtime commit boundary维护persistent noisy state；只在起点调用一次`mix_fixed_noise()`，之后每个microstep都使用上一次Euler输出。训练/runtime共用`LDF.create_input()/denoise_step()/commit_step()`，因此visibility、更新、heading投影、commit和rebase只有一份数学实现。

## 3. Root-x0目标

Root Transformer的raw输出是未投影physical root5。只在active mask上分别计算：

```text
L_root_xz      = SmoothL1(raw_x0[x,z], target[x,z]).mean()
L_root_height  = SmoothL1(raw_x0[y],   target[y]).mean()
L_root_heading = SmoothL1(raw_x0[cos,sin], target[cos,sin]).mean()
L_root          = L_root_xz + L_root_height + L_root_heading
```

三块默认等权，避免维数隐式改变语义。heading loss读取投影前向量，因此精确180度反向时仍有非零梯度；单位圆投影生成Root→Body physical view并规范化committed state。角度、raw norm分布、low-norm ratio和antipodal ratio是detached指标，不再作为辅助loss。

history和不可更新token始终保留当前authoritative root。active mask中任何`beta<=0`都是合同错误；实现不会先除零再用`where`遮盖。

## 4. Body-velocity目标

Ideal bridge仍监督原始flow displacement：

```text
x_beta = (1-beta) * x0 + beta * noise
v_target = x0 - noise
L_body_ideal = MSE(v_raw, v_target)
```

Persistent state可能偏离ideal bridge，因此直接监督当前state所需的corrective velocity：

```text
corrective_error = (x_current + beta * v_raw - x0) / beta
L_body_persistent = SmoothL1(corrective_error, 0)
```

这里只在真实active beta上计算，不使用`beta_min`。代数上它等价于`v_raw-(x0-x_current)/beta`；SmoothL1直接限制velocity-space梯度，避免旧`endpoint_loss/beta²`在Huber线性区产生倒beta放大。最后一个完整`delta_beta=beta`更新能精确落到模型所预测的endpoint。

Root/Body四种prediction组合共享同一个派生接口：

```text
raw_root_output
raw_body_output
clean_motion
solver_velocity
local_root_motion
local_root_feature_valid
```

## 5. Persistent cold

部署首次生成会从`H=0,current_step=0`连续积分，而早期训练曾只覆盖cold ideal bridge和有history的persistent state。`debug/body_prefix_oracle_experiment.py`对`step_160000.ckpt`的坏例`000021/seed4322`显示：全部模型预测的feet/root反向帧比例为`0.517`；前两个committed body tokens替换GT后降为0，只替换第一个commit仍会在token 2重新进入错误mode。direct position与FK foot方向同时反向，说明主要是cold active生命周期内Body latent进入错误下肢mode，而不是单一VAE输出head冲突。

因此cold batch内部固定为ideal/persistent各50%：

- ideal cold随机选择一个denoise phase并保留原始flow目标；
- persistent cold从同一fixed source执行第一个完整10-microstep commit及第二个2-microstep commit；
- 12个生命周期位置中只选一个保留梯度，之前步骤以`no_grad`真实推进；
- Dynamic Future、commit、translation rebase和condition重编译与runtime一致；
- 不decode→encode body，不重新采样source。

Validation的`persistent_cold` probe固定观察microstep `0/4/9/11`，分别对应首次update、中期可见性、第一commit边界和第二commit边界。它与只测ideal `H=0,K=1`的teacher-cold probe职责不同。

## 6. XZ条件与文本

每个rollout先从translation-anchored、random-yaw后的physical root采样一份absolute XZ计划，teacher/self-forcing各step复用。constraint仅包含XZ，不暴露height或heading。

候选future覆盖rollout后移所需superset；每个microstep再根据实际`history_mask | generation_mask`过滤，使future始终从当前可见motion末端开始。cold早期尚未成为motion query的active XZ因此仍作为future条件可见，同一absolute token不会同时作为motion query和future query。

正式采样比例为dense/waypoint/goal=`0.5/0.25/0.25`，constraint dropout为0.2，text dropout为0.1，两者独立。训练与dense-XZ评测默认joint CFG；T2M也使用joint CFG，`cfg_scale_joint=3.0`。

HumanML caption在整段token重复；BABEL任意frame区间按最大重叠编译。每个motion token直接cross-attend自己的prompt，但visible motion之间的self-attention仍可传播语义，因此它是direct token-aligned cross-attention，不是严格信息隔离。

## 7. Loss组合与指标

```text
L_total = rollout_scale * (
    root_weight * L_root
  + body_weight * L_body
) + root_boundary_weight * L_boundary
```

`root_boundary_weight`默认0，并直接读取physical clean root。日志为所有prediction类型返回固定键，未启用分支写零，以保持DDP静态日志/参数图。

Root观察指标至少包含raw heading角误差、raw heading norm和antipodal ratio；完整评测另记录Root/GT轨迹、Root/Body及feet/root方向关系。指标不参与主loss。

## 8. 必须保持的回归

- 四种Root/Body prediction组合都满足clean endpoint与solver update恒等式；
- CFG先组合raw输出，再投影Root heading；
- active beta0 fail-fast，history不做除法且不被预测覆盖；
- beta 1.0与0.1下corrective velocity梯度均由SmoothL1限制；
- persistent cold、K>1与runtime单microstep/commit/rebase数值一致；
- per-commit rebase前后world-space committed root不变；
- DDP各rank走相同K/cold计算分支，静态参数图不超时。
