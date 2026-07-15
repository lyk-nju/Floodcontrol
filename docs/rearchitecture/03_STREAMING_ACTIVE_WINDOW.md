# 03 在线推理与 Active Window

状态：`INFERENCE_CORE_IMPLEMENTED / WEB_ADAPTER_OPEN`

## 1. 范围

本阶段已经实现新版Hybrid LDF与BodyVAE之间的在线推理核心：world-space条件编译、逐token生成、commit-time causal decode、窗口rolling后的translation rebase、session snapshot/restore和失败回滚。Web已接入per-session锁、后台生产、四帧chunk backpressure队列、route/text/CFG更新与浏览器播放；正式LDF checkpoint/text encoder loader和视频编码仍属于后续阶段。

推理层不拥有第二个生成器，不从body reconstruction反推root，也不实现另一套LDF active-window调度。

## 2. 顶层数据流

```text
text / world XZ route / sparse root observations
                    |
                    v
        InferenceConditionCompiler
                    |
                    v
               LDFCondition
                    |
                    v
LDFStreamState -> LDF.stream_generate_step()
                    |
                    v
 committed normalized HybridMotion [B,1]
                    |
      root denormalize + world translation
                    |
        backward local-root4 derivation
                    |
                    v
VAEDecoderState -> BodyVAE.detokenize_step()
                    |
                    v
 GeneratedMotionChunk: physical root5/body prediction [B,4]
```

只有最终committed token进入VAE decoder。active window中仍在变化的root/latent不得提前decode，也不得通过decoder结果反馈或替换LDF root。

LDF在commit边界将clean root heading投影回单位圆，并把投影后的同一token写入persistent clean history；inference只验证该合同，不在模型外产生第二份修正root。

## 3. 唯一状态所有权

| 状态 | 唯一所有者 | 推理层职责 |
|---|---|---|
| `LDFStreamState` | LDF scheduler | 保存并交换out-of-place candidate state |
| `VAEDecoderState` | BodyVAE | 保存并交换out-of-place causal caches |
| world/model `origin_xz` | `InferenceSession` | world条件编译与输出恢复 |
| `previous_root_frame_world` | `InferenceSession` | 下一token的backward local-root边界 |
| text/route/root observations | condition模块 | 保存权威外部输入与revision |
| 完整已输出动作 | session调用方 | 收集`GeneratedMotionChunk`，不复制到核心session |

因此正式runtime不存在`RootTimeline`、body recovery anchor、RootRefiner state、LDF Transformer KV cache或第二套rolling buffer。完整输出历史由Web/eval调用方保存；模型核心只保存有限窗口，session只保存上一committed root frame。

## 4. 时间协议

```text
token k = frames [4k, 4k+1, 4k+2, 4k+3]
FPS = 20
token_dt = 0.2 seconds
```

route按absolute frame time在20 FPS采样后组成root patch，不能每token只采一个点再静默复制。`window_origin`、`commit_index`和condition stamp均使用absolute token index。

三类长度必须分开：

- LDF persistent motion window包含clean history、partially-noisy generation和pure-noise frontier；单次denoiser forward只把history与本步已经开始更新的generation组成有效前缀，尚未触及的pure-noise tail不进入attention；
- LDF future constraint tokens位于motion window之后，只进入Root Transformer条件token；
- VAE decoder没有future，只消费当前committed token和persistent causal state。

future constraint数量不得改变motion beta、motion RoPE、latent长度或VAE调用次数。

## 5. 坐标协议

V1只使用translation origin，不使用SE(2) yaw anchor：

```text
model_xz = world_xz - origin_xz
world_xz = model_xz + origin_xz
yaw_model = yaw_world
```

理由是body latent保存global rotations，模型外单独旋转root会破坏root/body一致性；XZ translation不改变global rotations或heading-local velocity。初始world位置由`origin_xz`提供，初始yaw属于显式root heading observation。

world route和root observations始终是权威事实。每次condition compilation都使用当前origin重新生成normalized model-space value/mask，不缓存坐标相关embedding。

## 6. Window roll与translation rebase

训练clip的planar root通常位于局部零点附近。LDF window roll完成后，session使用新窗口前一physical root frame作为translation offset，并调用LDF拥有的`scheduler-aware`变换：

```text
delta_normalized = delta_physical / root_std[xz]
root_t[xz] -= (1 - beta) * delta_normalized
previous_root_frame[xz] -= delta_physical
origin_xz += delta_physical
```

所以：

- clean history完整平移；
- partially-noisy token按clean coefficient平移；
- pure noise不变；
- latent、heading、root height、VAE state与RNG不变；
- 下一step从world事实重新编译所有root条件。

该操作由`LDF.rebase_stream_state()`实现，因为root statistics、三角beta和normalized noisy state属于LDF scheduler，不属于Web或condition compiler。

## 7. 条件合同

### 7.1 Text

`TextTimeline`使用严格半开区间`[start_token,end_token)`。相同文本只通过`TextEmbeddingCache`编码一次；cache保存CPU派生结果，不属于确定性snapshot状态。text update不能修改committed token，但可以在尚未commit的未来token预先安排。

condition compiler可以预先解析完整window的prompt timeline，但Root/Body cross-attention会按motion token逐项选择自己的prompt，并由当前有效前缀mask屏蔽pure-noise tail。可见token注入的prompt可以在后续层通过non-causal motion self-attention参与动作过渡；不可见future token则不能通过“扩大seq_len”或共享文本序列提前泄漏到当前动作。

### 7.2 Route

`RoutePlan`只表示time-parameterized world XZ route：

```text
times:        float32 [N], seconds, starts at zero and strictly increases
points_xz:    float32 [N,2]
start_token:  absolute token
end_behavior: hold | release
```

`WORLD`与`RELATIVE_TO_ACTOR`只描述route输入时的坐标参考。相对route在update时根据当前committed actor root转换一次，此后作为world事实保存；不得每step追随actor重新anchor。

首版route update只在当前commit立即生效。未来若需要scheduled route update，必须实现真正的route timeline和feature-wise compilation；不保留旧版只记录却不执行的`delay_tokens/blend_tokens`字段。

route默认只约束root5的XZ：

```text
value = [x, 0, z, 0, 0]
mask  = [1, 0, 1, 0, 0]
```

不从route tangent隐式推导heading。root height、heading和稀疏root pose由`RootObservationTimeline`显式提供；heading cos/sin mask必须成对且位于单位圆。显式root observation对同一frame、同一feature的route值具有优先权。

所有`token < commit_index`的root condition mask在编译时清空。约束只作为branch-local CFG input view，不hard replace generated root或persistent state。

### 7.3 Condition stamp

`CompiledCondition`记录：

```text
window_origin
commit_index
window_tokens
text_revision
route_revision
observation_revision
```

调用LDF前必须与当前state精确匹配。这样rolling后shape相同但时间坐标过期的condition不能被静默复用。

## 8. Session级CFG

共享LDF权重上的默认CFG值不能被某个Web session原地修改。`GuidanceConfig`由每个`InferenceSession`持有，并显式传给`stream_generate_step()`：

```text
mode
scale_text
scale_constraint
scale_joint
```

LDF参数、generated clean root、committed history和VAE state不是CFG对象。多个session可共享eval-mode模型，但route、text、guidance、RNG、LDF state和decoder state必须隔离。

## 9. 原子commit顺序

`InferenceSession.generate_step()`固定执行：

1. 读取当前committed session state；
2. 编译并校验当前window condition；
3. LDF返回candidate state与一个committed Hybrid token；
4. root反归一化并恢复到world；
5. 使用上一committed world root派生backward local-root4；
6. BodyVAE返回candidate decoder state与四帧body；
7. 校验shape、finite、heading manifold和唯一commit index；
8. 如本step发生window roll，对candidate LDF state执行translation rebase；
9. 最后一次性交换LDF state、decoder state、origin和previous root。

第3至第8步任何一步失败，旧session state保持不变。正常step依赖out-of-place模型接口，不需要每token预先复制完整snapshot。

## 10. Snapshot语义

`InferenceSnapshot`是同一已加载模型上的in-memory deterministic snapshot，包含：

- LDF snapshot及RNG；
- cloned `VAEDecoderState`；
- origin与previous world root；
- text intervals/revision；
- current world route/revision；
- sparse root observations/revision；
- session guidance。

Text embedding cache不进入snapshot，因为它是由文本和固定encoder重新计算的派生数据。完整输出帧也不进入snapshot；调用方根据snapshot的commit index截断自己的输出记录。跨进程持久化及checkpoint identity校验留给Web部署层。

## 11. 在线更新限制

text/route update从当前`commit_index`影响所有未提交token；committed past永远不变。三角调度意味着最近的未提交token可能已经接近clean，对突发转向的响应弱于pure-noise frontier。这是control latency，不通过混淆`seq_len`、future horizon或root replacement隐藏。

后续scheduled training/self-forcing必须模拟active band中途发生条件切换。runtime trace至少记录update revision、commit token、window origin/epoch和是否rebase，以便区分模型响应问题与condition indexing错误。

## 12. 文件结构与未完成项

```text
utils/inference/
├── geometry.py       # 严格无状态XZ route几何
├── route.py          # RoutePlan与输入参考
├── text.py           # TextTimeline与embedding cache
├── condition.py      # world条件到stamped LDFCondition
└── session.py        # 原子LDF/VAE commit事务
```

尚未完成：

- 正式LDF checkpoint与text encoder loader；
- Web视频编码与跨进程输出持久化；
- 真实在线route/text update质量评测；
- scheduled condition-update训练；
- 跨进程snapshot格式和checkpoint identity绑定；
- 如确有需求，再设计scheduled route timeline及pose/body observation，不恢复旧runtime兼容层。
