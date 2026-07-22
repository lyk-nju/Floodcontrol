# Floodcontrol Web Runtime

状态：`BLOCKED_ON_LDF_CHECKPOINT`。

Web 层已经迁移到新版 Hybrid 推理协议：每个浏览器会话只持有一个
`utils.inference.InferenceSession`，LDF commit、BodyVAE causal decode、窗口
rolling 与 route/text revision 不再由 Web 重复维护。后台生产与 HTTP 传输都以
一个 latent token 对应的四帧 `WebMotionChunk` 为原子单位。

## 当前已经完成

- `WebRuntime` 管理共享只读模型、单活跃会话和进程级 GPU execution lock；
- `WebSession` 串行化生成、text/route/CFG 更新、pause/resume/reset；
- bounded chunk buffer 使用 backpressure，禁止静默丢帧；
- API 直接接受 typed XZ route，并支持 world/relative-to-actor 与 hold/release；
- 浏览器一次拉取四帧 chunk，再本地按 20 FPS 播放；
- 旧 `ModelManager`、`TrajectoryController`、root feedback、route delay/blend 已删除。

## 当前阻塞项

正式Root-x0/Body-velocity LDF尚未完成训练，因此仓库还没有可以冻结的LDF checkpoint
和checkpoint loader contract。`model_loader.load_model_bundle()`
会在尝试加载任何 legacy checkpoint 前明确抛出 `BLOCKED_ON_LDF_CHECKPOINT`。

完成正式 LDF 训练后，需要实现唯一的 `ModelBundle` loader，加载并校验：

1. LDF checkpoint与Root/Body prediction合同；
2. 自包含EMA BodyVAE checkpoint及其physical body/local-root buffers；
3. UMT5 text encoder/tokenizer；
4. FPS、latent dimension和checkpoint内容身份。

静态服务器可以启动并通过 `/api/status` 报告阻塞原因；在 loader 完成前，
`POST /api/sessions` 会返回 HTTP 503，而不会回退到旧推理路径。
