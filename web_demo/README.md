# Floodcontrol Web Runtime

状态：`BLOCKED_ON_STRICT4_VAE`。

当前仓库已经发布Hybrid LDF与strict `4 frames / token` BodyVAE模型核心。旧的full-motion VAE、ControlNet和外置root planner已经删除；原生rotations数据、正式VAE/latent artifacts、commit-time decoder事务与Web runtime尚未接线。

因此本目录保留HTTP/UI边界和显式迁移守卫，但不能启动真实动作生成。`web_demo.model_manager.ModelManager`会在加载legacy checkpoint之前抛出带有`BLOCKED_ON_STRICT4_VAE`的错误。

恢复Web生成需要依次完成：

1. 从原生SMPL rotations生成explicit root/body265与真实statistics；
2. 训练strict4 BodyVAE并生成带checkpoint hash的latent artifacts；
3. runtime condition compiler与`LDF.stream_generate_step()`接线；
4. Hybrid commit与VAE decoder state的原子snapshot/restore；
5. full/cached decode parity和端到端stream测试。

在这些条件满足前，不应把静态前端或旧server脚本视为可用生成demo。
