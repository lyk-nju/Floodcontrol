# Floodcontrol Web Runtime

状态：`BLOCKED_ON_STRICT4_VAE`。

当前仓库只发布新版Hybrid LDF模型核心。旧的full-motion VAE、ControlNet和外置root planner已经删除，而strict `4 frames / token` body VAE、commit-time decoder事务与Web runtime尚未接线。

因此本目录保留HTTP/UI边界和显式迁移守卫，但不能启动真实动作生成。`web_demo.model_manager.ModelManager`会在加载legacy checkpoint之前抛出带有`BLOCKED_ON_STRICT4_VAE`的错误。

恢复Web生成需要依次完成：

1. strict-4 body VAE与causal decoder state；
2. explicit root + body latent数据协议和真实statistics；
3. runtime condition compiler与`LDF.stream_generate_step()`接线；
4. Hybrid commit与VAE decoder state的原子snapshot/restore；
5. full/cached decode parity和端到端stream测试。

在这些条件满足前，不应把静态前端或旧server脚本视为可用生成demo。
