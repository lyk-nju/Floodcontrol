# Root5 / Body259 数据管线

## 1. 数据层边界

正式协议固定为20 FPS和四帧一token。HumanML3D与BABEL的当前源都必须是HumanML-style 263D，并调用同一个`tools.convert_motion_263_to_259`转换器；BABEL不得维护近似副本。

```text
HumanML3D/new_joint_vecs or BABEL_streamed/motions
  -> common HumanML263 recovery/conversion
  -> *_motion_local/{split,all,artifacts,texts}
  -> Dataset完整sample
  -> VAE/LDF task collator
```

处理后目录同构：

```text
HumanML3D_motion_local/ or BABEL_motion_local/
├── train.txt
├── val.txt
├── test.txt          # 源存在真实test时
├── all.txt
├── artifacts/<sample>.npz
└── texts/<sample>.txt
```

NPZ只保存：

```text
root_motion                    float32 [F,5]
body_motion                    float32 [F,259]
body_feature_valid_mask        bool    [F,259]
```

`F>=4`且`F%4==0`。预处理丢弃不足四帧的尾部，不补伪pose。artifact必须finite，Root heading必须为单位向量；旧Body265或错误dtype不会被静默转换。

Dataset只加载完整序列并返回：

```text
dataset, name
root_motion, body_motion, body_feature_valid_mask
text_data[{text,tokens,start_frame,end_frame}]
```

公开source名称固定为`HumanML3D`/`BABEL`，不依赖目录名。Dataset不做crop、augmentation、translation rebase、context、previous-root或padding。

## 2. 263D转换顺序

公共转换器完成：

```text
recover HumanML canonical root C and world root XYZ
recover direct world joint positions
compose HumanML IK parent-local rotations into cumulative A
derive physical root yaw R=C^T
construct heading-frame cumulative B=R^T A
recompute current-heading-local backward velocity in m/s
pack Root5 + Body259 + validity mask
```

Body position减去完整root XYZ。cold-start的Body velocity为零且对应66维mask为false；其余连续/rotation/contact feature有效。Body中的root velocity不拥有最终root运动。

HumanML/BABEL的caption文件原样复制。`all.txt`严格等于已发布正式split的唯一sample ID并集；T5预编码从`all.txt`覆盖全部caption，而不是只读train/val。BABEL任意frame文本区间在LDF collator中按四帧token最大重叠编译，不要求原始边界token-aligned。

## 3. Training collator

`utils/training/vae/data.py`负责四帧对齐crop、previous-root、batch右padding、translation rebase和random world yaw：

```text
translation rebase -> only Root5 XYZ
random yaw         -> only Root5 XZ + heading
Body259            -> unchanged
```

这是因为Body259已对世界XYZ平移和统一global yaw严格不变。VAE validation取确定性前缀且关闭随机增强。

`utils/training/ldf/data.py`最多保留50 tokens/200 frames的parent，并携带最多24个真实VAE encoder context token。不存在cold-start左侧假零历史。窗口层逐样本采样history/active/rollout/future，active固定5 tokens；future XZ只作为condition，不把真实future motion送入模型。

LDF random yaw和translation anchor也只修改Root5及由Root派生的XZ constraints/previous-root；Body259与body mask必须逐元素保持不变。

## 4. Statistics与latent

仅生成HumanML train split的VAE physical statistics：

```text
body_cont_mean/std  [255]
local_root_mean/std [4]
```

HumanML+BABEL VAE复用HumanML statistics，不生成联合motion statistics。不生成root statistics、latent statistics或逐样本latent cache。

VAE训练从NPZ读取physical statistics，checkpoint保存四个buffer。LDF启动从VAE checkpoint恢复EMA参数和buffers；在线`tokenize_window()`直接得到raw deterministic posterior `mu`。

## 5. 一键资产入口

```bash
python -m tools.prepare_training_assets pre-vae \
  --raw-data-root /path/to/raw_data \
  --deps-root /path/to/deps \
  --workers 16 \
  --t5-devices 0,1,2,3
```

`pre-vae`一次完成：

1. 发布`HumanML3D_motion_local`；
2. 发布`BABEL_motion_local`；
3. 仅用HumanML train计算VAE statistics；
4. 复用已有T5表并补齐缺失caption，或从`all.txt`新建；
5. 全量验证每个split ID的motion/text、全部artifact字段/shape/dtype/finite/heading以及T5 caption覆盖率。

重复执行会跳过字段、shape、dtype和数值合同均正确的artifact；格式错误或旧Body265 artifact会被重建。输出使用原子替换，不将半成品当作完成阶段。

VAE训练完成后：

```bash
python -m tools.prepare_training_assets verify \
  --raw-data-root /path/to/raw_data \
  --deps-root /path/to/deps \
  --vae-checkpoint /path/to/new_body259_vae.ckpt
```

`verify`用公共EMA loader验证自包含VAE并检查LDF启动资产。旧Body265 checkpoint因网络输入/输出shape不符明确失败。

## 6. 发布验收

- 世界XYZ平移后Body259在`1e-6`内不变；
- 0/45/90/180度和随机世界yaw后Body259在`1e-5`内不变；
- Root5+Body position恢复world joints；
- `A=RB`、rotation matrix正交且det约为1；
- backward velocity严格等于当前heading下相邻pose差分乘20；
- cold velocity全零且mask无效；
- HumanML与BABEL对相同263D输入产生完全相同tensor；
- `263 -> Root5/Body259 -> 263`可观测字段与T2M feature漂移受控；
- direct/FK差异不显著超过源HumanML自身基线；
- Dataset、VAE collator和LDF collator真实样本smoke通过。

Root5/Body259已经完成的单样本闭环、32样本T2M冒烟、1450条完整val round-trip、
固定/随机yaw旋转FID及tail策略对照的协议和数值，统一记录在
[`02_VAE_AND_BODY_REPRESENTATION.md`第8节](02_VAE_AND_BODY_REPRESENTATION.md#8-root5--body259-表征实验记录)。
这些结果验证数据表示与evaluator adapter，不替代新VAE的MPJPE/FK重构验收。
