# OpenVLA-M6 在 TRON2 双臂真机上的完整执行流程

本文覆盖 LeRobot 数据转换、OpenVLA baseline/Residual/Chunk 训练、LoRA 合并、WebSocket 服务端、
影子推理和真机执行。当前 OpenVLA 部署与 OpenPI 的 TRON2 观测/动作语义保持一致，但模型结构仍是
OpenVLA，不使用 Pi0 的 RTC 推理算法。

## 1. OpenPI 对齐契约

每个策略时刻使用以下输入和输出：

| 类型 | TRON2/OpenPI 键 | OpenVLA RLDS 键 | 形状 |
|---|---|---|---|
| 头部 RGB | `images.cam_high` | `image_primary` | `480x640x3` |
| 左腕 RGB | `images.cam_left_wrist` | `image_left_wrist` / `image_secondary` | `480x640x3` |
| 右腕 RGB | `images.cam_right_wrist` | `image_right_wrist` / `image_wrist` | `480x640x3` |
| 双臂状态 | `state` | `observation.state` / `proprio` | `16` |
| 语言 | `prompt` | `language_instruction` | 字符串 |
| 绝对动作 | `actions` | `action` | `[H,16]` |

16 维顺序固定为：

```text
[左臂7关节, 左夹爪1, 右臂7关节, 右夹爪1]
```

原始 LeRobot state/action 是 18 维，最后两维为头部 pitch/yaw。转换器只取前 16 维，因此当前策略不控制
头部。三路图像按 `cam_high -> cam_left_wrist -> cam_right_wrist` 的固定顺序编码；训练和部署不得换序。

`state_encoding` 只是 OXE 数据集的语义描述，物化配置时会被删除；真正控制状态是否进入流水线的是
`state_obs_keys=["state"]` 和训练参数 `load_proprio=True`。TRON2 训练脚本已经启用该参数，因此 loader 会把
16 维 state 归一化后放入 `observation.proprio`，模型再把它投影为一个 proprio token。

## 2. 目录约定

训练脚本沿用原 M6 的固定 AutoDL 路径格式：

```text
/root/autodl-tmp/openvla/                         # openvla-m6-residual 项目
/root/autodl-tmp/openvla/openvla-7b/              # OpenVLA-7B 基础权重
/root/autodl-tmp/lerobot_2026-07-17_16-12-18/    # 原始 TRON2 LeRobot 数据
/root/autodl-tmp/tensorflow_datasets/             # 转换后的 TFDS/RLDS 根目录
/root/autodl-tmp/openvla/output_tron2/             # 训练输出
/root/autodl-tmp/openvla/adapter-tmp_tron2/        # 兼容保留目录
```

如果服务器目录不同，直接统一修改四个 `finetune_tron2_*.sh` 和下文命令中的绝对路径。

## 3. 准备训练环境

进入已经能运行原 M6 训练的环境，然后安装数据转换依赖：

```bash
cd /root/autodl-tmp/openvla
pip install av pyarrow tensorflow tensorflow-datasets

python -c "import av, pyarrow, tensorflow, tensorflow_datasets; print('dataset dependencies ok')"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

转换器使用 PyAV 自带的 FFmpeg 软件解码 AV1，不要求宿主机提供 AV1 硬件解码能力。

## 4. 转换三相机 RLDS 数据

```bash
cd /root/autodl-tmp/openvla

python tron2_lerobot_dataset_builder.py \
  --dataset-root /root/autodl-tmp/lerobot_2026-07-17_16-12-18 \
  --output-dir /root/autodl-tmp/tensorflow_datasets
```

转换器逐 episode 流式读取三个 MP4，校验每路视频帧数与 Parquet 行数一致，读取任务语言，并将 18 维
state/action 截为前 16 维，并将三路图像逐帧编码为 JPEG。当前三相机 schema 版本是 `1.0.0`。
如果输出目录中已有同版本的旧数据或未完成的转换结果，应先使用一个空的 TFDS 输出目录重新转换。

检查结果：

```bash
test -d /root/autodl-tmp/tensorflow_datasets/tron2_lerobot/1.0.0
find /root/autodl-tmp/tensorflow_datasets/tron2_lerobot/1.0.0 -maxdepth 1 -type f
```

视频缺帧、多帧、非有限 state/action 或维度错误都会中止转换。必须先修复数据，不能跳过校验。

## 5. 训练四组对照

四个脚本都使用三相机、16 维 proprio、16 维绝对动作、LoRA rank 32、seed 7 和有效批量 16。由于三路
视觉 token 显著增加显存，默认使用 `batch_size=1`、`grad_accumulation_steps=16`；修改批量时四组必须同步。

```bash
cd /root/autodl-tmp/openvla

# 原始 OpenVLA 头，不启用 M6 创新头
bash finetune_tron2_baseline.sh

# M6 Residual
bash finetune_tron2_residual.sh

# M6 Chunk，H=4
bash finetune_tron2_chunk.sh

# M6 Residual + Chunk，H=4
bash finetune_tron2_residual_chunk.sh
```

共同的关键参数为：

```text
camera_views=primary,secondary,wrist
use_proprio=True
proprio_dim=16
action_dim=16
max_steps=20000
save_steps=2000
```

Residual/Chunk 只学习 14 个臂关节维度 `0..6,8..14`，不修改左右夹爪 `7,15`。Residual scale 为
`0.25`，Chunk scale 为 `0.75`，Chunk horizon 为 `4`。

## 6. 合并 LoRA checkpoint

训练 checkpoint 是 adapter，服务端必须使用合并后的目录。先找到某一实验的运行目录：

```bash
find /root/autodl-tmp/openvla/output_tron2 \
  -path '*/checkpoints/step-20000' -type d -print
```

对每个待评估实验执行：

```bash
RUN_DIR="/root/autodl-tmp/openvla/output_tron2/替换为实际运行目录"

python vla-scripts/merge_lora_checkpoint.py \
  --base_model /root/autodl-tmp/openvla/openvla-7b \
  --adapter_checkpoint "$RUN_DIR/checkpoints/step-20000" \
  --output_dir "$RUN_DIR/merged-step-20000" \
  --seed 7

test -f "$RUN_DIR/merged-step-20000/config.json"
test -f "$RUN_DIR/merged-step-20000/dataset_statistics.json"
```

合并脚本会同时合并 LoRA，并保留 `proprio_projector`、Residual head、Chunk head 中实际启用的模块。

## 7. 启动 OpenVLA 服务端

服务端必须从 `openvla-m6-residual` 项目根目录启动，以便注册本项目的自定义 OpenVLA 类：

```bash
cd /root/autodl-tmp/openvla
pip install "msgpack>=1.0.5" "websockets>=13"

python -m experiments.robot.tron2.serve_openvla \
  --model-path "$RUN_DIR/merged-step-20000" \
  --host 127.0.0.1 \
  --port 8000 \
  --default-prompt "Put the banana on the plate."
```

两台机器位于受控机器人局域网时，将 `--host` 改为 `0.0.0.0`，但不要把无认证、无 TLS 的端口暴露到
互联网。默认相机顺序已经是：

```text
cam_high cam_left_wrist cam_right_wrist
```

服务端启动时会拒绝单相机、无 proprio、非 16 维 state/action 的旧 checkpoint。它使用训练 checkpoint
中的 proprio q01/q99 对实时 state 做 `[-1,1]` 归一化，再执行动作预测和反归一化。服务端会先完成一次
三相机/state warmup，成功后才开放 WebSocket；调试时可用 `--skip-warmup` 跳过。

服务端 metadata 应包含：

```text
model_family=openvla-m6
camera_names=[cam_high, cam_left_wrist, cam_right_wrist]
state_dim=16
action_dim=16
action_horizon=1 或 4
rtc_enabled=false
```

## 8. 准备 TRON2 客户端

机器人侧使用 `tron2_openpi`，并让公开 `tron2_env` 与它保持同级：

```bash
cd /path/to/babana1/tron2_openpi
uv sync
uv pip install -e .
uv pip install -e "../tron2_env[bridge,openpi]"

cp configs/deploy/openvla_client.example.yaml \
   configs/deploy/openvla_client.local.yaml
```

至少修改：

- `client.policy_host`：GPU 服务端的机器人局域网 IP。
- `client.prompt`：与采集任务语义一致的英文指令。
- `robot.ip`、`bridge.host`：现场地址。
- `bridge.state_source`：现场使用 `bridge` 或 `legacy`。
- `robot.init_joints`：人工审核的 14 维安全初始姿态。
- `safety.joint_lower/joint_upper`：硬件批准的 14 维关节限位。

相机配置必须保留：

```yaml
camera:
  camera_names: [cam_high, cam_left_wrist, cam_right_wrist]
```

客户端使用与 OpenVLA 训练一致的 224x224 naive resize，再以 CHW uint8 发送三路图像；16 维 state 和 prompt
使用与 OpenPI 相同的顶层键发送。

## 9. 先做影子推理

```bash
cd /path/to/babana1/tron2_openpi

uv run python examples/tron2/openvla_client.py \
  --profile configs/deploy/openvla_client.local.yaml
```

不带 `--execute` 时不会调用 `env.reset()` 或 `env.step()`，但仍会连接机器人和相机。操作员仍需在场并准备
急停。至少检查：三路图像持续更新、metadata 契约通过、输出为 `[H,16]`、夹爪在 `[0,1]`、首帧关节
跳变未触发安全门、推理延迟低于配置上限。

## 10. 真机低风险执行

只有影子模式稳定、硬件限位已填写、机器人工作区清空、初始姿态确认且操作员持有急停时，才执行：

```bash
uv run python examples/tron2/openvla_client.py \
  --profile configs/deploy/openvla_client.local.yaml \
  --execute
```

第一轮保持：

```yaml
client:
  max_inferences: 20
  execution_horizon: 1
```

即使服务端返回 H=4 的 Chunk，也先只执行第一步。安全门会检查 observation 时间戳是否超龄或停滞、
动作形状、有限值、块长度、14 维硬件限位、相邻关节跳变和推理耗时，并把夹爪裁剪到配置范围。
`max_inference_ms` 同时作为 WebSocket 接收超时，不再无限等待服务端。安全检查失败会停止发送新动作。

## 11. 公平对照实验

Baseline、Residual、Chunk、Residual+Chunk 必须共享同一 RLDS 2.0 数据、基础权重、训练步数、有效批量、
学习率、LoRA rank、seed、三相机顺序、proprio、语言、初始姿态和客户端安全配置。先比较 baseline 与
Residual 以验证已在 MetaWorld 证明的创新，再单独分析 Chunk 带来的实时性和成功率变化。

建议记录成功率、完成时间、推理 P50/P95、最大关节跳变、安全拦截次数、人工急停次数，以及每个 Chunk
实际执行的步数。实验顺序应随机化或交替，避免环境漂移偏向某个模型。

## 12. 常见问题

### 服务端拒绝旧 checkpoint

旧模型没有 `num_image_views=3` 或 `proprio_enabled=true`。必须用 RLDS 2.0 和更新后的训练脚本重新训练、
重新合并，不能靠修改 `config.json` 伪造，因为旧权重中不存在训练后的 proprio projector。

### 缺少相机

检查 Bridge 输出键、YAML 的 `camera.camera_names` 和服务端 metadata。必须同时存在 `cam_high`、
`cam_left_wrist`、`cam_right_wrist`，且顺序一致。

### checkpoint 缺少 proprio 统计量

确认合并目录包含训练 checkpoint 对应的 `dataset_statistics.json`。没有 q01/q99 时服务端会拒绝推理，
不能直接把未归一化关节状态送入模型。

### 首帧被安全门拒绝

检查动作确为绝对关节目标、state/action 顺序一致、当前姿态位于训练分布内。不要直接放大
`max_arm_delta_rad` 绕过问题。

### AV1 视频无法解码

确认当前训练环境安装了 PyAV：`pip install av`。转换器不再通过 OpenCV 解码 AV1；如果仍看到
`Video ended before parquet row 0`，先执行 README 后面的 PyAV 单视频检查，确认上传的视频文件完整。

## 13. 当前边界

- OpenVLA 使用同步 WebSocket 推理，不支持 Pi0 专属 RTC/VJP guidance。
- 当前客户端默认顺序执行动作；尚未实现异步策略队列和 temporal ensemble。
- 软件安全门不等于硬件安全认证，真机实验仍依赖现场限位、急停和操作规程。
