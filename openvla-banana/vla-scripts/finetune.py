"""
finetune.py

用于参数高效微调 OpenVLA 模型的简单脚本，通过 HuggingFace AutoClasses 加载，
并使用 HuggingFace PEFT 库进行低秩自适应 (LoRA) 微调。

说明与基准测试:
    - 需要安装 PEFT (`pip install peft==0.11.1`)
    - LoRA 微调 (参考以下参数 -- 无量化, LoRA rank = 32, target_modules = all-linear):
        + 一张 48 GB GPU 可以容纳 Batch Size 为 12
        + 一张 80 GB GPU 可以容纳 Batch Size 为 24
da
运行方式:
    - [单节点多 GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [覆盖配置参数运行]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
"""

import os
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import draccus
import numpy as np

import torch
import torch.nn.functional as F
# === 开启 TF32 硬件加速 ===
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
import tensorflow as tf

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

OPENVLA_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "lm_head",
    "qkv",
    "proj",
    "fc1",
    "fc2",
    "fc3",
]


def parse_action_dims(action_dims: str, action_dim: int) -> List[int]:
    dims = [int(dim.strip()) for dim in action_dims.split(",") if dim.strip()]
    if not dims or any(dim < 0 or dim >= action_dim for dim in dims):
        raise ValueError(f"Action dimensions must be indices in [0, {action_dim - 1}], got: {dims}")
    if len(set(dims)) != len(dims):
        raise ValueError(f"Action dimensions must not contain duplicates, got: {dims}")
    return dims


def get_base_vla(model):
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()
    return model


def quantize_actions_to_bin_centers(actions: torch.Tensor, n_action_bins: int) -> torch.Tensor:
    bins = torch.linspace(-1.0, 1.0, n_action_bins, device=actions.device, dtype=torch.float32)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    discretized = torch.bucketize(actions.float().clamp(-1.0, 1.0), bins)
    discretized = torch.clamp(discretized - 1, min=0, max=bin_centers.shape[0] - 1)
    return bin_centers[discretized].to(actions.dtype)


def extract_action_features(last_hidden_state, labels, num_condition_tokens, action_token_begin_idx):
    aligned_hidden = last_hidden_state[:, num_condition_tokens:-1]
    aligned_labels = labels[:, 1:].to(aligned_hidden.device)
    action_mask = aligned_labels > action_token_begin_idx
    features = []
    for batch_idx in range(aligned_hidden.shape[0]):
        if action_mask[batch_idx].any():
            features.append(aligned_hidden[batch_idx][action_mask[batch_idx]].mean(dim=0))
        else:
            features.append(aligned_hidden[batch_idx, -1])
    return torch.stack(features, dim=0)

# 合理的默认设置
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# # === 工具函数 ===
# # fmt: off
# def create_vision_transform(vla: nn.Module, input_size: int) -> Callable[[Image.Image], torch.Tensor]:
#     """获取视觉编码器的图像预处理变换。"""
#     data_cfg = timm.data.resolve_model_data_config(vla.vision_backbone)
#     data_cfg["input_size"] = (3, input_size, input_size)
#     return timm.data.create_transform(
#         input_size=data_cfg["input_size"],
#         interpolation=data_cfg["interpolation"],
#         mean=data_cfg["mean"],
#         std=data_cfg["std"],
#         crop_pct=1.0,           # 设置为 1.0 以禁用裁剪
#         crop_mode="center",     # 默认裁剪模式 --> 当 `crop_pct == 1.0` 时无操作
#         is_training=False,      # 加载变换时禁用图像增强；图像增强由 RLDS dataloader 处理
#     )
#
# # fmt: on


def set_training_seed(seed: int, deterministic: bool = True) -> None:
    """Seed all training-side RNGs used by PyTorch, TensorFlow, NumPy, and Python."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        try:
            tf.config.experimental.enable_op_determinism()
        except (AttributeError, TypeError):
            pass


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"                            # OpenVLA 模型路径 (在 HuggingFace Hub 上)

    # 目录路径
    data_root_dir: Path = Path("datasets/open-x-embodiment")        # Open-X 数据集目录路径
    dataset_name: str = "metaworld_m6_50e"                           # Fine-tuning dataset name
    run_root_dir: Path = Path("runs")                               # 存储日志和检查点的目录路径
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # 保留兼容旧命令；训练中不再合并 LoRA

    # 微调参数
    batch_size: int = 16                                            # 微调批大小 (Batch Size)
    max_steps: int = 200_000                                        # 微调的最大步数
    save_steps: int = 5000                                          # 保存检查点的间隔步数
    learning_rate: float = 5e-4                                     # 微调学习率
    grad_accumulation_steps: int = 1                                # 梯度累加步数
    image_aug: bool = True                                          # 是否使用图像增强进行训练
    shuffle_buffer_size: int = 100_000                              # Dataloader 的洗牌缓冲区大小 (如果显存溢出可减小)
    seed: int = 7                                                   # Training seed for model init and data pipeline
    deterministic_training: bool = True                             # Prefer reproducible CUDA kernels when available
    save_latest_checkpoint_only: bool = False                       # 是否每次运行只保存一个检查点并
                                                                    #   不断覆盖最新的检查点
                                                                    #   (如果为 False，则保存所有检查点)

    # LoRA 参数
    use_lora: bool = True                                           # 是否使用 LoRA 微调
    lora_rank: int = 32                                             # LoRA 权重矩阵的秩 (Rank)
    lora_dropout: float = 0.0                                       # 应用于 LoRA 权重的 Dropout
    use_quantization: bool = False                                  # 是否对 VLA 进行 4-bit 量化以进行 LoRA 微调
                                                                    #   => 警告: 会减少显存占用但可能损害性能

    # 日志跟踪参数
    wandb_project: str = "m6-finetune"                     # W&B project name
    wandb_entity: str = "1469512941-"                          # 要记录到的实体名称
    run_id_note: Optional[str] = None                               # 供 Weights & Biases 日志记录的额外备注
    use_residual_action_head: bool = False
    action_dim: int = 7                                             # Explicit model action dimension; TRON2 uses 16
    residual_action_dims: str = "0,1,2,6"
    residual_action_scale: float = 0.25
    residual_loss_weight: float = 1.0
    use_action_chunking: bool = False
    action_chunk_size: int = 4
    action_chunk_dims: str = "0,1,2,6"
    action_chunk_scale: float = 0.25
    action_chunk_loss_weight: float = 1.0
    camera_views: str = "primary"                                # Comma-separated RLDS views in model token order
    use_proprio: bool = False
    proprio_dim: int = 0

    # fmt: on


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")
    if cfg.action_chunk_size < 1:
        raise ValueError(f"action_chunk_size must be >= 1, got {cfg.action_chunk_size}")
    if cfg.max_steps < 1 or cfg.save_steps < 1 or cfg.grad_accumulation_steps < 1:
        raise ValueError("max_steps, save_steps, and grad_accumulation_steps must all be >= 1.")
    camera_views = tuple(view.strip() for view in cfg.camera_views.split(",") if view.strip())
    valid_camera_views = {"primary", "secondary", "wrist"}
    if not camera_views or len(set(camera_views)) != len(camera_views):
        raise ValueError(f"camera_views must be a non-empty list without duplicates, got {camera_views}")
    invalid_camera_views = set(camera_views) - valid_camera_views
    if invalid_camera_views:
        raise ValueError(f"Unsupported camera_views: {sorted(invalid_camera_views)}")
    if cfg.use_proprio and cfg.proprio_dim < 1:
        raise ValueError(f"proprio_dim must be positive when use_proprio=True, got {cfg.proprio_dim}")

    # [验证] 确保 GPU 可用并设置设备 / 分布式上下文
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()
    set_training_seed(cfg.seed, cfg.deterministic_training)

    # 配置唯一的实验 ID 和日志目录
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if camera_views != ("primary",):
        exp_id += f"+views-{'-'.join(camera_views)}"
    if cfg.use_proprio:
        exp_id += f"+proprio-{cfg.proprio_dim}"
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.use_residual_action_head:
        exp_id += f"+residual-{cfg.residual_action_dims}-w{cfg.residual_loss_weight}"
    if cfg.use_action_chunking:
        exp_id += f"+chunk-h{cfg.action_chunk_size}-w{cfg.action_chunk_loss_weight}"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    # 开始 =>> 创建目录
    run_dir = cfg.run_root_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # 量化配置 =>> 仅在进行 LoRA 微调时使用
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # 注册 OpenVLA 模型到 HF Auto Classes (如果模型已经在 HF Hub 上则不需要这一步)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # 使用 HF AutoClasses 加载 OpenVLA 处理器和模型
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    if cfg.action_dim < 1:
        raise ValueError(f"action_dim must be positive, got {cfg.action_dim}")
    residual_action_dims = parse_action_dims(cfg.residual_action_dims, cfg.action_dim)
    action_chunk_dims = parse_action_dims(cfg.action_chunk_dims, cfg.action_dim)
    model_config = OpenVLAConfig.from_pretrained(cfg.vla_path, trust_remote_code=True)
    model_config.action_dim = cfg.action_dim
    model_config.residual_action_head_enabled = cfg.use_residual_action_head
    model_config.residual_action_dims = residual_action_dims
    model_config.residual_action_scale = cfg.residual_action_scale
    model_config.action_chunk_head_enabled = cfg.use_action_chunking
    model_config.action_chunk_size = cfg.action_chunk_size if cfg.use_action_chunking else 1
    model_config.action_chunk_dims = action_chunk_dims
    model_config.action_chunk_scale = cfg.action_chunk_scale
    model_config.num_image_views = len(camera_views)
    model_config.proprio_enabled = cfg.use_proprio
    model_config.proprio_dim = cfg.proprio_dim if cfg.use_proprio else None
    use_auxiliary_heads = cfg.use_residual_action_head or cfg.use_action_chunking
    use_new_modules = use_auxiliary_heads or cfg.use_proprio
    vla = OpenVLAForActionPrediction.from_pretrained(
        cfg.vla_path,
        config=model_config,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=not use_new_modules,
        trust_remote_code=True,
    )
    
    # ======== ✨ 新增这行代码以拯救 4090D 的 24G 显存 ✨ ========
    vla.gradient_checkpointing_enable()
    # ========================================================
    
    # 测试随机初始化 (已注释)
    # import torch.nn as nn
    # 
    # def randomize_weights(module):
    #     if isinstance(module, (nn.Linear, nn.Conv2d)):
    #         nn.init.xavier_uniform_(module.weight)
    #         if module.bias is not None:
    #             nn.init.zeros_(module.bias)
    #     elif isinstance(module, nn.Embedding):
    #         nn.init.normal_(module.weight, mean=0, std=1)
    #     elif isinstance(module, nn.LayerNorm):
    #         nn.init.ones_(module.weight)
    #         nn.init.zeros_(module.bias)

    # vla.apply(randomize_weights)

    # 设备放置 =>> 注意 BitsAndBytes 会自动处理量化训练的设备放置逻辑
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] 使用 PEFT `LoraConfig` 包装模型 =>> 默认情况下我们将 `target_modules` 设置为 `all-linear`
    if cfg.use_lora:
        modules_to_save = []
        if cfg.use_residual_action_head:
            modules_to_save.append("action_residual_head")
        if cfg.use_action_chunking:
            modules_to_save.append("action_chunk_head")
        if cfg.use_proprio:
            modules_to_save.append("proprio_projector")
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules=OPENVLA_LORA_TARGET_MODULES if use_new_modules else "all-linear",
            init_lora_weights="gaussian",
            modules_to_save=modules_to_save or None,
        )
        vla = get_peft_model(vla, lora_config)
        for name, param in vla.named_parameters():
            if any(module_name in name for module_name in modules_to_save):
                param.requires_grad = True
        vla.print_trainable_parameters()

    # 将 VLA 包装在 PyTorch DDP (DistributedDataParallel) 包装器中以进行多 GPU 训练
    #vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)
    # 将 VLA 包装在 PyTorch DDP (DistributedDataParallel) 包装器中以进行多 GPU 训练
    # 添加 static_graph=True 以解决梯度检查点与 DDP 结合时的重复点名报错
    vla = DDP(vla, device_ids=[device_id], static_graph=True, gradient_as_bucket_view=True)
    
    # 创建优化器 =>> 注意：我们默认使用简单的恒定学习率！
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # 创建动作分词器 (Action Tokenizer)
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # 加载微调数据集 =>> 注意：我们默认使用遵循 Open X-Embodiment 格式的 RLDS 数据集。
    #   =>> 如果你想使用非 RLDS 数据集 (例如，标准的 PyTorch Dataset)，请参考下面注释掉的代码块。
    #   =>> 请注意，由于 RLDS 加载器会隐式地循环遍历数据，我们的训练代码中没有按 epoch 循环的逻辑；
    #       如果使用你自己的 Dataset，请确保在训练循环中添加适当的逻辑！
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # vla_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    # )
    # ---
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
        action_chunk_size=cfg.action_chunk_size if cfg.use_action_chunking else 1,
        camera_views=camera_views,
        load_proprio=cfg.use_proprio,
    )
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        seed=cfg.seed + distributed_state.process_index,
        future_action_window_size=(cfg.action_chunk_size - 1) if cfg.use_action_chunking else 0,
        camera_views=camera_views,
        load_proprio=cfg.use_proprio,
    )

    dataset_action_dims = {
        len(stats["action"]["q01"])
        for stats in vla_dataset.dataset_statistics.values()
    }
    if dataset_action_dims != {cfg.action_dim}:
        raise ValueError(
            "Training dataset action dimensions do not match the configured model head: "
            f"dataset={sorted(dataset_action_dims)}, action_dim={cfg.action_dim}"
        )
    if cfg.use_proprio:
        dataset_proprio_dims = {
            len(stats["proprio"]["q01"])
            for stats in vla_dataset.dataset_statistics.values()
        }
        if dataset_proprio_dims != {cfg.proprio_dim}:
            raise ValueError(
                "Training dataset proprio dimensions do not match the configured model input: "
                f"dataset={sorted(dataset_proprio_dims)}, proprio_dim={cfg.proprio_dim}"
            )

    # [重要] 保存数据集统计信息 =>> 用于在推理时对输出动作进行反归一化！
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # 创建数据整理器 (Collator) 和数据加载器 (DataLoader)
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # 重要 =>> 如果使用 RLDS，请设置为 0；因为 TFDS 会自己管理并行处理！
        generator=torch.Generator().manual_seed(cfg.seed + distributed_state.process_index),
    )

    # 初始化日志记录 =>> Weights & Biases (W&B)
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}", mode="online")

    # 用于存储最近训练指标的双端队列 (Deque)（用于计算梯度累加时的平滑指标）
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_residual_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_chunk_losses = deque(maxlen=cfg.grad_accumulation_steps)

    # 开始训练！
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            residual_loss = torch.zeros((), device=device_id)
            chunk_loss = torch.zeros((), device=device_id)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    proprio=batch["proprio"].to(device_id) if cfg.use_proprio else None,
                    labels=batch["labels"].to(device_id),
                    output_last_hidden_state=use_auxiliary_heads,
                )
                loss = output.loss

                if use_auxiliary_heads:
                    base_vla = get_base_vla(vla)
                    num_patches = base_vla.vision_backbone.featurizer.patch_embed.num_patches
                    num_condition_tokens = num_patches * len(camera_views) + int(cfg.use_proprio)
                    action_features = extract_action_features(
                        output.last_hidden_state,
                        batch["labels"].to(device_id),
                        num_condition_tokens,
                        action_tokenizer.action_token_begin_idx,
                    )
                    if cfg.use_residual_action_head:
                        residual_dims = torch.tensor(residual_action_dims, dtype=torch.long, device=device_id)
                        gt_actions = batch["actions"].to(device_id)
                        quantized_actions = quantize_actions_to_bin_centers(gt_actions, base_vla.config.n_action_bins)
                        residual_target = gt_actions - quantized_actions
                        residual_pred = base_vla.action_residual_head(action_features).float()
                        residual_pred = torch.tanh(residual_pred) * cfg.residual_action_scale
                        residual_loss = F.l1_loss(
                            residual_pred[:, residual_dims], residual_target[:, residual_dims].float()
                        )
                        loss = loss + cfg.residual_loss_weight * residual_loss

                    if cfg.use_action_chunking:
                        chunk_dims = torch.tensor(action_chunk_dims, dtype=torch.long, device=device_id)
                        gt_chunk = batch["action_chunks"].to(device_id)
                        gt_actions = batch["actions"].to(device_id)
                        quantized_actions = quantize_actions_to_bin_centers(gt_actions, base_vla.config.n_action_bins)
                        chunk_anchor = gt_actions if cfg.use_residual_action_head else quantized_actions
                        chunk_target = gt_chunk - chunk_anchor[:, None, :]
                        chunk_pred = base_vla.action_chunk_head(action_features).float()
                        chunk_pred = torch.tanh(chunk_pred) * cfg.action_chunk_scale
                        chunk_pred = chunk_pred.view(gt_chunk.shape[0], cfg.action_chunk_size, gt_chunk.shape[-1])
                        chunk_loss = F.l1_loss(
                            chunk_pred[:, :, chunk_dims], chunk_target[:, :, chunk_dims].float()
                        )
                        loss = loss + cfg.action_chunk_loss_weight * chunk_loss

            # 标准化 loss 以适配梯度累加步数
            normalized_loss = loss / cfg.grad_accumulation_steps

            # 反向传播
            normalized_loss.backward()

            # 计算准确率和 L1 Loss 用于日志记录
            metric_vla = get_base_vla(vla)
            num_condition_tokens = (
                metric_vla.vision_backbone.featurizer.patch_embed.num_patches * len(camera_views)
                + int(cfg.use_proprio)
            )
            action_logits = output.logits[:, num_condition_tokens:-1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx

            # 计算动作准确率
            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()

            # 计算预测（连续）动作的 L1 Loss
            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
            )
            action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

            # 存储最近的训练指标
            recent_losses.append(loss.item())
            recent_action_accuracies.append(action_accuracy.item())
            recent_l1_losses.append(action_l1_loss.item())
            recent_residual_losses.append(residual_loss.item())
            recent_chunk_losses.append(chunk_loss.item())

            # 计算梯度步数索引
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
            optimizer_step_idx = (batch_idx + 1) // cfg.grad_accumulation_steps

            # 计算平滑的训练指标
            #   =>> 当不使用梯度累加时，等于当前步的指标
            #   =>> 否则，等于用于梯度累加的各个微批次 (micro-batches) 指标的平均值
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
            smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)
            smoothened_residual_loss = sum(recent_residual_losses) / len(recent_residual_losses)
            smoothened_chunk_loss = sum(recent_chunk_losses) / len(recent_chunk_losses)

            # 将指标推送到 W&B (每 10 个梯度步推送一次)
            if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                wandb.log(
                    {
                        "train_loss": smoothened_loss,
                        "action_accuracy": smoothened_action_accuracy,
                        "l1_loss": smoothened_l1_loss,
                        "residual_loss": smoothened_residual_loss,
                        "chunk_loss": smoothened_chunk_loss,
                    },
                    step=gradient_step_idx,
                )

            # 优化器步进 (更新权重)
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                progress.update()

            # 保存检查点：LoRA 训练期间只保存 adapter，避免在训练进程中额外加载并合并 7B base model。
            if (
                (batch_idx + 1) % cfg.grad_accumulation_steps == 0
                and optimizer_step_idx > 0
                and (optimizer_step_idx % cfg.save_steps == 0 or optimizer_step_idx == cfg.max_steps)
            ):
                if distributed_state.is_main_process:
                    print(f"Saving Model Checkpoint for Step {optimizer_step_idx}")

                    if cfg.save_latest_checkpoint_only:
                        checkpoint_dir = run_dir / "checkpoints" / "latest"
                    else:
                        checkpoint_dir = run_dir / "checkpoints" / f"step-{optimizer_step_idx}"
                    os.makedirs(checkpoint_dir, exist_ok=True)

                    # LoRA: save adapter + processor/statistics needed by the standalone merge script.
                    # Full fine-tuning: save the model directly, preserving the old non-LoRA behavior.
                    processor.save_pretrained(checkpoint_dir)
                    save_dataset_statistics(vla_dataset.dataset_statistics, checkpoint_dir)
                    vla.module.save_pretrained(checkpoint_dir)
                    if cfg.use_lora and not (checkpoint_dir / "adapter_config.json").exists():
                        raise RuntimeError(
                            f"Expected an adapter-only LoRA checkpoint at {checkpoint_dir}, "
                            "but adapter_config.json was not created."
                        )
                    model_config.save_pretrained(checkpoint_dir)

                    print(f"Saved Checkpoint for Step {optimizer_step_idx} at: {checkpoint_dir}")

                # 阻塞等待主进程完成检查点保存
                dist.barrier()

            # 达到最大步数时停止训练
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0 and optimizer_step_idx == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()
