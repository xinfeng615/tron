# openvla-zero

## TRON2 双臂真机部署

从 LeRobot 数据转换、baseline/Residual 训练、LoRA 合并、WebSocket 服务端、影子推理到真机执行的
完整中文流程见 [`experiments/robot/tron2/README.md`](experiments/robot/tron2/README.md)。

## MetaWorld M6 workflow

The M6 suite uses these detailed task instructions:

- `hammer-v3`: `hammer a screw into the wall`
- `coffee-pull-v3`: `pull a mug from a coffee machine`
- `pick-out-of-hole-v3`: `pick up a puck from a hole`
- `box-close-v3`: `grasp the cover and close the box with it`
- `peg-insert-side-v3`: `insert a peg sideways`
- `basketball-v3`: `dunk the basketball into the basket`

Main commands:

```bash
python collect_metaworld_m6_data.py
python metaworld_m6_50e_dataset_builder.py
bash finetune_m6.sh
bash eval_m6.sh
bash eval_m6_zeroshot.sh
```

The current suite evaluates in the order shown above. Dataset version 1.1.0
replaces `pick-place-wall-v3` with `hammer-v3`; the builder reads only these six
task files, so a stale `pick-place-wall-v3.hdf5` is ignored.

The RLDS dataset name is `metaworld_m6_50e`. W&B projects default to
`m6-finetune` and `m6-eval`.

## M6 residual / chunk experiments

The three innovation variants reuse the same M6 HDF5 and RLDS dataset. Action
chunk targets are created online by the RLDS trajectory window, so data does not
need to be collected or converted again for each variant.

```bash
# 1. Collect and convert the shared six-task dataset.
python collect_metaworld_m6_data.py
python metaworld_m6_50e_dataset_builder.py

# 2. Train one of the three variants.
bash finetune_m6_residual.sh
bash finetune_m6_chunk.sh
bash finetune_m6_residual_chunk.sh
```

Training saves LoRA adapters under each run's `checkpoints/step-*` directory.
No model merge is performed inside the training process. A checkpoint is saved
at every `save_steps` interval and at the final optimizer step, even when the
final step is not divisible by `save_steps`. Merge only the checkpoint selected
for evaluation; for example, for the residual +
chunk step-20000 checkpoint:

```bash
RUN_DIR=/root/autodl-tmp/openvla/output_m6/openvla-7b+metaworld_m6_50e+b16+lr-0.0005+lora-r32+dropout-0.0+residual-0,1,2,6-w1.0+chunk-h4-w1.0--image_aug

python vla-scripts/merge_lora_checkpoint.py \
  --base_model /root/autodl-tmp/openvla/openvla-7b \
  --adapter_checkpoint "$RUN_DIR/checkpoints/step-20000" \
  --output_dir "$RUN_DIR/merged-step-20000" \
  --seed 7
```

Then run the matching evaluator:

```bash
bash eval_m6_residual.sh
bash eval_m6_chunk.sh
bash eval_m6_residual_chunk.sh
```

The evaluation scripts point to `merged-step-20000`. Change that path when
evaluating another saved step. Chunk variants use horizon 4 and temporal
ensembling with decay 0.8. As in the basketball innovation experiments, the
residual dimensions are `0,1,2,6`, residual scale is 0.25, chunk scale is 0.75,
and gripper binarization is disabled during innovation evaluation.

All M6 training launchers explicitly use `seed=7`, deterministic PyTorch/CUDA
kernels, deterministic TensorFlow operations, seeded RLDS sampling and
frame-level shuffling, and deterministic image augmentation. Unseeded TFDS file
shuffling is disabled for seeded runs. M6 evaluation passes the same seed into
the MetaWorld benchmark and environment. With the same dataset, checkpoint
step, software versions, GPU type, and evaluation settings, repeated runs should
therefore produce identical or very close results. A nondeterministic PyTorch
kernel now raises an error instead of silently introducing run-to-run drift.
