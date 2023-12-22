#!/usr/bin/env bash

# GMFlow with hierarchical matching refinement (1/8 + 1/4 features)

# number of gpus for training, please set according to your hardware
# can be trained on 4x 32G V100 or 4x 40GB A100 or 8x 16G V100 gpus
NUM_GPUS=4

# chairs
CHECKPOINT_DIR=debug/chairs-gmflow-scale2 && \
mkdir -p ${CHECKPOINT_DIR} && \
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=9989 main_flow.py \
--launcher pytorch \
--checkpoint_dir ${CHECKPOINT_DIR} \
--stage chairs \
--batch_size 16 \
--val_dataset chairs sintel kitti \
--lr 4e-4 \
--image_size 384 512 \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--with_speed_metric \
--val_freq 10000 \
--save_ckpt_freq 10000 \
--num_steps 100000 \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log