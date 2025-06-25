#! /bin/bash
export CUDA_VISIBLE_DEVICES=1
echo "RUN on $(hostname), CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

run_cmd="torchrun --standalone --nproc_per_node=1 test_video.py --base configs/cogvideox_2b_stage2_bg2fg_rgba.yaml configs/sft_stage2_bg2fg_rgba_test.yaml --seed $RANDOM"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"
