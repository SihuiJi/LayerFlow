#! /bin/bash
export CUDA_VISIBLE_DEVICES=0
echo "RUN on $(hostname), CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

run_cmd="torchrun --standalone --nproc_per_node=1 test_video_rgb.py --base configs/cogvideox_2b_stage2_seg_rgb.yaml configs/sft_stage2_seg_rgb_test.yaml --seed $RANDOM"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"
