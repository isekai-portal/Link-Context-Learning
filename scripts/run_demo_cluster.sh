#!/bin/bash

server_port=20488
model_path=/mnt/lustrenew/share_data/taiyan/checkpoint/okapis/demo0828

gpus=1
cpus=$(($gpus*8))

rm -r gradio_cached_examples
srun -p atshare_a100_40g  --quotatype=auto --comment "wbsR-SC230999.001.02" --job-name=icl_demo -n 1 -N 1 --gres=gpu:$gpus -c $cpus \
    python -u ./mllm/demo_lcl/demo.py \
    --model_path $model_path \
    --server_port $server_port 
    # --load_in_8bit