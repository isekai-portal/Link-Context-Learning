#!/bin/bash
export CUDA_VISIBLE_DEVICES=${1:-'7'}
server_port=${2:-'20488'}
model_path=${3:-'/home/taiyan/ckpt/okapis/demo_mix_1w'}


rm -r gradio_cached_examples
python -u ./demo/demo.py \
    --model_path $model_path \
    --server_name 0.0.0.0 \
    --server_port $server_port 
    # --load_in_8bit