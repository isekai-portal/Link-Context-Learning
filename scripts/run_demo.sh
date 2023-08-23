#!/bin/bash

model_path=/mnt/lustrenew/share_data/liwei1/MLLM/model/kosmos-2/kosmos-2.pt

master_port=20399

CUDA_LAUNCH_BLOCKING=1 srun -p atshare_a100_40g -N 1 --gres=gpu:1 --comment "wbsR-SC230999.001.01" python -m torch.distributed.launch --master_port=$master_port --nproc_per_node=1 demo/gradio_app.py None \
    --task generation_obj \
    --path $model_path \
    --model-overrides "{'visual_pretrained': '',
            'dict_path':'data/dict.txt'}" \
    --dict-path 'data/dict.txt' \
    --required-batch-size-multiple 1 \
    --remove-bpe=sentencepiece \
    --max-len-b 500 \
    --add-bos-token \
    --beam 1 \
    --buffer-size 1 \
    --image-feature-length 64 \
    --locate-special-token 1 \
    --batch-size 1 \
    --nbest 1 \
    --no-repeat-ngram-size 3 \
    --location-bin-size 32
