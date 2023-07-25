# name=test100_eval
name=eval_debug

config=config/icl_imagenet1k_v3_eval.py
pretrained=/mnt/lustrenew/share_data/taiyan/checkpoint/okapis/concept_v3/checkpoint-2100
output_dir=/mnt/lustrenew/share_data/taiyan/checkpoint/okapis/concept_v3/eval

bs=1
shot=2
use_icl=True

gpus=4
cpus=$(($gpus*8))
port=23789

sbatch -p atshare_a100_40g  --quotatype=auto \
    --comment "wbsR-SC230999.001.02" \
    --job-name=$name \
    -o ./output/$name-%j.out \
    -n 1 -N 1 --gres=gpu:$gpus -c $cpus \
    accelerate launch --num_processes $gpus \
    --main_process_port $port \
    mllm/pipeline/finetune.py $config \
    --tf32=False --bf16=False --fp16=True \
    --cfg-options model_args.model_name_or_path=$pretrained \
    --per_device_eval_batch_size $bs \
    --output_dir $output_dir \
    --cfg-options data_args.use_icl=$use_icl \
    --cfg-options data_args.shot=$shot
