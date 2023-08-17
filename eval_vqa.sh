sbatch -p mm_v100_32g \
    -n 1 \
    -N 1 \
    --gres=gpu:8 \
    --phx-priority P0 \
    --preempt \
    -c 64 \
    --job-name=eval_mmv100 \
    -x SH-IDC1-10-142-4-22 \
    --comment "wbsR-SC230999.001.02" \
    accelerate launch --num_processes 8 --main_process_port 23781 mllm/pipeline/finetune.py \
        config/lcl_eval_vqa.py \
        --tf32=False --bf16=False --fp16=True \
        --cfg-options model_args.model_name_or_path=/mnt/lustre/fanweichen2/Research/MLLM/ckpt/checkpoint-2800 \
        --per_device_eval_batch_size 4 \
        --output_dir /mnt/cache/fanweichen2/Code/unify_mllm/result/output/mix/vqa_new/ \
        --cfg-options data_args.use_icl=False