sbatch -p atshare_a100_40g  --quotatype=auto \
    --comment "wbsR-SC230999.001.02" \
    --job-name=icl_v3_2shot \
    -o ./output/slurm-%j.out \
    -n 1 -N 1 --gres=gpu:8 -c 64 \
    accelerate launch --num_processes 8 \
    --main_process_port 23789 \
    mllm/pipeline/finetune.py config/icl_imagenet.py \
    --tf32=False --bf16=False --fp16=True --overwrite_output_dir \
    --cfg-options data_args.use_icl=True \
    --cfg-options model_args.model_name_or_path='/mnt/lustre/share_data/chenkeqin/dummy_exp_unify_mllm/llava_pretrain_final19/checkpoint-44000' \
    --cfg-options data_args.shot=2 \
    --cfg-options training_args.per_device_train_batch_size=1 \
    --cfg-options training_args.save_steps=700 \
    --cfg-options training_args.output_dir='/mnt/lustrenew/share_data/taiyan/shikra/icl_imagenet1k_v3_2shot'