name=icl_imagenet1k2way_2shot_baseline

sbatch -p atshare_a100_40g  --quotatype=auto \
    --comment "wbsR-SC230999.001.02" \
    --job-name=$name \
    -o ./output/$name-%j.out \
    -n 1 -N 1 --gres=gpu:8 -c 64 \
    accelerate launch --num_processes 8 \
    --main_process_port 23789 \
    mllm/pipeline/finetune.py config/icl_imagenet1k2way.py \
    --tf32=False --bf16=False --fp16=True --overwrite_output_dir \
    --cfg-options data_args.use_icl=True \
    --cfg-options model_args.model_name_or_path='/mnt/lustrenew/share_data/taiyan/shikra/shikra-checkpoint-44000' \
    --cfg-options data_args.shot=2 \
    --cfg-options training_args.per_device_train_batch_size=1 \
    --cfg-options training_args.save_steps=700 \
    --cfg-options training_args.num_train_epochs=5 \
    --cfg-options training_args.output_dir="/mnt/lustrenew/share_data/taiyan/icl/$name" \
    --cfg-options training_args.ceph_dir="ty-sdc:s3://ICL/checkpoint/taiyan/$name"