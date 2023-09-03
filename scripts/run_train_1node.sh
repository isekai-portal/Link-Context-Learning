# name=concept_v3
name=train_debug

config=config/icl_imagenet1k_v3_train.py
pretrained=/mnt/lustrenew/share_data/taiyan/checkpoint/shikras/shikra-checkpoint-44000
output_dir=/mnt/lustrenew/share_data/taiyan/checkpoint/okapis/$name
ceph_dir=ty-sdc:s3://ICL/checkpoint/taiyan/$name

bs=1
shot=2
epochs=5
save_steps=700
use_icl=True

gpus=8
cpus=$(($gpus*8))
port=23789

sbatch -p atshare_a100_40g  --quotatype=auto \
    --comment "wbsR-SC230999.001.02" \
    --job-name=$name \
    -o ./output/$name-%j.out \
    -n 1 -N 1 --gres=gpu:$gpus -c $cpus \
    accelerate launch --num_processes $gpus \
    --main_process_port $port \
    mllm/pipeline/finetune.py $config\
    --tf32=False --bf16=False --fp16=True --overwrite_output_dir \
    --cfg-options data_args.use_icl=$use_icl \
    --cfg-options model_args.model_name_or_path=$pretrained \
    --cfg-options data_args.shot=$shot \
    --cfg-options training_args.per_device_train_batch_size=$bs \
    --cfg-options training_args.save_steps=$save_steps \
    --cfg-options training_args.num_train_epochs=$epochs \
    --cfg-options training_args.output_dir=$output_dir \
    --cfg-options training_args.ceph_dir=$ceph_dir