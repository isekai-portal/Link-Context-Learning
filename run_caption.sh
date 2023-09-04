

export PATH=/mnt/cache/share/gcc/gcc-7.3.0/bin:$PATH
export LD_LIBRARY_PATH=/mnt/cache/share/gcc/gcc-7.3.0/lib64:/mnt/cache/share/gcc/mpc-0.8.1/lib:$LD_LIBRARY_PATH

export CUDA_HOME=/mnt/cache/share/cuda-11.7
export PATH=/mnt/cache/share/cuda-11.7/bin::$PATH
export LD_LIBRARY_PATH=/mnt/cache/share/cuda-11.7/lib64:$LD_LIBRARY_PATH

name=train_caption_pretrain_new2

config=config/caption_pretrain.py
pretrained=/mnt/lustre/share_data/chenkeqin/ckpt/llava7b_pretrain8_fc/checkpoint-166000/
#/mnt/lustre/fanweichen2/Research/MLLM/Llama7b/llama-7b-hf
output_dir=/mnt/cache/fanweichen2/Code/unify_mllm/result/$name
ceph_dir=ty-sdc:s3://ICL/checkpoint/fanweichen2/$name

bs=4
shot=8
epochs=100
save_steps=200
use_icl=False

sbatch -p mm_v100_32g  --quotatype=auto \
    --comment "wbsR-SC230999.001.02" \
    --job-name=$name \
    --phx-priority P0 \
    --preempt \
    --nodes 4 \
    -x SH-IDC1-10-142-4-22 \
    launcher_intelmpi.sh mllm/pipeline/finetune.py $config\
    --tf32=False --bf16=False --fp16=True --overwrite_output_dir \
    --cfg-options data_args.use_icl=$use_icl \
    --cfg-options model_args.model_name_or_path=$pretrained \
    --cfg-options data_args.shot=$shot \
    --cfg-options training_args.per_device_train_batch_size=$bs \
    --cfg-options training_args.save_steps=$save_steps \
    --cfg-options training_args.num_train_epochs=$epochs \
    --cfg-options training_args.output_dir=$output_dir \
    --cfg-options training_args.ceph_dir=$ceph_dir \
    #--cfg-options model_args.freeze_mm_projector=False \
    #--cfg-options model_args.freeze_backbone=True \
    #--cfg-options training_args.save_strategy="steps" \ 

