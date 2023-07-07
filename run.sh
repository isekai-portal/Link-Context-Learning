export PATH=/mnt/cache/share/gcc/gcc-7.3.0/bin:$PATH
export LD_LIBRARY_PATH=/mnt/cache/share/gcc/gcc-7.3.0/lib64:/mnt/cache/share/gcc/mpc-0.8.1/lib:$LD_LIBRARY_PATH

export CUDA_HOME=/mnt/cache/share/cuda-11.7
export PATH=/mnt/cache/share/cuda-11.7/bin::$PATH
export LD_LIBRARY_PATH=/mnt/cache/share/cuda-11.7/lib64:$LD_LIBRARY_PATH

sbatch --nodes 1 \
       --phx-priority P0 \
       --partition=mm_v100_32g \
       --job-name=unify_mm_v100 \
       --comment "wbsR-SC230999.001.02" \
       launcher_intelmpi.sh mllm/pipeline/finetune.py config/llava_pretrain5.py \
       --tf32=False --bf16=False --fp16=True --overwrite_output_dir \
       --cfg-options model_args.model_name_or_path='/mnt/lustre/share_data/chenkeqin/ckpt/llava_pretrain_final19/checkpoint-40000'