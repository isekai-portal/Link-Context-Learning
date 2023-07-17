export PATH=/mnt/cache/share/gcc/gcc-7.3.0/bin:$PATH
export LD_LIBRARY_PATH=/mnt/cache/share/gcc/gcc-7.3.0/lib64:/mnt/cache/share/gcc/mpc-0.8.1/lib:$LD_LIBRARY_PATH

export CUDA_HOME=/mnt/cache/share/cuda-11.7
export PATH=/mnt/cache/share/cuda-11.7/bin::$PATH
export LD_LIBRARY_PATH=/mnt/cache/share/cuda-11.7/lib64:$LD_LIBRARY_PATH

# rm -rf *.out
# sbatch --nodes 2 \
#        --phx-priority P0 \
#        --partition=mm_v100_32g \
#        --job-name=unify_mm_v100 \
#        --comment "wbsR-SC230999.001.02" \
#        launcher_intelmpi.sh mllm/pipeline/finetune.py config/llava_pretrain5.py \
#        --tf32=False --bf16=False --fp16=True --overwrite_output_dir \
#        --cfg-options model_args.model_name_or_path='/mnt/lustre/share_data/chenkeqin/ckpt/llava_pretrain_final19/checkpoint-40000'
# rm -rf /mnt/lustre/fanweichen2/Research/MLLM/dummy_exp/result


# sbatch -p mm_v100_32g \
#     -n 1 \
#     -N 1 \
#     --gres=gpu:4 \
#     -c 64 \
#     --job-name=eval_mmv100 \
#     --comment "wbsR-SC230999.001.02" \
#     accelerate launch --num_processes 4 --main_process_port 23781 mllm/pipeline/finetune.py \
#         config/llava_eval_multi_rec.py \
#         --tf32=False --bf16=False --fp16=True \
#         --cfg-options model_args.model_name_or_path=/mnt/lustre/fanweichen2/Research/MLLM/dummy_exp/Icl_pretrain/checkpoint-4000 \
#         --per_device_eval_batch_size 4 \
#         --output_dir /mnt/cache/fanweichen2/Code/unify_mllm/result/Icl_pretrain/checkpoint-4000_new \
#        --cfg-options model_args.qformer_config.num_query_token=32 \
#        --cfg-options model_args.image_token_len=32 \
#        --cfg-options model_args.qformer_config.load_model=True


#SH-IDC1-10-142-4-22
sbatch -p mm_v100_32g \
    -n 1 \
    -N 1 \
    --gres=gpu:4 \
    -c 64 \
    --job-name=eval_mmv100 \
    -x SH-IDC1-10-142-4-22 \
    --comment "wbsR-SC230999.001.02" \
    accelerate launch --num_processes 4 --main_process_port 23781 mllm/pipeline/finetune.py \
        config/icl_eval_vqa.py \
        --tf32=False --bf16=False --fp16=True \
        --cfg-options model_args.model_name_or_path=/mnt/lustre/share_data/feijingjing1/tmp/for_weichen/v2/checkpoint-2000 \
        --per_device_eval_batch_size 1 \
        --output_dir /mnt/cache/fanweichen2/Code/unify_mllm/result/v2/checkpoint-2000_new3 \
        --cfg-options data_args.use_icl=True \
        --cfg-options data_args.shot=2 \
