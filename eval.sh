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
# sbatch -p mm_v100_32g \
#     -n 1 \
#     -N 1 \
#     --gres=gpu:8 \
#     --phx-priority P0 \
#     --preempt \
#     -c 64 \
#     --job-name=eval_mmv100 \
#     -x SH-IDC1-10-142-4-22 \
#     --comment "wbsR-SC230999.001.02" \
#     accelerate launch --num_processes 8 --main_process_port 23781 mllm/pipeline/finetune.py \
#         config/icl_imagenet1k_v9_eval.py \
#         --tf32=False --bf16=False --fp16=True \
#         --cfg-options model_args.model_name_or_path=/mnt/lustre/fanweichen2/Research/MLLM/ckpt/train_imagenet1k_policy9 \
#         --per_device_eval_batch_size 1 \
#         --output_dir /mnt/cache/fanweichen2/Code/unify_mllm/result/output/train_imagenet1k_policy9/1k2way-10shot-real \
#         --cfg-options data_args.use_icl=True \
#         --cfg-options data_args.shot=10 \


# sbatch -p mm_v100_32g  --quotatype=auto \
#     --comment "wbsR-SC230999.001.02" \
#     --job-name=$name \
#     --phx-priority P0 \
#     --preempt \
#     --nodes 4 \
#     -x SH-IDC1-10-142-4-22 \
#     launcher_intelmpi.sh mllm/pipeline/finetune.py config/icl_imagenet1k_v9_eval.py \
#         --tf32=False --bf16=False --fp16=True \
#         --cfg-options model_args.model_name_or_path=/mnt/lustre/fanweichen2/Research/MLLM/ckpt/train_imagenet1k_policy9 \
#         --per_device_eval_batch_size 1 \
#         --output_dir /mnt/cache/fanweichen2/Code/unify_mllm/result/output/train_imagenet1k_policy9/1k2way-10shot-real \
#         --cfg-options data_args.use_icl=True \
#         --cfg-options data_args.shot=10 \

#wrong
# sbatch -p mm_v100_32g \
#     -n 1 \
#     -N 1 \
#     --gres=gpu:8 \
#     --phx-priority P0 \
#     --preempt \
#     -c 64 \
#     --job-name=eval_mmv100 \
#     -x SH-IDC1-10-142-4-22 \
#     --comment "wbsR-SC230999.001.02" \
#     accelerate launch --num_processes 8 --main_process_port 23781 mllm/pipeline/finetune.py \
#         config/icl_imagenet1k_v9_eval.py \
#         --tf32=False --bf16=False --fp16=True \
#         --cfg-options model_args.model_name_or_path=/mnt/lustre/fanweichen2/Research/MLLM/ckpt/train_imagenet1k_policy9_8shot \
#         --per_device_eval_batch_size 1 \
#         --output_dir /mnt/lustre/fanweichen2/tmp_save/output/train_imagenet1k_policy9_8shot/test100_ablation/8-shot/1-14/ \
#         --cfg-options data_args.use_icl=True \
#         --cfg-options data_args.shot=8 \

# sbatch -p mm_v100_32g \
#     -n 1 \
#     -N 1 \
#     --gres=gpu:8 \
#     --phx-priority P0 \
#     --preempt \
#     -c 64 \
#     --job-name=eval_mmv100 \
#     -x SH-IDC1-10-142-4-22 \
#     --comment "wbsR-SC230999.001.02" \
#     accelerate launch --num_processes 8 --main_process_port 23781 mllm/pipeline/finetune.py \
#         config/icl_imagenet1k_v13_mix_eval.py \
#         --tf32=False --bf16=False --fp16=True \
#         --cfg-options model_args.model_name_or_path=/mnt/lustre/fanweichen2/Research/MLLM/ckpt/checkpoint-3200 \
#         --per_device_eval_batch_size 1 \
#         --output_dir /mnt/lustre/fanweichen2/tmp_save/output/train_lcl_mix_finetune/test100_eval/false_label/2-shot/2/ \
#         --cfg-options data_args.use_icl=True \
#         --cfg-options data_args.shot=2 \


# sbatch -p mm_v100_32g \
#     -n 1 \
#     -N 1 \
#     --gres=gpu:8 \
#     --phx-priority P0 \
#     --preempt \
#     -c 64 \
#     --job-name=eval_mmv100 \
#     -x SH-IDC1-10-142-4-22 \
#     --comment "wbsR-SC230999.001.02" \
#     accelerate launch --num_processes 8 --main_process_port 23781 mllm/pipeline/finetune.py \
#         config/icl_imagenet1k_v13_mix_eval_ISEKAI.py \
#         --tf32=False --bf16=False --fp16=True \
#         --cfg-options model_args.model_name_or_path=/mnt/lustre/fanweichen2/Research/MLLM/ckpt/train_icl_imagenet1k_v13_train_8shot_weight-2 \
#         --per_device_eval_batch_size 1 \
#         --output_dir /mnt/cache/fanweichen2/Code/unify_mllm/result/output/train_icl_imagenet1k_v13_train_8shot_weight-2/isekai/8-shot/ \
#         --cfg-options data_args.use_icl=True \
#         --cfg-options data_args.shot=8 \

# for(( i=1;i<9;i++)) do
#     echo ${i}
#     sbatch -p mm_v100_32g \
#         -n 1 \
#         -N 1 \
#         --gres=gpu:8 \
#         --phx-priority P0 \
#         --preempt \
#         -c 64 \
#         --job-name=eval_mmv100 \
#         -x SH-IDC1-10-142-4-22 \
#         --comment "wbsR-SC230999.001.02" \
#         accelerate launch --num_processes 8 --main_process_port 23781 mllm/pipeline/finetune.py \
#             config/icl_imagenet1k_v13_mix_eval_ISEKAI.py \
#             --tf32=False --bf16=False --fp16=True \
#             --cfg-options model_args.model_name_or_path=/mnt/lustre/fanweichen2/Research/MLLM/ckpt/checkpoint-10000 \
#             --per_device_eval_batch_size 1 \
#             --output_dir /mnt/lustre/fanweichen2/tmp_save/output/mix/ISEKAI-20/"${i}"-shot/ \
#             --cfg-options data_args.use_icl=True \
#             --cfg-options data_args.shot=${i}∂∂
# done

for(( i=1;i<2;i++)) do
    echo ${i}
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
            config/icl_business_qa_all.py \
            --tf32=False --bf16=False --fp16=True \
            --cfg-options model_args.model_name_or_path=/mnt/lustre/share_data/taiyan/checkpoint/okapis/checkpoints/2way_weight \
            --per_device_eval_batch_size 1 \
            --output_dir /mnt/lustre/fanweichen2/tmp_save/output/icl_imagenet1k_v9_weigth_tune_new_all_20ep_eweight/business_easy_det_sim6/"${i}"-shot/ \
            --cfg-options data_args.use_icl=True \
            --cfg-options data_args.shot=${i}
done