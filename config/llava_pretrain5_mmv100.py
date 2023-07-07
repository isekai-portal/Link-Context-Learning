_base_ = ['_base_/dataset/mix_pretrain_prob.py', '_base_/model/llava_v1_7b.py', '_base_/train/llava_fsdp.py']

training_args = dict(
    output_dir='/mnt/lustre/share_data/chenkeqin/dummy_exp_unify_mllm/{{fileBasenameNoExtension}}',
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    dataloader_num_workers=4,
    tf32=False,
    bf16=False,
    fp16=True,
)