_base_ = ['_base_/dataset/rec_tiny.py', '_base_/model/otter_9b.py', '_base_/train/otter_ds.py']

training_args = dict(
    output_dir='/mnt/lustre/share_data/chenkeqin/exp_unify_mllm/{{fileBasenameNoExtension}}',
    overwrite_output_dir=True,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    dataloader_num_workers=4,
)
