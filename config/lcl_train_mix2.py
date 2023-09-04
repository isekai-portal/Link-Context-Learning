_base_ = ['_base_/dataset/mix_multitrain2.py', '_base_/model/lcl_7b.py', '_base_/train/llava_fsdp.py']

training_args = dict(
    num_train_epochs=50,
    save_strategy='steps'
)