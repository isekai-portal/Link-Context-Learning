_base_ = ['./otter_9b.py']
# openflamingo not train <answer> token, so this version model only support
# eval on ckpt of finetuned model. not the origin openflamingo weight.

model_args = dict(
    model_name_or_path="",
)
