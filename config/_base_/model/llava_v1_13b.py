_base_ = ['llava_v1_7b.py']

model_args = dict(
    model_name_or_path=r'/mnt/lustre/share_data/chenkeqin/dummy_exp_unify_mllm/llava13bo_pretrain3+concat+recvg+e1',
    vision_tower=r'/mnt/lustre/share_data/chenkeqin/VG/ckpt/openai/clip-vit-large-patch14',
)