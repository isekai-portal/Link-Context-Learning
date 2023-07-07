_base_ = ['llava_v1_7b.py']

model_args = dict(
    model_name_or_path=r'/mnt/lustre/share_data/chenkeqin/ckpt/huggingface/vicuna-13b',
    vision_tower=r'/mnt/lustre/share_data/chenkeqin/VG/ckpt/openai/clip-vit-large-patch14',
    pretrain_mm_mlp_adapter=r'/mnt/lustre/share_data/chenkeqin/ckpt/huggingface/mm_projector_llava-13bv1.1-sft.bin',
)