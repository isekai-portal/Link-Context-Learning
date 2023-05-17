model_args = dict(
    type='llava',
    version='v1',

    # checkpoint config
    cache_dir=None,
    model_name_or_path=r'/mnt/lustre/share_data/zhangzhao2/VG/ckpt/llava/llava_v1/7B',
    vision_tower=r'/mnt/lustre/share_data/zhangzhao2/VG/ckpt/openai/clip-vit-large-patch14',
    pretrain_mm_mlp_adapter=None,

    # model config
    mm_vision_select_layer=-2,
    model_max_length=2048,

    # finetune config
    freeze_backbone=False,
    tune_mm_mlp_adapter=False,
    freeze_mm_mlp_adapter=False,

    # data process config
    is_multimodal=True,
    sep_image_conv_front=False,
    image_token_len=256,
    mm_use_im_start_end=True,
)