model_args = dict(
    type='flamingo',
    create=dict(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=r"/mnt/lustre/share_data/zhangzhao2/VG/ckpt/llama/7B_hf",
        tokenizer_path=r"/mnt/lustre/share_data/zhangzhao2/VG/ckpt/llama/7B_hf",
        cross_attn_every_n_layers=4,
    ),
    load=dict(
        checkpoint_path=r'/mnt/lustre/share_data/zhangzhao2/VG/ckpt/open_flamingo_9b/9B/checkpoint.pt',
        strict=False,
        map_location='cpu',
    ),
    quantize='fp16',
)
