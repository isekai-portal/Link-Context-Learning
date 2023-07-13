model_args = dict(
    type='llava',
    # TODO: process version; current version use default version
    version='v1',

    # checkpoint config
    cache_dir=None,
    model_name_or_path=r'/mnt/lustre/share_data/chenkeqin/VG/ckpt/llava/llava_v1/7B',
    vision_tower=r'/mnt/lustre/share_data/chenkeqin/VG/ckpt/openai/clip-vit-large-patch14',
    pretrain_mm_mlp_adapter=None,
    # model config
    mm_vision_select_layer=-2,
    model_max_length=2048,
    qformer_config=dict(
        load_model=False,
        ckpt_path='/mnt/lustre/share_data/zhangzhao2/VG/ckpt/blip-2/blip2_pretrained_vitL.pth',
        num_query_token=257,
        num_features=1024,
        bert_pretrain_path=r'/mnt/lustre/share/hezhiqun/Model/huggingface.co/bert-base-uncased/',
        cross_attention_freq=2,
        hidden_size=768,
        only_qformer=True
    ),
    # finetune config
    freeze_backbone=False,
    tune_mm_mlp_adapter=False,
    freeze_mm_mlp_adapter=False,

    # data process config
    is_multimodal=True,
    sep_image_conv_front=False,
    image_token_len=256,
    mm_use_im_start_end=True,

    target_processor=dict(
        boxes=dict(type='PlainBoxFormatter'),
    ),

    process_func_args=dict(
        conv=dict(type='LLavaConvProcessV1'),
        target=dict(type='BoxFormatProcess'),
        text=dict(type='LlavaTextProcessV1'),
        image=dict(type='LlavaImageProcessorV1'),
    ),

    conv_args=dict(
        conv_template='vicuna_v1.1',
        transforms=dict(type='Expand2square'),
        tokenize_kwargs=dict(truncation_size=2048),
    ),

    gen_kwargs_set_pad_token_id=True,
    gen_kwargs_set_bos_token_id=True,
    gen_kwargs_set_eos_token_id=True,
)