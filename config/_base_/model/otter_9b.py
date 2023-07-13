model_args = dict(
    type='otter',
    model_name_or_path="/mnt/lustre/share_data/chenkeqin/ckpt/huggingface/hub/models--luodian--otter-9b-hf/snapshots/main",
    tokenizer_name_or_path="/mnt/lustre/share_data/chenkeqin/ckpt/huggingface/hub/models--decapoda-research--llama-7b-hf/snapshots/main",

    target_processor=dict(
        boxes=dict(type='PlainBoxFormatter'),
    ),

    process_func_args=dict(
        conv=dict(type='OtterConvProcess'),
        target=dict(type='BoxFormatProcess'),
        text=dict(type='OtterTextProcess'),
        image=dict(type='OtterImageProcess'),
    ),

    conv_args=dict(
        conv_template='otter',
        transforms=dict(type='Expand2square'),
        tokenize_kwargs=dict(),
    ),

    conv_processor=dict(
        image_token_at_begin='True',
    ),

    gen_kwargs_set_pad_token_id=True,
    gen_kwargs_set_bos_token_id=True,
    gen_kwargs_set_eos_token_id=False,
)
