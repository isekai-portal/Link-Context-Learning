model_args = dict(
    type='flamingo',
    model_name_or_path="/mnt/lustre/taiyan/ckpt/huggingface/openflamingo-9b-hf",
    tokenizer_name_or_path="/mnt/lustre/taiyan/ckpt/huggingface/llama-7b-hf",

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