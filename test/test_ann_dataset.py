from pathlib import Path


def test_ann_dataset():
    from mllm.dataset import DATASETS

    cfg = dict(
        type='ConvAnnotationDataset',
        filename=r'D:\home\code\unify_mllm\data\annotations.jsonl',
        image_folder=r'D:\home\dataset\mscoco\images',
        add_image_placeholder=True,
    )

    ds = DATASETS.build(cfg=cfg)
    for i, item in enumerate(ds):
        print(item)
        if i > 10:
            break


def test_ann_conv_dataset_llava():
    from transformers import CLIPImageProcessor, LlamaTokenizer
    from mllm.dataset import (
        PlainBoxFormatter,
        SingleImageConvDataset,
        DATASETS,
        FUNCTIONS,
        TRANSFORMS,
    )

    cfg = dict(
        type='ConvAnnotationDataset',
        filename=r'D:\home\code\unify_mllm\data\annotations.jsonl',
        image_folder=r'D:\home\dataset\mscoco\images',
        add_image_placeholder=True,
    )

    dataset = DATASETS.build(cfg=cfg)

    tokenizer = LlamaTokenizer.from_pretrained(rf"{Path(__file__).parent}\llava_7b_tk")
    tokenizer.pad_token_id = tokenizer.unk_token_id
    preprocessor = dict(
        image=CLIPImageProcessor(),
        text=tokenizer,
        target={'boxes': PlainBoxFormatter()},
        conv=dict(
            image_token_len=256,
            is_multimodal=True,
            sep_image_conv_front=False,
            use_im_start_end=True,
        )
    )

    process_func = dict(
        image=FUNCTIONS.build(cfg=dict(type='LlavaImageProcessorV1')),
        text=FUNCTIONS.build(cfg=dict(type='LlavaTextProcessV1')),
        conv=FUNCTIONS.build(cfg=dict(type='LLavaConvProcessV1')),
        target=FUNCTIONS.build(cfg=dict(type='BoxFormatProcess')),
    )

    transforms = TRANSFORMS.build(cfg=dict(type='Expand2square', ))

    train_ds = SingleImageConvDataset(
        dataset=dataset,
        preprocessor=preprocessor,
        process_func=process_func,
        transforms=transforms,
        mode='train',
    )
    for i, item in enumerate(train_ds):
        print(item)
        if i > 10:
            break

