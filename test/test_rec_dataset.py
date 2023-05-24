from pathlib import Path

import matplotlib.pyplot as plt


def test_rec_dataset():
    from mllm.dataset import DATASETS
    from mllm.utils import draw_bounding_boxes, show

    filename = r'D:\home\code\unify_mllm\data\dummy_rec_ann.jsonl'
    image_folder = r'D:\home\dataset\mscoco\images'
    cfg = dict(
        type='RECDataset',
        filename=filename,
        image_folder=image_folder,
        template_string='where is <expr>?'
    )
    dataset = DATASETS.build(cfg)
    for i in range(5):
        item = dataset[i]
        print(item)
        res = draw_bounding_boxes(image=item['image'], boxes=item['target']['boxes'], colors='red', width=4)
        show(res)
        plt.title(item['conversations'][0]['value'])
        plt.show()
        plt.close()


def test_rec_conv_dataset_llava():
    from transformers import CLIPImageProcessor, LlamaTokenizer
    from mllm.dataset import (
        PlainBoxFormatter,
        SingleImageConvDataset,
        DATASETS,
        FUNCTIONS,
        TRANSFORMS,
        de_norm_box_xyxy,
    )
    from mllm.utils import decode_generate_ids, show, draw_bounding_boxes

    filename = r'D:\home\code\unify_mllm\data\dummy_rec_ann.jsonl'
    image_folder = r'D:\home\dataset\mscoco\images'
    cfg = dict(
        type='RECDataset',
        filename=filename,
        image_folder=image_folder,
        template_string='where is <expr>?'
    )
    dataset = DATASETS.build(cfg)

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
    print(train_ds[0])

    test_ds = SingleImageConvDataset(
        dataset=dataset,
        preprocessor=preprocessor,
        process_func=process_func,
        transforms=transforms,
        mode='validation',
    )
    print(test_ds[0])

    for i in range(5, 10):
        item = train_ds.__getitem__(index=i, debug_mode=True)
        image = item['image']
        item = item['ret']
        inputs = decode_generate_ids(preprocessor['text'], item['input_ids'])
        print(inputs)
        targets = decode_generate_ids(preprocessor['text'], item['labels'])
        extracted_ans = preprocessor['target']['boxes'].extract(targets)
        assert len(extracted_ans) == 1 and len(extracted_ans[0]) == 1 and len(extracted_ans[0][0]) == 4
        extract_box = extracted_ans[0][0]
        extract_box = de_norm_box_xyxy(extract_box, w=image.width, h=image.height)
        res = draw_bounding_boxes(image, boxes=[extract_box], colors='red', width=4)
        show(res)
        plt.title(targets)
        plt.show()
        plt.close()
