import pytest


@pytest.fixture(scope='session')
def plain_formatter():
    from mllm.dataset.process_function import PlainBoxFormatter
    pbf = PlainBoxFormatter()
    return pbf


@pytest.fixture(scope='session')
def rec_dummy_data_file(tmpdir_factory: pytest.TempdirFactory):
    import json

    data_dir = tmpdir_factory.mktemp('dummy_data_dir')
    data_file = data_dir.join('rec_annotation.jsonl')
    objs = [
        {"img_path": "train2014/COCO_train2014_000000549256.jpg", "expression": "guy wihtout cap", "bbox": [0.389, 0.388, 0.816, 0.918],
         "dataset_name": "refcoco+", "height": 500, "width": 375},
        {"img_path": "train2014/COCO_train2014_000000570019.jpg", "expression": "a half drunk beer in a large glass",
         "bbox": [0.452, 0.002, 0.83, 0.659], "dataset_name": "refcocog", "height": 425, "width": 640},
        {"img_path": "train2014/COCO_train2014_000000059708.jpg", "expression": "a lime to the top left of a bunch of bananas",
         "bbox": [0.138, 0.056, 0.348, 0.404], "dataset_name": "refcocog", "height": 287, "width": 442},
        {"img_path": "train2014/COCO_train2014_000000120644.jpg", "expression": "second cup left botto",
         "bbox": [0.181, 0.691, 0.435, 0.878], "dataset_name": "refcoco", "height": 640, "width": 480},
        {"img_path": "train2014/COCO_train2014_000000021717.jpg", "expression": "man with the hat", "bbox": [0.212, 0.068, 0.567, 0.998],
         "dataset_name": "refcoco+", "height": 426, "width": 640},
    ]
    with open(data_file, 'w') as f:
        for obj in objs:
            obj_str = json.dumps(obj)
            f.write(obj_str)
            f.write('\n')
    return data_file


@pytest.fixture(scope='session')
def rec_dataset(cfg_dir, rec_dummy_data_file):
    from mllm.dataset import DATASETS

    filename = rec_dummy_data_file
    image_folder = cfg_dir.COCO_IMAGES_DIR
    cfg = dict(
        type='RECDataset',
        filename=filename,
        image_folder=image_folder,
        template_string='where is <expr>?'
    )
    dataset = DATASETS.build(cfg)
    return dataset


@pytest.fixture(scope='session')
def reverse_rec_dataset(cfg_dir, rec_dummy_data_file):
    from mllm.dataset import DATASETS

    filename = rec_dummy_data_file
    image_folder = cfg_dir.COCO_IMAGES_DIR
    cfg = dict(
        type='ReverseRECDataset',
        filename=filename,
        image_folder=image_folder,
        template_string='please find a caption for <boxes>.'
    )
    dataset = DATASETS.build(cfg)
    return dataset


@pytest.fixture(
    scope='session',
    params=[
        pytest.lazy_fixture('rec_dataset'),
        pytest.lazy_fixture('reverse_rec_dataset'),
    ]
)
def dataset(request):
    return request.param


@pytest.fixture(
    scope='session',
    params=[
        pytest.lazy_fixture('plain_formatter'),
    ]
)
def boxes_processor(request):
    return {'boxes': request.param}


@pytest.fixture(
    scope='session',
    params=[
        pytest.lazy_fixture('boxes_processor'),
    ],
)
def target_processor(request):
    return request.param


@pytest.fixture(scope='session')
def llava_processor(cfg_dir, target_processor):
    from transformers import CLIPImageProcessor, LlamaTokenizer

    tokenizer = LlamaTokenizer.from_pretrained(cfg_dir.LLAVA_7B_TK_PATH)
    tokenizer.pad_token_id = tokenizer.unk_token_id

    preprocessor = dict(
        image=CLIPImageProcessor(),
        text=tokenizer,
        target=target_processor,
        conv=dict(
            image_token_len=256,
            is_multimodal=True,
            sep_image_conv_front=False,
            use_im_start_end=True,
        )
    )
    return preprocessor


@pytest.fixture(scope='session')
def llava_process_func():
    from mllm.dataset import FUNCTIONS

    process_func = dict(
        image=FUNCTIONS.build(cfg=dict(type='LlavaImageProcessorV1')),
        text=FUNCTIONS.build(cfg=dict(type='LlavaTextProcessV1')),
        conv=FUNCTIONS.build(cfg=dict(type='LLavaConvProcessV1')),
        target=FUNCTIONS.build(cfg=dict(type='BoxFormatProcess')),
    )
    return process_func
