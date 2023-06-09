from functools import partial

import pytest
from pytest_lazyfixture import lazy_fixture

from mllm.conversation import get_conv_template


@pytest.mark.parametrize('preprocessor,process_func,conv_template', [
    (lazy_fixture('llava_processor'), lazy_fixture('llava_process_func'), partial(get_conv_template, name='vicuna_v1.1')),
    # (lazy_fixture('otter_processor'), lazy_fixture('otter_process_func'), partial(get_conv_template, name='otter')),
])
def test(preprocessor, process_func, conv_template):
    def dummy_dataset():
        from PIL import Image

        return [
            # dict(
            #     image=[
            #         Image.new('RGB', (224, 224)),
            #         Image.new('RGB', (224, 224)),
            #         Image.new('RGB', (224, 224)),
            #     ],
            #     target=[
            #         dict(boxes=[[10, 10, 20, 20], [20, 20, 30, 30], [30, 30, 40, 40]], points=[[50, 50], [60, 60]]),
            #         dict(boxes=[[100, 100, 110, 110]], points=[[120, 120]]),
            #         dict(boxes=[[120, 120, 130, 130]], points=[[140, 140]]),
            #     ],
            #     conversations=[
            #         {
            #             "from": "human",
            #             "value": "Human 1. <image> has <boxes>. <image> has <boxes>.",
            #             "boxes_seq": [[(0, 2), (0, 1)], [(0, 0), (1, 0)]],
            #             "points_seq": [],
            #             "image_seq": [0, 1],
            #         },
            #         {
            #             "from": "gpt",
            #             "value": "GPT 1. <points> <points>",
            #             "boxes_seq": [],
            #             "points_seq": [[(1, 0)], [(1, 0)]],
            #             "image_seq": [],
            #         },
            #         {
            #             "from": "human",
            #             "value": "Human 2. <image> has <boxes>.",
            #             "boxes_seq": [[(2, 0)]],
            #             "points_seq": [],
            #             "image_seq": [2],
            #         },
            #         {
            #             "from": "gpt",
            #             "value": "GPT 2. <boxes> <points>",
            #             "boxes_seq": [[(2, 0)]],
            #             "points_seq": [[(2, 0)]],
            #         }
            #     ]
            # ),
            dict(
                image=Image.new('RGB', (224, 224)),
                target=dict(boxes=[[10, 10, 20, 20], [20, 20, 30, 30], [30, 30, 40, 40]], points=[[50, 50], [60, 60]]),
                conversations=[
                    {
                        "from": "human",
                        "value": "<image> Given Box<boxes> and Box<boxes> Guess Next Box? Also Give<points>.",
                        "boxes_seq": [[0], [1], ],
                        "points_seq": [[0]],
                    },
                    {
                        "from": "gpt",
                        "value": "The Next Box is <boxes>. The Next two points is <points> and <points>. the final is <points>.",
                        "boxes_seq": [[2]],
                        "points_seq": [[0], [1], [0, 1]],
                    },
                ]
            )
        ]

    from mllm.dataset import SingleImageConvDataset

    ds = SingleImageConvDataset(
        dataset_generator=dummy_dataset,
        preprocessor=preprocessor,
        process_func=process_func,
        conv_template=conv_template,
        transforms=lambda x, y: (x, y),
        mode='train',
    )

    item = ds[0]
    print(item)

