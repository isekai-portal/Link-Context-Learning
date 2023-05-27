import itertools

import matplotlib.pyplot as plt
import pytest


def test_flickr(cfg_dir):
    from mllm.dataset.single_image_dataset.flickr import FlickrParser

    ds = FlickrParser(
        filename=cfg_dir.FLICKR_ANNOTATION_FILE,
        annotation_dir=cfg_dir.FLICKR_ANNOTATION_DIR,
    )

    item = ds[0]
    print(item)


def test_flickr_caption(cfg_dir):
    from mllm.dataset.single_image_dataset.flickr import FlickrDataset

    ds = FlickrDataset(
        filename=cfg_dir.FLICKR_ANNOTATION_FILE,
        annotation_dir=cfg_dir.FLICKR_ANNOTATION_DIR,
        image_dir=cfg_dir.FLICKR_IMAGE_DIR,
        template_string='give a caption for the image.',
    )

    for i in range(0, 25, 5):
        item = ds[i]
        print(item)
        colors = ['red', 'green', 'blue', 'yellow', '#533c1b', '#c04851']
        bidxs_to_draw = []
        color_to_draw = []
        for idx, boxes in enumerate(item['conversations'][-1]['boxes_seq']):
            color = colors[idx % len(colors)]
            for box in boxes:
                bidxs_to_draw.append(box)
                color_to_draw.append(color)

        boxes_to_draw = []
        for idx in bidxs_to_draw:
            boxes_to_draw.append(item['target']['boxes'][idx])

        from mllm.utils import draw_bounding_boxes, show
        res = draw_bounding_boxes(image=item['image'], boxes=boxes_to_draw, colors=color_to_draw, width=4)
        show(res)
        plt.show()
        plt.close()


@pytest.mark.parametrize(
    "caption_with_box, box_max_num",
    list(itertools.product(['none', 'question', 'all'], [3, 5])),
)
def test_flickr_box2caption(cfg_dir, caption_with_box, box_max_num):
    from mllm.dataset.single_image_dataset.flickr import FlickrBox2Caption

    ds = FlickrBox2Caption(
        filename=cfg_dir.FLICKR_ANNOTATION_FILE,
        annotation_dir=cfg_dir.FLICKR_ANNOTATION_DIR,
        image_dir=cfg_dir.FLICKR_IMAGE_DIR,
        caption_with_box=caption_with_box,
        box_max_num=box_max_num,
        template_string='give a caption for the objects<boxes> in the image.',
    )
    for i in range(0, 25, 5):
        item = ds[i]
        print(item)
        colors = ['red', 'green', 'blue', 'yellow', '#533c1b', '#c04851']
        bidxs_to_draw = []
        color_to_draw = []
        if "boxes_seq" in item['conversations'][-1] and bool(item['conversations'][-1]['boxes_seq']):
            for idx, boxes in enumerate(item['conversations'][-1]['boxes_seq']):
                color = colors[idx % len(colors)]
                for box in boxes:
                    bidxs_to_draw.append(box)
                    color_to_draw.append(color)

            boxes_to_draw = []
            for idx in bidxs_to_draw:
                boxes_to_draw.append(item['target']['boxes'][idx])

            from mllm.utils import draw_bounding_boxes, show
            res = draw_bounding_boxes(image=item['image'], boxes=boxes_to_draw, colors=color_to_draw, width=4)
            show(res)
            plt.show()
            plt.close()
