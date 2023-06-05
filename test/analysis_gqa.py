from mllm.dataset import DATASETS

cfg = dict(
    type='GQADataset',
    filename=r'',
    image_folder=r'',
    template_string=r"Can you analyze the image <image> and answer my question: '<question>'? Please include your reasoning process and identify the bounding boxes of any objects in the image with square brackets after the objects.",

)


ds = DATASETS.build(cfg)

from matplotlib import pyplot as plt

for item in ds:
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
