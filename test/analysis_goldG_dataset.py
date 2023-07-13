from matplotlib import pyplot as plt

from pathlib import Path
from mllm.dataset.single_image_dataset.goldG import GoldGDataset
from mllm.utils import draw_bounding_boxes, show

dataset = GoldGDataset(
    root_coco=Path(r'D:\home\dataset\mscoco\images\train2014'),
    root_vg=Path('datasets'),
    ann_file=r"D:\home\code\mdetr\mdetr_annotations\final_mixed_train.json",
    template_string='caption with the image<image> with box',
)

colors = ('red', 'green', 'blue', 'yellow', '#533c1b', '#c04851')


def cycle_colors(size):
    from itertools import cycle
    return [x for _, x in zip(range(size), cycle(colors))]

# TODO: the dataset need cleanup. filter out the gqa question. control the number of obj?
#  also, it repeat with ref dataset and gqa dataset.
for idx in range(len(dataset)):
    if dataset.coco.loadImgs(dataset.ids[idx])[0]["data_source"] == 'coco':
        item = dataset[idx]
        print(item)
        print()

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

        show(draw_bounding_boxes(item['image'], boxes_to_draw, colors=color_to_draw, width=4))
        plt.show()
        plt.close()

        _ = input()
