from ..root import DATASETS
from ..utils import MInstrDataset


@DATASETS.register_module()
class InstructDataset(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(), template_string='', template_file=None)

    def __getitem__(self, index):
        item = self.get_raw_item(index)
        img_path = item['image']
        conversations = item['conversations']

        image = self.get_image(img_path)
        ret = {
            'image': image,
            'conversations': conversations,
        }
        return ret
