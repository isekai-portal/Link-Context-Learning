import copy
from PIL import Image
from functools import partial
from typing import Dict, Any, Optional
from mllm.conversation import get_conv_template
from mllm.dataset.root import TRANSFORMS, FUNCTIONS
from mllm.dataset.single_image_convsation import SingleImageConvDatasetMixin

def prepare_demo_dataset(
        model_args,
        preprocessor: Dict[str, Any],
):
    conv_args = model_args.conv_args
    tokenize_kwargs = conv_args.get('tokenize_kwargs', {})
    conv_template_ = conv_args.get('conv_template', 'vicuna_v1.1')
    if isinstance(conv_template_, list):
        conv_template = {item: partial(get_conv_template, name=item) for item in conv_template_}
    else:
        conv_template = partial(get_conv_template, name=conv_template_)
    transforms = conv_args.get('transforms', None)
    if transforms is not None:
        transforms = TRANSFORMS.build(transforms)
    # process func
    process_func = {}
    for k, v in model_args.process_func_args.items():
        process_func[k] = FUNCTIONS.build(cfg=v)

    ds = SingleImageInteractive(
        preprocessor=preprocessor,
        process_func=process_func,
        tokenize_kwargs=tokenize_kwargs,
        conv_template=conv_template,
        training_args=None,
        transforms=transforms,
        use_icl=True,
        mode='test',
    )
    return ds

class SingleImageInteractive(SingleImageConvDatasetMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_meta = None

    def update_data(self, data):
        self.data_meta = data

    def clear_data(self):
        self.data_meta = None
    
    def get_ret(self, image, question, answer, conv_mode=None):
        ret = {
            'image': image,
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': f"{answer}",
                },
            ]
        }
        if conv_mode is not None:
            ret['mode'] = conv_mode
        return ret

    # def get_raw_item(self, index):
    #     assert self.data_meta is not None

    #     return [self.get_ret(image=self.data_meta['infer_img'], question=self.data_meta['infer_q'], answer='')]

    def get_raw_icl_item(self, index, shot):
        assert self.data_meta is not None
        result = []
        question = '[BEGIN EXAMPLE] What is in the image <image>?'
        for img in self.data_meta['pos_img']:
            answer = self.data_meta['pos_a'][0] + ' [END EXAMPLE]'
            result.append(self.get_ret(image=img, question=question, answer=answer))
        for img in self.data_meta['neg_img']:
            answer = self.data_meta['neg_a'][0] + ' [END EXAMPLE]'
            result.append(self.get_ret(image=img, question=question, answer=answer))

        # remove system infomation in the middle of prompt
        for i in range(len(result)):
            if i == 0:
                support_mode = 'causal_v1.0'
            else:
                support_mode = 'hypnotized_ans_v1.0'
            result[i]['mode'] = support_mode

        if len(result) == 0:
            infer_mode = 'vicuna_v1.1' # vqa
        else:
            infer_mode = 'final_v1.0' # lcl

        assert len(self.data_meta['infer_img']) == 1
        infer_img = self.data_meta['infer_img'][0]
        infer_question = self.data_meta['infer_q'][0] + '<image>'
        result.append(self.get_ret(image=infer_img, question=infer_question, answer='', conv_mode=infer_mode))
        return result

    def __getitem__(self, index, debug_mode=False) -> Dict[str, Any]:
        item = super().__getitem__(index, debug_mode)
        update_keys = ['image', 'input_ids', 'attention_mask', 'labels']

        ret = dict()
        for k, v in item.items():
            if k == 'image':
                k = 'images'
            ret[k] = v.unsqueeze(0).cuda()
        return ret


    def __len__(self):
        return 1
