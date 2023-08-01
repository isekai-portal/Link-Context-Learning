import warnings
from functools import partial
from typing import Dict, Any, Callable, List, Optional, Tuple, Type

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import TrainingArguments

from mllm.dataset.process_function.llava_process_function import IGNORE_INDEX

from .root import IMAGE_PLACEHOLDER, BOXES_PLACEHOLDER
from ..conversation import Conversation, get_conv_template
from ..utils import post_process_generate_ids


class SingleImageConvDatasetMixin:

    def __init__(
            self,
            *args,
            preprocessor: Dict[str, Any],
            process_func: Dict[str, Any],
            conv_template: Callable[[], Conversation] = partial(get_conv_template, name='vicuna_v1.1'),
            conv_template_icl: Callable[[], Conversation] = partial(get_conv_template, name='icl_v1.0'),
            mode='train',
            tokenize_kwargs: dict = None,
            training_args: TrainingArguments = None,
            transforms: Optional[Callable] = None,
            use_icl = False,
            shot=1,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert mode in ['train', 'validation', 'test']

        self.preprocessor = preprocessor
        self.process_func = process_func
        self.conv_template = conv_template
        self.conv_template_icl = conv_template_icl
        self.mode = mode
        self.tokenize_kwargs = tokenize_kwargs if tokenize_kwargs is not None else {}
        self.training_args = training_args
        self.transforms = transforms
        self.use_icl = use_icl
        self.shot = shot

    def __get_icl_item__(self, item, do_mask=None, debug_mode=False, train_mode='train', mode='common', eval_icl=False) -> Dict[str, Any]:
        # get_icl_item
        #item = self.get_raw_item(index)
        image: Image.Image = item.get('image', None)
        target: Dict[str, Any] = item.get('target', None)
        raw_conv: List[Dict[str, Any]] = item['conversations']

        # transform
        assert isinstance(image, list) == isinstance(target, list)
        multimage_mode = isinstance(image, list)
        if isinstance(image, list):
            # TODO: validate raw item
            transformed_image, transformed_target = [], []
            for img, tgt in zip(image, target):
                if self.transforms is not None and image is not None:
                    img, tgt = self.transforms(img, tgt)
                if tgt is not None:
                    tgt['width'], tgt['height'] = img.width, img.height
                transformed_image.append(img)
                transformed_target.append(tgt)
            image, target = transformed_image, transformed_target
        else:
            self.validate_raw_item(item)  # only validate for single image.
            if self.transforms is not None and image is not None:
                image, target = self.transforms(image, target)
            if target is not None:
                target['width'], target['height'] = image.width, image.height

        # preprocess
        raw_conv = self.process_conv(raw_conv,mode)
        raw_conv, image = self.process_conv_multimage(raw_conv, image)
        raw_conv, _ = self.process_target(raw_conv, target, multimage_mode=multimage_mode)
        conv = self.build_conv(raw_conv,mode)
        text_dict = self.process_text(conv, do_mask, eval_icl)
        image_dict = self.process_image(image)

        # return
        ret_dict = {}
        ret_dict.update(text_dict)
        ret_dict.update(image_dict)

        if debug_mode:
            return {'ret': ret_dict, 'raw_conv': raw_conv, 'conv': conv, 'image': image}
        return ret_dict

    def __getitem__(self, index, debug_mode=False) -> Dict[str, Any]:
        # getitem
        # if self.training_args is not None:
        #     using_icl = self.training_args.icl
        # else:
        if self.use_icl:
            res = self.__getitem_icl__(index,debug_mode)
        else:
            res = self.__getitem_origin__(index,debug_mode)
        
        return res

    def __getitem_icl__(self, index, debug_mode=False) -> Dict[str, Any]:
        # getitem
        #dict_keys(['input_ids', 'attention_mask', 'labels', 'image'])
        update_keys = ['input_ids', 'attention_mask', 'labels']
        ret_dict = {'image':[]}
        #idx_list = [item for item in range(index, index+5)]
        dict_list = self.get_raw_icl_item(index,self.shot)
        if self.mode != 'train':
            for i in range(len(dict_list)):
                item = dict_list[i]

                conv_mode = 'icl'
                if 'mode' in item.keys():
                    conv_mode = item['mode']
                if i != len(dict_list) - 1:
                    sub_dict = self.__get_icl_item__(item,mode=conv_mode, eval_icl=True)
                else:
                    sub_dict = self.__get_icl_item__(item,mode=conv_mode)

                ret_dict['image'].append(sub_dict['image'].unsqueeze(0))
                
                for k in update_keys:
                    if k not in ret_dict.keys():
                        ret_dict[k] = sub_dict[k]

                    else:
                        ret_valid = ret_dict[k][:-1]
                        sub_valid = sub_dict[k]
                        if k != 'attention_mask':
                            if k == 'input_ids':
                                ret_valid = ret_dict[k][:-1]

                        # if k == 'labels' and i == len(dict_list)-1:
                        #     ret_valid = torch.zeros_like(ret_valid) - 100
                        #     ret_valid[:] = -100

                        ret_dict[k] = torch.cat([ret_valid,sub_valid],dim=0)
            ret_dict['image'] = torch.cat(ret_dict['image'],dim=0)
        else:
            for i in range(len(dict_list)):
                item = dict_list[i]

                conv_mode = 'icl'
                if 'mode' in item.keys():
                    conv_mode = item['mode']

                sub_dict = self.__get_icl_item__(item,mode=conv_mode)

                ret_dict['image'].append(sub_dict['image'].unsqueeze(0))
                
                # concatenate multi-context
                if i == 0:
                    for k in update_keys:
                        value = sub_dict[k][:-1]
                        ret_dict[k] = value
                    # # mask the first context label
                    #     if k == "labels":
                    #         ret_dict[k][:] = IGNORE_INDEX
                else:
                    for k in update_keys:
                        # remove the sep2 symbolic for each context (except the last round)
                        if i != len(dict_list) -1:
                            value = sub_dict[k][:-1]
                        else:
                            value = sub_dict[k]
                            # mask all labels except the last one
                            if k == "labels":
                                ret_dict[k][:] = IGNORE_INDEX
                            
                        ret_dict[k] = torch.cat([ret_dict[k], value], dim = 0)

            ret_dict['image'] = torch.cat(ret_dict['image'],dim=0)


        if not hasattr(self, '_printed_sample'):

            print('mask: ',ret_dict['attention_mask'].shape)
            print('labels: ',ret_dict['labels'].shape)
            print('input_ids: ',ret_dict['input_ids'].shape)
            print('image: ',ret_dict['image'].shape)


            self._printed_sample = True
            post_processed_labels = post_process_generate_ids(self.preprocessor['text'], ret_dict['labels'])
            print(f"           labels: {self.preprocessor['text'].convert_ids_to_tokens(post_processed_labels)}")

            print(f"=================== {self.mode} sample ===================", flush=True)
            print(f"        input_ids: {self.preprocessor['text'].convert_ids_to_tokens(ret_dict['input_ids'])}")

            print(f"decoded input_ids: {self.preprocessor['text'].decode(ret_dict['input_ids'])}")
            print(f"decoded    labels: {self.preprocessor['text'].decode(post_processed_labels)}")

        return ret_dict


    def __getitem_origin__(self, index, debug_mode=False) -> Dict[str, Any]:
        # getitem
        item = self.get_raw_item(index)
        image: Image.Image = item.get('image', None)
        target: Dict[str, Any] = item.get('target', None)
        raw_conv: List[Dict[str, Any]] = item['conversations']

        # transform
        assert isinstance(image, list) == isinstance(target, list)
        multimage_mode = isinstance(image, list)
        if isinstance(image, list):
            # TODO: validate raw item
            transformed_image, transformed_target = [], []
            for img, tgt in zip(image, target):
                if self.transforms is not None and image is not None:
                    img, tgt = self.transforms(img, tgt)
                if tgt is not None:
                    tgt['width'], tgt['height'] = img.width, img.height
                transformed_image.append(img)
                transformed_target.append(tgt)
            image, target = transformed_image, transformed_target
        else:
            self.validate_raw_item(item)  # only validate for single image.
            if self.transforms is not None and image is not None:
                image, target = self.transforms(image, target)
            if target is not None:
                target['width'], target['height'] = image.width, image.height

        # preprocess
        raw_conv = self.process_conv(raw_conv)
        raw_conv, image = self.process_conv_multimage(raw_conv, image)
        raw_conv, _ = self.process_target(raw_conv, target, multimage_mode=multimage_mode)
        conv = self.build_conv(raw_conv)
        text_dict = self.process_text(conv)
        image_dict = self.process_image(image)


        # return
        ret_dict = {}
        ret_dict.update(text_dict)
        ret_dict.update(image_dict)
        self._print_sample(ret_dict, raw_conv, conv)

        if debug_mode:
            return {'ret': ret_dict, 'raw_conv': raw_conv, 'conv': conv, 'image': image}
        return ret_dict

    def __len__(self):
        raise NotImplementedError

    # noinspection PyMethodMayBeStatic
    def process_conv_multimage(self, raw_conv, image):
        # re-sort multi image
        if image is None:
            return raw_conv, image
        if not isinstance(image, (list, tuple)):
            return raw_conv, image
        image_seqs = []
        for conv in raw_conv:
            image_seqs.extend(conv['image_seq'] if 'image_seq' in conv else [])
        images = []
        for idx in image_seqs:
            images.append(image[idx])
        return raw_conv, images

    def get_raw_item(self, index) -> Dict[str, Any]:
        """
        return item format like this.
        item = {
            'image': # PIL.Image.Image,
            'target': {
                # xmin, ymin, xmax, ymax
                'boxes': [
                    [10, 10, 256, 265],  # dog1
                    [24, 18, 378, 768],  # dog2
                    [100, 310, 670, 653],  # man
                    [278, 320, 809, 673],  # rope
                ],
            }

            "conversations": [
                {
                    'from': 'human',
                    'value': 'What is the relation between the two dogs <boxes> and the man <boxes> in the image <image> ?',
                    'boxes_seq': [[0, 1], [2], ],
                },
                {
                    'from': 'gpt',
                    'value': 'a rope <boxes> is connecting the left dog <boxes> with the man <boxes>. '
                             'So the man <boxes> is walking the dog <boxes>.'
                            'And the man <boxes> has no relationship with the right dog <boxes>',
                    'boxes_seq': [[3], [0], [2], [2], [0], [2], [1]],
                }
            ]
        }
        # placeholder: <image> <boxes>
        """
        raise NotImplementedError

    # noinspection PyMethodMayBeStatic
    def validate_raw_item(self, item):
        has_image = 'image' in item
        has_target = 'target' in item
        has_target_boxes = 'boxes' in item['target'] if has_target else False
        raw_conv: List[Dict[str, Any]] = item['conversations']

        # check image
        human_input_has_image_placeholder = any(
            sentence['from'] == 'human' and IMAGE_PLACEHOLDER in sentence['value'] for sentence in raw_conv
        )
        if human_input_has_image_placeholder:
            assert has_image
        if has_image and (not human_input_has_image_placeholder):
            warnings.warn(f'item has image but the question has no image placeholder.\n{item}')
        gpt_input_has_image_placeholder = any(
            sentence['from'] == 'gpt' and IMAGE_PLACEHOLDER in sentence['value'] for sentence in raw_conv
        )
        assert not gpt_input_has_image_placeholder

        # check target
        has_boxes_placeholder = any(
            BOXES_PLACEHOLDER in sentence['value'] for sentence in raw_conv
        )
        if has_boxes_placeholder:
            assert has_target_boxes
        # not check box placeholder num this will be checked in format process

    def build_conv(self, source: List[Dict[str, Any]], mode='common') -> Conversation:
        if mode == 'common':
            conv = self.conv_template()
        elif mode == 'icl':
            conv = self.conv_template_icl()
        else:
            conv = self.conv_template[mode]()

        role_map = {"human": conv.roles[0], "gpt": conv.roles[1]}
        assert len(source) > 0
        assert source[0]['from'] == 'human'
        for sentence in source:
            role = role_map[sentence['from']]
            conv.append_message(role, sentence['value'])
        return conv


    def process_conv(self, raw_conv: List[Dict[str, Any]],mode='common') -> List[Dict[str, Any]]:
        """
        some utils preprocess for raw_conv.
            e.g. replace <image> placeholder to sequence <im_start> <im_patch>*256 <im_end>
        """
        if mode == 'common':
            conv = self.conv_template
        elif mode == 'icl':
            conv = self.conv_template_icl()
        else:
            conv = self.conv_template[mode]()

        return self.process_func['conv'](raw_conv, self.preprocessor, conv)


    def process_target(self, raw_conv: List[Dict[str, Any]], target: Dict[str, Any], multimage_mode=False) -> Tuple[
        List[Dict[str, Any]], Dict[str, Any]]:
        """
        convert target placeholder to actual information in raw_conv.
            e.g. normalize bounding boxes; convert bounding boxes format; replace <boxes> placeholder
        """
        return self.process_func['target'](raw_conv, target, self.preprocessor, multimage_mode=multimage_mode)

    def process_text(self, conv: Conversation , mode=None, icl=False) -> Dict[str, Any]:
        """
        convert Conversation object to torch.Tensor, e.g. input_ids, labels, attention_mask, etc.
            self.tokenize_kwargs control something like padding/truncation behavior.
        """
        if mode is None:
            mode = self.mode
        return self.process_func['text'](conv, self.preprocessor, mode, icl, **self.tokenize_kwargs)

    def process_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        convert Image.Image object to torch.Tensor
        """
        return self.process_func['image'](image, self.preprocessor)

    def _print_sample(self, ret_dict, raw_conv, conv):
        if not hasattr(self, '_printed_sample'):
            self._printed_sample = True
            post_processed_labels = post_process_generate_ids(self.preprocessor['text'], ret_dict['labels'])
            print(f"=================== {self.mode} sample ===================", flush=True)
            print(f"        input_ids: {self.preprocessor['text'].convert_ids_to_tokens(ret_dict['input_ids'])}")
            print(f"           labels: {self.preprocessor['text'].convert_ids_to_tokens(post_processed_labels)}")
            print(f"decoded input_ids: {self.preprocessor['text'].decode(ret_dict['input_ids'])}")
            print(f"decoded    labels: {self.preprocessor['text'].decode(post_processed_labels)}")
            if 'image' in ret_dict and ret_dict['image'] is not None:
                image = ret_dict['image']
                if isinstance(image, torch.Tensor):
                    print(f"            image: {image.shape}")
                elif isinstance(image, dict):
                    print(f"            image: {image.keys()}")
                elif isinstance(image, list) and len(image) > 0:
                    print(f"            image: {len(image)}, {type(image[0])}")
                else:
                    print(f"            image: {type(image)}")
            print("====================================================", flush=True)
            try:
                if self.training_args is not None:
                    _save_obj = {
                        'ret_dict': ret_dict,
                        'raw_conv': raw_conv,
                        'conv': conv.get_prompt(),
                    }
                    from pathlib import Path
                    output_dir = Path(self.training_args.output_dir)
                    output_dir.mkdir(exist_ok=True, parents=True)
                    _local_rank = self.training_args.local_rank
                    _word_size = self.training_args.world_size
                    _file_path = str(output_dir / f'sample_check_{self.mode}_{_local_rank}_{_word_size}.pt')
                    print(f'saving some sample to {_file_path} for check.')
                    torch.save(_save_obj, _file_path)
            except Exception as e:
                warnings.warn(f'try to save samples but get exception: {e.args}. ignored.')


class SingleImageConvDataset(SingleImageConvDatasetMixin, Dataset):
    _repr_indent = 4

    def __init__(self, *args, dataset_generator: Type[Dataset], **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_generator = dataset_generator
        self.dataset = None

    def initialize_if_needed(self):
        """
        lazy initialize for big in-memory python object due to python 'copy-on-read' behavior
        when num_worker > 0. refer: https://github.com/pytorch/pytorch/issues/13246
        """
        if self.dataset is None:
            warnings.warn("it's highly recommended that set persistent_workers=True, "
                          "otherwise this initialize code will run in every epoch beginning."
                          "(ignore me if set)")
            self.dataset = self.dataset_generator()

    def __len__(self):
        self.initialize_if_needed()
        return len(self.dataset)

    def get_raw_item(self, index) -> Dict[str, Any]:
        self.initialize_if_needed()
        return self.dataset[index]

    def get_raw_icl_item(self, index, shot) -> Dict[str, Any]:
        self.initialize_if_needed()
        return self.dataset.__get_icl_item__(index,shot)
    
    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [
            f"Number of datapoints: {self.__len__()}",
        ]
        body += self.dataset.__repr__().splitlines()
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)


__all__ = ['SingleImageConvDatasetMixin', 'SingleImageConvDataset']
