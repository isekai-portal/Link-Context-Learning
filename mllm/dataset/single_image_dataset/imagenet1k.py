from audioop import reverse
from audioop import reverse
import imp
from pydoc import classname
import sys
import logging
import warnings
import os
import os.path as osp
import jsonlines
import random
import string
from inspect import isfunction
from typing import Dict, Any, Sequence
from .icl_train import ICLTrainDataset, logger
from .icl_eval import ICLEvalDataset, ICLComputeMetrics
from ..root import (
    DATASETS,
    METRICS,
    EXPR_PLACEHOLDER
)
LABEL_PLACEHOLDER = "<label>"

@DATASETS.register_module()
class ImageNet1kDatasetTrain(ICLTrainDataset):
    def __init__(self, policy: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy

    # get policy function according to name 
    def __get_icl_item__(self, index, shot):
        func = getattr(self, self.policy)
        assert func is not None
        return func(index, shot)

    def policy_v1(self):
        raise NotImplementedError

    def policy_v2(self):
        raise NotImplementedError

    def policy_v3(self, index, shot):
        ret_list = []
        normal_question = 'What is the class of the image <image>?'
        real_question = 'What is the "real" class of the image <image>?'

        for _ in range(shot):
            image, label = self.get_samples(index, mode = "cls_negative")
            ret = self.get_ret(image, question=normal_question, answer=label)
            ret_list.append(ret)

        for _ in range(shot):
            image, label = self.get_samples(index, mode = "neighbors")
            ret = self.get_ret(image, question=normal_question, answer=label)
            ret_list.append(ret)

        random.shuffle(ret_list)
        # inference sample
        mode = random.choice(["cls_positive", "cls_negative", "neighbors"])
        if mode == "cls_positive":
            question = real_question
        elif mode == "cls_negative":
            question = normal_question
        elif mode == "neighbors":
            question = normal_question
        
        image, label = self.get_samples(index, mode = mode)
        ret = self.get_ret(image, question=question, answer=label)
        ret_list.append(ret)
        self.cls_neg_label = None
        return ret_list

    def policy_v4(self, index, shot):
        ret_list = []
        bind_question = 'What is the "binding" class of the image <image>?'
        real_question = 'What is the "real" class of the image <image>?'

        for _ in range(shot):
            image, label = self.get_samples(index, mode = "cls_negative")
            ret = self.get_ret(image, question=bind_question, answer=label)
            ret_list.append(ret)
            
        for _ in range(shot):
            question = random.choice([bind_question, real_question])
            image, label = self.get_samples(index, mode = "neighbors")
            ret = self.get_ret(image, question=question, answer=label)
            ret_list.append(ret)
        
        random.shuffle(ret_list)
        # inference sample
        mode = random.choice(["cls_positive", "cls_negative", "neighbors"])
        if mode == "cls_positive":
            question = real_question
        elif mode == "cls_negative":
            question = bind_question
        elif mode == "neighbors":
            question = random.choice([bind_question, real_question])

        image, label = self.get_samples(index, mode = mode)
        ret = self.get_ret(image, question=question, answer=label)
        ret_list.append(ret)
        self.cls_neg_label = None
        return ret_list

    def policy_v5(self, index, shot):
        ret_list = []
        normal_question = 'What is the class of the image <image>?'
        bind_question = 'What is the "binding" class of the image <image>?'
        real_question = 'What is the "real" class of the image <image>?'

        for _ in range(shot):
            image, label = self.get_samples(index, mode = "cls_negative")
            ret = self.get_ret(image, question=bind_question, answer=label)
            ret_list.append(ret)

        for _ in range(shot):
            image, label = self.get_samples(index, mode = "neighbors")
            ret = self.get_ret(image, question=real_question, answer=label)
            ret_list.append(ret)
        
        random.shuffle(ret_list)
        mode = random.choice(["cls_positive", "cls_negative", "neighbors"])
        if mode == "cls_positive":
            question = real_question
        elif mode == "cls_negative":
            question = normal_question
        elif mode == "neighbors":
            question = normal_question

        image, label = self.get_samples(index, mode = mode)
        ret = self.get_ret(image, question=question, answer=label)
        ret_list.append(ret)
        self.cls_neg_label = None
        return ret_list

    # v6
    def policy_v6(self, index, shot):
        random_string = None
        def _convert_qa(question, label, mode):
            nonlocal random_string
            assert mode in ['cls_positive','cls_negative', 'neighbors']
            
            answer = f'there is {LABEL_PLACEHOLDER} in the image'

            if mode == "cls_positive":
                # current class image, current label
                question = question.replace(LABEL_PLACEHOLDER, label)
                answer = answer.replace(LABEL_PLACEHOLDER, label)
            elif mode == "cls_negative":
                # current class image, random string or current label
                if random_string:
                    label = random_string
                else:
                    if random.random() < 0.5:
                        label = ''.join(random.choices(\
                            string.ascii_uppercase, k=random.randint(4,8))).lower()
                    random_string = label
                question = question.replace(LABEL_PLACEHOLDER, random_string)
                answer = answer.replace(LABEL_PLACEHOLDER, random_string)
            elif mode == "neighbors":
                # random neighbor image and label
                assert random_string is not None
                question = question.replace(LABEL_PLACEHOLDER, random_string)
                answer = answer.replace(LABEL_PLACEHOLDER, 'no '+ random_string)
            else:
                raise NotImplementedError

            return question, answer

        ret_list = []
        correct_question = 'Forget about the previous conversation, Is there any <label> in the image <image> ?'
        mix_question = 'What is the charateristic about the image <image> ?'
        infer_question = 'Is there any <label> in the image <image> according to the previous conversation ?'

        shot = random.randint(1, shot)
        for _ in range(shot):
            image, label = self.get_samples(index, mode = 'cls_positive')
            # convert correct label to random string(or not)
            question, answer = _convert_qa(mix_question, label, mode = 'cls_negative')
            ret = self.get_ret(image, question = question, answer = answer)
            ret_list.append(ret)

        for _ in range(shot):
            image, label = self.get_samples(index, mode = 'neighbors')
            question, answer = _convert_qa(mix_question, label = None, mode = 'neighbors')#use random string
            ret = self.get_ret(image, question = question, answer = answer)
            ret_list.append(ret)
        
        random.shuffle(ret_list)
        mode = random.choice(["cls_positive","cls_negative","neighbors"])
        if mode == "cls_positive":
            question = correct_question
            image, label = self.get_samples(index, mode = mode)
            question, answer = _convert_qa(question, label, mode = mode)
        elif mode == "cls_negative":
            question = infer_question
            # need correct label and optional convert to random string
            image, label = self.get_samples(index, mode = "cls_positive")
            question, answer = _convert_qa(question, label, mode = "cls_negative")
        elif mode == "neighbors":
            question = infer_question
            image, label = self.get_samples(index, mode = mode)
            question, answer = _convert_qa(question, label = None, mode = mode)#use random string

        ret = self.get_ret(image, question = question, answer = answer)
        ret_list.append(ret)
        random_string = None
        self.cls_neg_label = None
        return ret_list

    # v7
    def policy_v7(self, index, shot):
        # Same to v6, only modified the shot to pos_shot and neg_shot
        random_string = None
        def _convert_qa(question, label, mode):
            nonlocal random_string
            assert mode in ['cls_positive','cls_negative', 'neighbors']
            
            answer = f'there is {LABEL_PLACEHOLDER} in the image'

            if mode == "cls_positive":
                # current class image, current label
                question = question.replace(LABEL_PLACEHOLDER, label)
                answer = answer.replace(LABEL_PLACEHOLDER, label)
            elif mode == "cls_negative":
                # current class image, random string or current label
                if random_string:
                    label = random_string
                else:
                    if random.random() < 0.5:
                        label = ''.join(random.choices(\
                            string.ascii_uppercase, k=random.randint(4,8))).lower()
                    random_string = label
                question = question.replace(LABEL_PLACEHOLDER, random_string)
                answer = answer.replace(LABEL_PLACEHOLDER, random_string)
            elif mode == "neighbors":
                # random neighbor image and label
                assert random_string is not None
                question = question.replace(LABEL_PLACEHOLDER, random_string)
                answer = answer.replace(LABEL_PLACEHOLDER, 'no '+ random_string)
            else:
                raise NotImplementedError

            return question, answer

        ret_list = []
        correct_question = 'Forget about the previous conversation, Is there any <label> in the image <image> ?'
        mix_question = 'What is the charateristic about the image <image> ?'
        infer_question = 'Is there any <label> in the image <image> according to the previous conversation ?'

        pos_shot = random.randint(1, shot)
        neg_shot = random.randint(0, shot)
        for _ in range(pos_shot):
            image, label = self.get_samples(index, mode = 'cls_positive')
            # convert correct label to random string(or not)
            question, answer = _convert_qa(mix_question, label, mode = 'cls_negative')
            ret = self.get_ret(image, question = question, answer = answer)
            ret_list.append(ret)

        for _ in range(neg_shot):
            image, label = self.get_samples(index, mode = 'neighbors')
            question, answer = _convert_qa(mix_question, label = None, mode = 'neighbors')#use random string
            ret = self.get_ret(image, question = question, answer = answer)
            ret_list.append(ret)
        
        random.shuffle(ret_list)
        mode = random.choice(["cls_positive","cls_negative","neighbors"])
        if mode == "cls_positive":
            question = correct_question
            image, label = self.get_samples(index, mode = mode)
            question, answer = _convert_qa(question, label, mode = mode)
        elif mode == "cls_negative":
            question = infer_question
            # need correct label and optional convert to random string
            image, label = self.get_samples(index, mode = "cls_positive")
            question, answer = _convert_qa(question, label, mode = "cls_negative")
        elif mode == "neighbors":
            question = infer_question
            image, label = self.get_samples(index, mode = mode)
            question, answer = _convert_qa(question, label = None, mode = mode)#use random string

        ret = self.get_ret(image, question = question, answer = answer)
        ret_list.append(ret)
        random_string = None
        self.cls_neg_label = None
        return ret_list

    # v8
    def policy_v8(self, index, shot):
        random_string = None
        def _convert_qa(question, label, mode):
            nonlocal random_string
            assert mode in ['cls_positive','cls_negative', 'neighbors']
            
            answer = f'there is {LABEL_PLACEHOLDER} in the image'

            if mode == "cls_positive":
                # current class image, current label
                question = question.replace(LABEL_PLACEHOLDER, label)
                answer = answer.replace(LABEL_PLACEHOLDER, label)
            elif mode == "cls_negative":
                # current class image, random string or current label
                if random_string:
                    label = random_string
                else:
                    if random.random() < 0.5:
                        label = ''.join(random.choices(\
                            string.ascii_uppercase, k=random.randint(4,8))).lower()
                    random_string = label
                question = question.replace(LABEL_PLACEHOLDER, random_string)
                answer = answer.replace(LABEL_PLACEHOLDER, random_string)
            elif mode == "neighbors":
                # random neighbor image and label
                assert random_string is not None
                question = question.replace(LABEL_PLACEHOLDER, random_string)
                answer = answer.replace(LABEL_PLACEHOLDER, 'no '+ random_string)
            else:
                raise NotImplementedError

            return question, answer

        ret_list = []
        correct_question = 'Forget about the previous conversation, Is there any <label> in the image <image> ?'
        mix_question = 'What is the charateristic about the image <image> ?'
        infer_question = 'Is there any <label> in the image <image> according to the previous conversation ?'

        for _ in range(shot):
            image, label = self.get_samples(index, mode = 'cls_positive')
            # convert correct label to random string(or not)
            question, answer = _convert_qa(mix_question, label, mode = 'cls_negative')
            ret = self.get_ret(image, question = question, answer = answer)
            ret_list.append(ret)

        for _ in range(shot):
            image, label = self.get_samples(index, mode = 'neighbors')
            question, answer = _convert_qa(mix_question, label = None, mode = 'neighbors')#use random string
            ret = self.get_ret(image, question = question, answer = answer)
            ret_list.append(ret)
        
        random.shuffle(ret_list)
        mode = random.choice(["cls_positive","cls_negative","neighbors"])
        if mode == "cls_positive":
            question = correct_question
            image, label = self.get_samples(index, mode = mode)
            question, answer = _convert_qa(question, label, mode = mode)
        elif mode == "cls_negative":
            question = infer_question
            # need correct label and optional convert to random string
            image, label = self.get_samples(index, mode = "cls_positive")
            question, answer = _convert_qa(question, label, mode = "cls_negative")
        elif mode == "neighbors":
            question = infer_question
            image, label = self.get_samples(index, mode = mode)
            question, answer = _convert_qa(question, label = None, mode = mode)#use random string

        ret = self.get_ret(image, question = question, answer = answer)
        ret_list.append(ret)
        random_string = None
        self.cls_neg_label = None
        return ret_list
    
    #v9
    def policy_v9(self, index, shot):
        random_string = None
        def _convert_qa(question, label, mode):
            nonlocal random_string
            assert mode in ['cls_negative', 'neighbors']
            
            answer = f'there is {LABEL_PLACEHOLDER} in the image'

            if mode == "cls_negative":
                # current class image, random string or current label
                if random_string:
                    #label = random_string
                    pass
                else:
                    random_string = ''.join(random.choices(\
                        string.ascii_uppercase, k=random.randint(1,10))).lower()
                answer = answer.replace(LABEL_PLACEHOLDER, random_string)
            elif mode == "neighbors":
                answer = answer.replace(LABEL_PLACEHOLDER, label)
            else:
                raise NotImplementedError

            return question, answer

        ret_list = []
        mix_question = '[INSTRUCTION] What is in the image <image> ?'
        infer_question = '[INFERENCE] What is in the image <image> ?'

        #shot = random.randint(1, shot)
        for _ in range(shot):
            image, label = self.get_samples_same_cls(index, mode = 'cls_negative')
            # convert correct label to random string(or not)
            question, answer = _convert_qa(mix_question, label, mode = 'cls_negative')
            ret = self.get_ret(image, question = question, answer = answer)
            ret_list.append(ret)

        for _ in range(shot):
            image, label = self.get_samples_same_cls(index, mode = 'neighbors')
            question, answer = _convert_qa(mix_question, label, mode = 'neighbors')#use random string
            ret = self.get_ret(image, question = question, answer = answer)
            ret_list.append(ret)
        
        random.shuffle(ret_list)
        mode = random.choice(["cls_negative","neighbors"])
        if mode == "cls_negative":
            question = infer_question
            # need correct label and optional convert to random string
            image, label = self.get_samples_same_cls(index, mode = "cls_negative")
            question, answer = _convert_qa(question, label, mode = "cls_negative")
        elif mode == "neighbors":
            question = infer_question
            image, label = self.get_samples_same_cls(index, mode = mode)
            question, answer = _convert_qa(question, label, mode = mode)#use random string

        ret = self.get_ret(image, question = question, answer = answer)
        ret_list.append(ret)
        random_string = None
        self.cls_neg_label = None
        self.cls_idx = None
        self.cls_name = None
        return ret_list


@DATASETS.register_module()
class ImageNet1kDatasetEval(ICLEvalDataset):
    def __init__(self, policy: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy

    def __get_icl_item__(self, index, shot):
        func = getattr(self, self.policy)
        assert func is not None
        class_name, context_imgs, test_img = self.get_samples(index, shot)
        question, answer = func(class_name, index = index)
        ret_list = []
        # context sample
        for i in range(shot):
            ret_list.append(self.get_ret(context_imgs[i], question=question, answer=answer))
        
        # eval sample
        ret_list.append(self.get_ret(test_img, question=question, answer=answer))
        return ret_list

    def policy_v1(self, class_name, **kargs):
        raise NotImplementedError

    def policy_v2(self, class_name, **kargs):
        raise NotImplementedError
    
    def policy_v3(self, class_name, **kargs):
        question = 'What is the class of the image <image>?'
        answer = class_name
        return question, answer

    def policy_v4(self, class_name, **kargs):
        question = 'What is the "binding" class of the image <image>?'
        answer = class_name
        return question, answer
    
    def policy_v5(self, class_name, **kargs):
        question = 'What is the class of the image <image>?'
        answer = class_name
        return question, answer

    def policy_v6(self, class_name, **kargs):
        question = f'Is there any {class_name} in the image <image> according to the previous conversation ?'
        answer = f'there is {class_name} in the image'
        return question, answer

    def policy_v7(self, class_name, **kargs):
        return self.policy_v6(class_name)

    def policy_v8(self, class_name, **kargs):
        return self.policy_v6(class_name)


@DATASETS.register_module()
class ImageNet1k1WayEval(ICLEvalDataset):
 
    def _rearrange(self):
        wrap_map = []
        data_map = super()._rearrange() #[cls_idx, sample_idx]
        cls_num = len(self.data)
        for i in range(cls_num):
            for data in data_map:
                wrap_map.append([i, data])
        return wrap_map

    def get_samples(self, index, shot):
        ctx_idx, sample_map = self.data_map[index]

        # context samples
        ctx_item = self.get_raw_item(ctx_idx)
        ctx_name = ctx_item["class_name"].lower()
        ctx_samples = ctx_item["context_samples"]
        ctx_imgs = []
        for i in range(shot):
            ctx_imgs.append(self.get_image(ctx_samples[i]))

        # inference sample
        cls_idx, sample_idx = sample_map
        cls_item = self.get_raw_item(cls_idx)
        cls_id = cls_item["class_id"]
        cls_name = cls_item["class_name"].lower()
        cls_samples = cls_item['test_samples']
        infer_img = self.get_image(cls_samples[sample_idx])

        sample_meta = dict(
            ctx_name = ctx_name,
            infer_name = cls_name,
            ctx_imgs = ctx_imgs,
            infer_img = infer_img
            )
        return sample_meta

    def __get_icl_item__(self, index, shot):
        func = getattr(self, self.policy)
        assert func is not None
        
        sample_meta = self.get_samples(index, shot)
        QnA = func(sample_meta["ctx_name"], sample_meta["infer_name"])
        
        ret_list = []
        # context sample
        for img in sample_meta["ctx_imgs"]:
            ret_list.append(self.get_ret(img, question=QnA["ctx_question"], answer=QnA["ctx_answer"]))

        # inference
        ret_list.append(self.get_ret(sample_meta["infer_img"], question=QnA["infer_question"], answer=QnA["infer_answer"])) 
        return ret_list

    def policy_v9(self, ctx_name, infer_name):
        ctx_question = '[INSTRUCTION] What is in the image <image> ?'
        infer_question = '[INFERENCE] What is in the image <image> ?'

        answer = f'there is {LABEL_PLACEHOLDER} in the image'
        ctx_answer = answer.replace(LABEL_PLACEHOLDER, ctx_name)
        infer_answer = answer.replace(LABEL_PLACEHOLDER, infer_name)

        return dict(
            ctx_question = ctx_question, 
            ctx_answer = ctx_answer,
            infer_question = infer_question,
            infer_answer = infer_answer
        )

# context: n-positive samples(class A) + n-negative samples(class B)
# inference: class A or B sample 
@DATASETS.register_module()
class ImageNet1k2WayYNEval(ICLEvalDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.data)%2 == 0
        self.reverse_flag = len(self.data_map)
        self.data_map.extend(self.data_map)

    def rotate_index(self, index):
        # all sample inference twice
        # if reverse = true, context reverse
        reverse = index >= self.reverse_flag
        if reverse:
            index = index - self.reverse_flag
        return index, reverse

    def get_samples(self, index, shot):
        cls_idx, sample_idx = self.data_map[index]
        item = self.get_raw_item(cls_idx)
        class_id = item["class_id"]
        pos_cls_name = item["class_name"].lower()
        test_samples = item['test_samples']

        # construct positive and negtive pairs
        if cls_idx % 2 == 0:
            neg_cls_idx = cls_idx + 1
        else:
            neg_cls_idx = cls_idx - 1
        neg_item = self.get_raw_item(neg_cls_idx)
        pos_samples = item["context_samples"]
        neg_samples = neg_item["context_samples"]
        neg_cls_name = neg_item["class_name"].lower()
        
        pos_imgs, neg_imgs = [], []
        for i in range(shot):
            pos_imgs.append(self.get_image(pos_samples[i]))
            neg_imgs.append(self.get_image(neg_samples[i]))            
        # inference sample (positive class)
        infer_img = self.get_image(test_samples[sample_idx])

        sample_meta = dict(
            pos_cls_name = pos_cls_name,
            neg_cls_name = neg_cls_name,
            pos_imgs = pos_imgs,
            neg_imgs = neg_imgs,
            infer_img = infer_img
            )
        return sample_meta

    def __get_icl_item__(self, index, shot):
        func = getattr(self, self.policy)
        assert func is not None
        
        index, reverse = self.rotate_index(index)
        sample_meta = self.get_samples(index, shot)

        if reverse:
            QnA = func(sample_meta["neg_cls_name"])
            QnA["infer_answer"] = QnA["infer_answer"].replace(sample_meta["neg_cls_name"], 'no '+ sample_meta["neg_cls_name"])
        else:
            QnA = func(sample_meta["pos_cls_name"])
        
        ret_list = []

        # context sample
        # Note: The traversal process is conducted on a per-class basis.
        # not reverse: pos A image(text: there is A) + neg B image(text: there is no A) + infer A image(label: there is A)
        # reverse: pos A image(text: there is no B) + neg B image(text: there is B) + infer A image(label: there is no B)
        for img in sample_meta["pos_imgs"]:
            if reverse:
                ret_list.append(self.get_ret(img, question=QnA["neg_question"], answer=QnA["neg_answer"]))
            else:
                ret_list.append(self.get_ret(img, question=QnA["pos_question"], answer=QnA["pos_answer"]))
        
        for img in sample_meta["neg_imgs"]:
            if reverse:
                ret_list.append(self.get_ret(img, question=QnA["pos_question"], answer=QnA["pos_answer"]))
            else:
                ret_list.append(self.get_ret(img, question=QnA["neg_question"], answer=QnA["neg_answer"]))
        random.shuffle(ret_list)

        # inference
        ret_list.append(self.get_ret(sample_meta["infer_img"], question=QnA["infer_question"], answer=QnA["infer_answer"])) 
        return ret_list

    def policy_v6(self, cls_name):
        pos_question = 'What is the charateristic about the image <image> ?'
        neg_question = pos_question
        infer_question = f'Is there any {cls_name} in the image <image> according to the previous conversation ?'

        answer = f'there is {LABEL_PLACEHOLDER} in the image'
        pos_answer = answer.replace(LABEL_PLACEHOLDER, cls_name)
        neg_answer = answer.replace(LABEL_PLACEHOLDER, 'no '+ cls_name)
        infer_answer = pos_answer

        return dict(
            pos_question = pos_question, 
            neg_question = neg_question,
            infer_question = infer_question,
            pos_answer = pos_answer,
            neg_answer = neg_answer, 
            infer_answer = infer_answer
        )

    def policy_v9(self, cls_name):
        pos_question = '[INSTRUCTION] What is in the image <image> ?'
        neg_question = pos_question
        infer_question = '[INFERENCE] What is in the image <image> ?'

        answer = f'there is {LABEL_PLACEHOLDER} in the image'
        pos_answer = answer.replace(LABEL_PLACEHOLDER, cls_name)
        neg_answer = answer.replace(LABEL_PLACEHOLDER, 'no'+ cls_name)
        infer_answer = pos_answer

        return dict(
            pos_question = pos_question, 
            neg_question = neg_question,
            infer_question = infer_question,
            pos_answer = pos_answer,
            neg_answer = neg_answer, 
            infer_answer = infer_answer
        )
    
    def policy_v7(self, cls_name):
        return self.policy_v6(cls_name)

    def policy_v8(self, cls_name):
        return self.policy_v6(cls_name)

# context: n-positive samples(class A) + n-negative samples(class B)
# inference: class A or B sample 
@DATASETS.register_module()
class ImageNet1k2WayEval(ICLEvalDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.data)%2 == 0

    def get_samples(self, index, shot):
        cls_idx, sample_idx = self.data_map[index]
        item = self.get_raw_item(cls_idx)
        class_id = item["class_id"]
        pos_cls_name = item["class_name"].lower()
        test_samples = item['test_samples']

        # construct positive and negtive pairs
        if cls_idx % 2 == 0:
            neg_cls_idx = cls_idx + 1
        else:
            neg_cls_idx = cls_idx - 1
        neg_item = self.get_raw_item(neg_cls_idx)
        pos_samples = item["context_samples"]
        neg_samples = neg_item["context_samples"]
        neg_cls_name = neg_item["class_name"].lower()
        
        pos_imgs, neg_imgs = [], []
        for i in range(shot):
            pos_imgs.append(self.get_image(pos_samples[i]))
            neg_imgs.append(self.get_image(neg_samples[i]))            
        # inference sample (positive class)
        infer_img = self.get_image(test_samples[sample_idx])

        sample_meta = dict(
            pos_cls_name = pos_cls_name,
            neg_cls_name = neg_cls_name,
            pos_imgs = pos_imgs,
            neg_imgs = neg_imgs,
            infer_img = infer_img
            )
        return sample_meta

    def __get_icl_item__(self, index, shot):
        func = getattr(self, self.policy)
        assert func is not None
        
        sample_meta = self.get_samples(index, shot)

        QnA = func(sample_meta["pos_cls_name"],sample_meta["neg_cls_name"])
        
        ret_list = []

        # context sample: pos A image(text: there is A) + neg B image(text: there is B) + infer A image(label: there is A)
        for img in sample_meta["pos_imgs"]:
            ret_list.append(self.get_ret(img, question=QnA["pos_question"], answer=QnA["pos_answer"]))
        
        for img in sample_meta["neg_imgs"]:
            ret_list.append(self.get_ret(img, question=QnA["neg_question"], answer=QnA["neg_answer"]))
        random.shuffle(ret_list)

        # inference
        ret_list.append(self.get_ret(sample_meta["infer_img"], question=QnA["infer_question"], answer=QnA["infer_answer"])) 
        return ret_list

    def policy_v9(self, cls_name_pos, cls_name_neg):
        pos_question = '[INSTRUCTION] What is in the image <image> ?'
        neg_question = pos_question
        infer_question = '[INFERENCE] What is in the image <image> ?'

        answer = f'there is {LABEL_PLACEHOLDER} in the image'
        pos_answer = answer.replace(LABEL_PLACEHOLDER, cls_name_pos)
        neg_answer = answer.replace(LABEL_PLACEHOLDER, cls_name_neg)
        infer_answer = pos_answer

        return dict(
            pos_question = pos_question, 
            neg_question = neg_question,
            infer_question = infer_question,
            pos_answer = pos_answer,
            neg_answer = neg_answer, 
            infer_answer = infer_answer
        )
    

# context: n-positive samples(class A)
# inference: class A or B sample 
@DATASETS.register_module()
class ImageNet1kNWayEval(ICLEvalDataset):

    def _rearrange(self):
        # Map dataloader index to self.data, according to class_idx and sample_idx
        data_map = []
        for cls_idx, item in enumerate(self.data):
            test_samples = item['test_samples']
            for sample_idx, sample in enumerate(test_samples):
                # sample_per_class = 0: all samples evaluation
                if sample_idx == self.sample_per_class and \
                    self.sample_per_class > 0:
                    break
                data_map.append([cls_idx, sample_idx])
        return data_map        

    def get_samples(self, index, shot):
        raise NotImplementedError