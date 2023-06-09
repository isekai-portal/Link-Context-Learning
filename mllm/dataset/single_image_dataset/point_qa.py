from ..root import (
    DATASETS,
    QUESTION_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
    BOXES_PLACEHOLDER,
    POINTS_PLACEHOLDER,
)
from ..utils import MInstrDataset


# noinspection PyPep8Naming
@DATASETS.register_module()
class Point_QA_local(MInstrDataset):
    def __init__(self, *args, version='p', qbp_p_prob=0.5, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER,))
        assert version in ['b', 'p', 'bp']
        self.version = version
        self.qbp_p_prob = qbp_p_prob

    def __getitem__(self, index):
        item = self.get_raw_item(index)
        # image
        img_path = item['file_path']
        image = self.get_image(img_path)
        # answer
        answer = item['answer']
        # question
        question = item['question']
        bbox = item['bbox']
        point = item['point']

        version = self.version
        if version == 'bp':
            version = 'p' if self.rng.random() < self.qbp_p_prob else 'b'
        if version == 'b':
            question = question + BOXES_PLACEHOLDER
            query_boxes_seq = [[0]]
            query_points_seq = None
        elif version == 'p':
            question = question + POINTS_PLACEHOLDER
            query_boxes_seq = None
            query_points_seq = [[0]]
        else:
            assert False
        final_question = self.get_template().replace(QUESTION_PLACEHOLDER, question)

        ret = {
            'image': image,
            'target': {
                'boxes': [bbox],
                'points': [point],
            },
            'conversations': [
                {
                    'from': 'human',
                    'value': final_question,
                    'boxes_seq': query_boxes_seq,
                    'points_seq': query_points_seq,
                },
                {
                    'from': 'gpt',
                    'value': f'Answer: {answer} .',
                }
            ]
        }
        return ret


# noinspection PyPep8Naming
@DATASETS.register_module()
class Point_QA_twice(MInstrDataset):
    def __init__(self, *args, version='gq-p', bp_p_prob=0.5, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER,))
        self.version = version
        self.bp_p_prob = bp_p_prob
        qtype, rtype = version.split('-')
        assert qtype in ['oq', 'sq', 'gq']
        assert rtype in ['b', 'p', 'bp']
        self.qtype = qtype
        self.rtype = rtype

    def __getitem__(self, index):
        item = self.get_raw_item(index)
        # image
        img_path = item['file_path']
        image = self.get_image(img_path)
        # answer
        answer = item['answer']
        # question
        bbox = item['bbox']
        point = item['point']
        if self.qtype == 'oq':
            question = item['obj_question']
        elif self.qtype == 'sq':
            question = item['super_question']
        elif self.qtype == 'gq':
            question = item['general_question']
        else:
            assert False
        rtype = self.rtype
        if rtype == 'bp':
            rtype = 'p' if self.rng.random() < self.bp_p_prob else 'b'
        if rtype == 'p':
            question = question + POINTS_PLACEHOLDER
            query_boxes_seq = None
            query_points_seq = [[0]]
        elif rtype == 'b':
            question = question + BOXES_PLACEHOLDER
            query_boxes_seq = [[0]]
            query_points_seq = None
        else:
            assert False
        final_question = self.get_template().replace(QUESTION_PLACEHOLDER, question)

        ret = {
            'image': image,
            'target': {
                'boxes': [bbox],
                'points': [point],
            },
            'conversations': [
                {
                    'from': 'human',
                    'value': final_question,
                    'boxes_seq': query_boxes_seq,
                    'points_seq': query_points_seq,
                },
                {
                    'from': 'gpt',
                    'value': f'Answer: {answer} .',
                }
            ]
        }
        return ret


# noinspection PyPep8Naming
@DATASETS.register_module()
class V7W_POINT(MInstrDataset):
    def __init__(self, *args, version, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER,))
        self.version = version
        assert version in ['p', 'b']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.get_raw_item(index)
        # image
        img_path = item['file_path']
        image = self.get_image(img_path)
        # question
        bboxes = item['candidates']
        points = []
        final_question = item['question'] + ' Candidates: ' + BOXES_PLACEHOLDER * len(bboxes)
        query_boxes_seq = []
        for _ in range(len(bboxes)):
            query_boxes_seq.append([_])
        # answer
        if self.version == 'p':
            final_question += f" answer in point format."
            points.append(item['point'])
            final_answer = f"Answer: {POINTS_PLACEHOLDER} ."
            answer_boxes_seq = None
            answer_points_seq = [[0]]
        elif self.version == 'b':
            final_question += f" answer in box format."
            idx = bboxes.index(item['answer'])
            final_answer = f"Answer: {BOXES_PLACEHOLDER} ."
            answer_boxes_seq = [[idx]]
            answer_points_seq = None
        else:
            assert False

        ret = {
            'image': image,
            'target': {
                'boxes': bboxes,
                'points': points,
            },
            'conversations': [
                {
                    'from': 'human',
                    'value': final_question,
                    'boxes_seq': query_boxes_seq,
                },
                {
                    'from': 'gpt',
                    'value': final_answer,
                    'boxes_seq': answer_boxes_seq,
                    'points_seq': answer_points_seq,

                }
            ]
        }
        return ret
