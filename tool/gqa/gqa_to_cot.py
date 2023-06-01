from typing import Dict, Any, List, Tuple
from root import PHRASE_ST_PLACEHOLDER, PHRASE_ED_PLACEHOLDER, get_box_xyxy
from binary_operator import Gqa2CoTBinaryMixin
from unary_operator import Gqa2CoTUnaryMixin


def build_chains(s, idx=None, debug=False):
    if idx is None:
        idx = len(s) - 1

    chains = []
    if debug:
        print(f"build_chains({idx})")

    while True:
        chains.append(idx)
        arg_size = len(s[idx]['dependencies'])
        if arg_size == 1:
            idx = s[idx]['dependencies'][0]
            continue
        elif arg_size == 0:
            return tuple(chains), ()
        elif arg_size == 2:
            args = []
            for idx in s[idx]['dependencies']:
                args.append(build_chains(s, idx, debug=debug))
            return tuple(chains), tuple(args)
        else:
            assert False


class GQA2CoT(Gqa2CoTBinaryMixin, Gqa2CoTUnaryMixin):

    def __init__(self, ann, scene):
        super().__init__()
        self.ann = ann
        self.operations = ann['semantic']
        self.scene = scene
        self.boxes_list = []
        self.boxes_seq = []
        self.res = []

    def travel(self, chain_arg, func):
        chain, arg = chain_arg
        arg_res = []
        for item in arg:
            arg_res.append(self.travel(item, func))
        return func(chain, arg_res)

    def build_chain_str_func(self, chain, arg_res):
        ret = " ".join(arg_res).strip()

        if len(arg_res) == 2:
            assert len(chain) == 1
            chain_res = self.binary_operator_to_str(chain[0])
        else:
            assert len(arg_res) == 0, f"error {chain} {arg_res}"
            chain_res = self.chain_to_str(chain)

        assert isinstance(chain_res, str), f'chain_res is {chain_res}, input: {chain}, {arg_res}'
        print(f"chain: {chain}\nres: {chain_res}")
        return ret + chain_res

    def __call__(self):
        chains = build_chains(ann['semantic'])
        cot = self.travel(chains, self.build_chain_str_func)
        self.res.append(cot)
        self.add_summary_and_answer_post_process()
        return " ".join(self.res)

    def add_summary_and_answer_post_process(self):
        ann = self.ann
        scene = self.scene

        sent = list(ann['fullAnswer'].split())
        for span, rids in ann['annotations']['fullAnswer'].items():
            span = tuple(map(int, span.split(':')))
            sent[span[0]:span[1]] = [f"{PHRASE_ST_PLACEHOLDER}{' '.join(sent[span[0]:span[1]])}{PHRASE_ED_PLACEHOLDER}"]
            boxes_idx = []
            for rid in rids.split(','):
                ref = scene['objects'][rid]
                ref_box = get_box_xyxy(ref)
                boxes_idx.append(self.get_boxes_idx(ref_box))
            self.boxes_seq.append(boxes_idx)
        sent_converted = " ".join(sent)

        self.res.append(f"So, {sent_converted}")
        self.res.append(f"In summary, the final answer is {ann['answer']}.")

    def get_boxes_idx(self, box):
        if box in self.boxes_list:
            return self.boxes_list.index(box)
        else:
            self.boxes_list.append(box)
            return len(self.boxes_list) - 1


if __name__ == '__main__':
    scene = {'width': 500,
             'objects': {'1259740': {'name': 'pole',
                                     'h': 71,
                                     'relations': [{'object': '1259713', 'name': 'to the right of'},
                                                   {'object': '1259720', 'name': 'to the left of'},
                                                   {'object': '1259730', 'name': 'to the left of'}],
                                     'w': 25,
                                     'attributes': ['yellow'],
                                     'y': 342,
                                     'x': 192},
                         '1259735': {'name': 'sign',
                                     'h': 38,
                                     'relations': [{'object': '1259721', 'name': 'to the left of'},
                                                   {'object': '1259730', 'name': 'to the right of'},
                                                   {'object': '1259716', 'name': 'to the left of'}],
                                     'w': 29,
                                     'attributes': ['green'],
                                     'y': 114,
                                     'x': 365},
                         '1259739': {'name': 'sticker',
                                     'h': 20,
                                     'relations': [{'object': '1259730', 'name': 'to the left of'},
                                                   {'object': '1259713', 'name': 'to the right of'},
                                                   {'object': '1259720', 'name': 'to the left of'}],
                                     'w': 15,
                                     'attributes': ['white'],
                                     'y': 350,
                                     'x': 193},
                         '1259726': {'name': 'grass',
                                     'h': 16,
                                     'relations': [{'object': '1259730', 'name': 'to the right of'},
                                                   {'object': '1259715', 'name': 'near'}],
                                     'w': 23,
                                     'attributes': ['green'],
                                     'y': 154,
                                     'x': 388},
                         '1259724': {'name': 'sign',
                                     'h': 128,
                                     'relations': [{'object': '1259712', 'name': 'to the left of'}],
                                     'w': 21,
                                     'attributes': [],
                                     'y': 221,
                                     'x': 0},
                         '1259730': {'name': 'pole',
                                     'h': 364,
                                     'relations': [{'object': '1259726', 'name': 'to the left of'},
                                                   {'object': '1259721', 'name': 'to the left of'},
                                                   {'object': '1259716', 'name': 'to the left of'},
                                                   {'object': '1259735', 'name': 'to the left of'},
                                                   {'object': '1259720', 'name': 'to the right of'},
                                                   {'object': '1259739', 'name': 'to the right of'},
                                                   {'object': '1259740', 'name': 'to the right of'},
                                                   {'object': '1259715', 'name': 'to the left of'},
                                                   {'object': '1259712', 'name': 'to the right of'}],
                                     'w': 36,
                                     'attributes': ['metal'],
                                     'y': 50,
                                     'x': 320},
                         '1259731': {'name': 'street sign',
                                     'h': 28,
                                     'relations': [{'object': '1259712', 'name': 'in front of'}],
                                     'w': 139,
                                     'attributes': ['black'],
                                     'y': 196,
                                     'x': 54},
                         '1259721': {'name': 'cars',
                                     'h': 14,
                                     'relations': [{'object': '1259730', 'name': 'to the right of'},
                                                   {'object': '1259735', 'name': 'to the right of'}],
                                     'w': 31,
                                     'attributes': ['parked'],
                                     'y': 132,
                                     'x': 407},
                         '1259720': {'name': 'trash can',
                                     'h': 88,
                                     'relations': [{'object': '1259740', 'name': 'to the right of'},
                                                   {'object': '1259713', 'name': 'on'},
                                                   {'object': '1259739', 'name': 'to the right of'},
                                                   {'object': '1259730', 'name': 'to the left of'}],
                                     'w': 34,
                                     'attributes': ['green'],
                                     'y': 326,
                                     'x': 292},
                         '1259716': {'name': 'trees',
                                     'h': 63,
                                     'relations': [{'object': '1259735', 'name': 'to the right of'},
                                                   {'object': '1259730', 'name': 'to the right of'}],
                                     'w': 55,
                                     'attributes': [],
                                     'y': 111,
                                     'x': 395},
                         '1259717': {'name': 'leaves',
                                     'h': 43,
                                     'relations': [],
                                     'w': 27,
                                     'attributes': ['orange'],
                                     'y': 20,
                                     'x': 472},
                         '1259715': {'name': 'road',
                                     'h': 170,
                                     'relations': [{'object': '1259726', 'name': 'near'},
                                                   {'object': '1259730', 'name': 'to the right of'}],
                                     'w': 135,
                                     'attributes': ['black'],
                                     'y': 180,
                                     'x': 364},
                         '1259712': {'name': 'traffic light',
                                     'h': 223,
                                     'relations': [{'object': '1259731', 'name': 'behind'},
                                                   {'object': '1259724', 'name': 'to the right of'},
                                                   {'object': '1259729', 'name': 'to the right of'},
                                                   {'object': '1259713', 'name': 'to the right of'},
                                                   {'object': '1259730', 'name': 'to the left of'},
                                                   {'object': '1259713', 'name': 'on'}],
                                     'w': 108,
                                     'attributes': ['red'],
                                     'y': 136,
                                     'x': 119},
                         '1259713': {'name': 'sidewalk',
                                     'h': 125,
                                     'relations': [{'object': '1259739', 'name': 'to the left of'},
                                                   {'object': '1259712', 'name': 'to the left of'},
                                                   {'object': '1259740', 'name': 'to the left of'}],
                                     'w': 72,
                                     'attributes': [],
                                     'y': 288,
                                     'x': 55},
                         '1259729': {'name': 'pole',
                                     'h': 124,
                                     'relations': [{'object': '1259712', 'name': 'to the left of'}],
                                     'w': 15,
                                     'attributes': ['skinny', 'yellow'],
                                     'y': 164,
                                     'x': 85},
                         '1259711': {'name': 'sign',
                                     'h': 67,
                                     'relations': [],
                                     'w': 260,
                                     'attributes': ['white'],
                                     'y': 56,
                                     'x': 218}},
             'height': 414}

    ann: Dict[str, Any] = {
        'semantic': [{'operation': 'select', 'dependencies': [], 'argument': 'traffic light (1259712)'},
                     {'operation': 'filter color', 'dependencies': [0], 'argument': 'red'},
                     {'operation': 'exist', 'dependencies': [1], 'argument': '?'},
                     {'operation': 'select', 'dependencies': [], 'argument': 'stop sign (-) '},
                     {'operation': 'filter color', 'dependencies': [3], 'argument': 'red'},
                     {'operation': 'exist', 'dependencies': [4], 'argument': '?'},
                     {'operation': 'or', 'dependencies': [2, 5], 'argument': ''}],
        'entailed': ['17124381'], 'equivalent': ['17124380'],
        'question': 'Is there either a red traffic light or stop sign?',
        'imageId': '2389539', 'isBalanced': True, 'groups': {'global': None, 'local': '04-traffic light_red'},
        'answer': 'yes',
        'semanticStr': 'select: traffic light (1259712)->filter color: red [0]->exist: ? [1]->select: stop sign (-) ->filter color: red [3]->exist: ? [4]->or:  [2, 5]',
        'annotations': {'answer': {}, 'question': {'5:7': '1259712'}, 'fullAnswer': {'5:7': '1259712'}},
        'types': {'detailed': 'existAttrOr', 'semantic': 'obj', 'structural': 'logical'},
        'fullAnswer': 'Yes, there is a red traffic light.'}

    gqa2cot = GQA2CoT(ann, scene)
    res = gqa2cot()
    print(res)
    print(gqa2cot.boxes_list)
    print(gqa2cot.boxes_seq)
