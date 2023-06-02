import re
from typing import Dict, Any
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

    def post_order_travel(self, chain_arg, func):
        chain, arg = chain_arg
        arg_res = []
        for item in arg:
            arg_res.append(self.post_order_travel(item, func))
        return func(chain, arg_res)

    def build_chain_str_func(self, chain, arg_res):
        assert len(chain) != 0
        dep = self.operations[chain[0]]['dependencies']
        if len(dep) == 2:
            assert len(chain) == 1
            return self.process_binary_op(chain[0], arg_res)
        else:
            assert len(arg_res) == 0, f"error {chain} {arg_res}"
            return self.process_chain_op(chain, arg_res)

    def __call__(self):
        if self.ann['types']['semantic'] == 'global':
            self.add_long_answer_cot()
        else:
            chains = build_chains(self.ann['semantic'])
            # print(chains)
            self.post_order_travel(chains, self.build_chain_str_func)
        self.add_summary_and_answer_post_process()
        return " ".join(self.res)

    def add_long_answer_cot(self):
        ann = self.ann
        origin_sent = ann['fullAnswer']
        sent = list(origin_sent.split())
        for span, rids_str in ann['annotations']['fullAnswer'].items():
            span = tuple(map(int, span.split(':')))
            if len(span) == 1:
                span = [span[0], span[0] + 1]
            sent[span[0]] = f"{PHRASE_ST_PLACEHOLDER}{sent[span[0]]}"
            sent[span[1] - 1] = f"{sent[span[1] - 1]}{PHRASE_ED_PLACEHOLDER}"
            # sent[span[0]:span[1]] = [f"{PHRASE_ST_PLACEHOLDER}{' '.join(sent[span[0]:span[1]])}{PHRASE_ED_PLACEHOLDER}"]
            self.add_boxes_by_rids(rids_str.split(','))
        sent_converted = " ".join(sent)
        sent_converted = re.sub('(?:^Yes,)|(?:^No,)', '', sent_converted)  # postpone answer for logic
        sent_converted = sent_converted.strip()
        self.res.append(sent_converted)
        self.res.append(f"So the answer is {ann['answer']}.")

    def add_short_answer_cot(self):
        ann = self.ann
        self.res.append(f"The answer is {ann['answer']}.")

    def add_summary_and_answer_post_process(self):
        ann = self.ann
        if ann['types']['structural'] in ['verify', 'choose']:
            self.add_short_answer_cot()
        else:
            self.add_long_answer_cot()

    def get_boxes_idx(self, box):
        if box in self.boxes_list:
            return self.boxes_list.index(box)
        else:
            self.boxes_list.append(box)
            return len(self.boxes_list) - 1

    def add_boxes_by_rids(self, rids):
        boxes_idx = []
        for rid in rids:
            ref = self.scene['objects'][rid]
            ref_box = get_box_xyxy(ref)
            boxes_idx.append(self.get_boxes_idx(ref_box))
        self.boxes_seq.append(boxes_idx)


if __name__ == '__main__':
    q={'semantic': [{'operation': 'select', 'dependencies': [], 'argument': 'person (1272801)'}, {'operation': 'same', 'dependencies': [0], 'argument': 'gender'}], 'entailed': ['17501270'], 'equivalent': ['17501269'], 'question': 'Are all the people the same gender?', 'imageId': '2387794', 'isBalanced': True, 'groups': {'global': None, 'local': '07same-allpeople'}, 'answer': 'yes', 'semanticStr': 'select: person (1272801)->same: gender [0]', 'annotations': {'answer': {}, 'question': {}, 'fullAnswer': {}}, 'types': {'detailed': 'sameGender', 'semantic': 'attr', 'structural': 'compare'}, 'fullAnswer': 'Yes, all the people are female.'}
    s={'width': 500, 'objects': {'1272804': {'name': 'water', 'h': 70, 'relations': [{'object': '1272802', 'name': 'near'}, {'object': '1272803', 'name': 'to the right of'}, {'object': '1272801', 'name': 'to the right of'}, {'object': '1272801', 'name': 'near'}], 'w': 95, 'attributes': ['blue', 'deep'], 'y': 147, 'x': 358}, '1272801': {'name': 'girls', 'h': 134, 'relations': [{'object': '1272804', 'name': 'near'}, {'object': '3829949', 'name': 'to the right of'}, {'object': '1272804', 'name': 'to the left of'}], 'w': 138, 'attributes': ['little', 'sitting', 'young', 'smiling', 'happy'], 'y': 164, 'x': 198}, '3829949': {'name': 'basket', 'h': 61, 'relations': [{'object': '1272801', 'name': 'to the left of'}], 'w': 108, 'attributes': ['green'], 'y': 241, 'x': 76}, '1272803': {'name': 'umbrella', 'h': 123, 'relations': [{'object': '1272804', 'name': 'to the left of'}], 'w': 165, 'attributes': ['brown'], 'y': 118, 'x': 135}, '1272802': {'name': 'ground', 'h': 44, 'relations': [{'object': '1272804', 'name': 'near'}], 'w': 498, 'attributes': ['brown', 'rocky', 'muddy', 'dirty'], 'y': 216, 'x': 0}}, 'height': 366}
    gqa2cot = GQA2CoT(q, s)
    res = gqa2cot()
    print(f"****: {gqa2cot.res}")
    print(q['question'])
    print(res)
    print(gqa2cot.boxes_list)
    print(gqa2cot.boxes_seq)
    print()
