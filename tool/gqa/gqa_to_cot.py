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

    def add_box_to_question(self):
        ann = self.ann
        origin_sent = ann['question']
        sent = list(origin_sent.split())
        for span, rids_str in ann['annotations']['question'].items():
            span = tuple(map(int, span.split(':')))
            if len(span) == 1:
                span = [span[0], span[0] + 1]
            sent[span[0]] = f"{PHRASE_ST_PLACEHOLDER}{sent[span[0]]}"
            sent[span[1] - 1] = f"{sent[span[1] - 1]}{PHRASE_ED_PLACEHOLDER}"
            # sent[span[0]:span[1]] = [f"{PHRASE_ST_PLACEHOLDER}{' '.join(sent[span[0]:span[1]])}{PHRASE_ED_PLACEHOLDER}"]
            self.add_boxes_by_rids(rids_str.split(','))
        sent_converted = " ".join(sent)
        sent_converted = sent_converted.strip()
        self.res.append(sent_converted)

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
    import json

    question = json.load(open(r"D:\home\dataset\GQA\questions1.2\train_balanced_questions.json", 'r'))
    scene = json.load(open(r"D:\home\dataset\GQA\sceneGraphs\train_sceneGraphs.json", 'r'))
    from tqdm import tqdm

    with open('gqa_balanced_with_cot.jsonl', 'w', encoding='utf8') as g:

        for k, q in tqdm(question.items()):
            s = scene[q['imageId']]
            gqa2cot = GQA2CoT(q, s)
            res = gqa2cot()

            out_dict = {}

            out_dict['cot'] = {
                'value': res,
                'boxes': gqa2cot.boxes_list,
                'seq': gqa2cot.boxes_seq,
            }
            out_dict['questionId'] = k
            for k in [
                'question',
                # 'semantic',
                'semanticStr',
                'imageId',
                'annotations',
                'fullAnswer',
                'answer'
            ]:
                out_dict[k] = q[k]
            g.write(json.dumps(out_dict))
            g.write('\n')
