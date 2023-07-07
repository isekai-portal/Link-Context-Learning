import glob
import json
import re
import sys
import typing
from pathlib import Path
from typing import List, Union


def format_str(s) -> [list, str]:
    middle_brackets_pat = re.compile(r'\[\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3}(?:;\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3})*(?:\]|;)')
    sentence = middle_brackets_pat.sub('', s)

    ret = []
    for bboxes_str in middle_brackets_pat.findall(s.replace(' ', '').replace('<ph_st>', '').replace('<ph_ed>', '')):
        bboxes = []
        bbox_strs = bboxes_str.replace("(", "").replace(")", "").replace("[", "").replace("]", "").strip(';').split(";")
        for bbox_str in bbox_strs:
            bbox = list(map(float, bbox_str.split(',')))
            for elem in bbox:
                if elem < 0 or elem > 1:
                    raise ValueError(f"bbox out of bound: {bbox}, {s}")
            bboxes.append(bbox)
        ret.append(bboxes)

    # sanity check
    if sentence.count('<ph_st>') != sentence.count('<ph_ed>'):
        raise ValueError(f"<st> <ed> not match: {s}")
    if sentence.count('<ph_st>') != len(ret):
        raise ValueError(f"{sentence.count('<ph_st>')}, {len(ret)}: {ret}, {sentence}\n{s}")
    if '[' in sentence or ']' in sentence:
        raise ValueError(f"[] should not in {sentence}")
    return ret, sentence


def merge_boxes(boxes_seqs1, boxes_seqs2):
    all_boxes = []
    idxes_seq1 = extract_box(all_boxes, boxes_seqs1)
    idxes_seq2 = extract_box(all_boxes, boxes_seqs2)
    return all_boxes, idxes_seq1, idxes_seq2


def extract_box(all_boxes, boxes_seqs):
    idxes_seq = []
    for boxes in boxes_seqs:
        idx_seq = []
        for box in boxes:
            if box in all_boxes:
                idx = all_boxes.index(box)
            else:
                idx = len(all_boxes)
                all_boxes.append(box)
            idx_seq.append(idx)
        idxes_seq.append(idx_seq)
    return idxes_seq


def de_norm_box_xyxy(box, *, w, h):
    x1, y1, x2, y2 = box
    x1 = x1 * w
    x2 = x2 * w
    y1 = y1 * h
    y2 = y2 * h
    box = x1, y1, x2, y2
    return box


if __name__ == '__main__':
    mode = 'train'
    # mode = 'test'
    pat = rf'D:\home\code\unify_mllm\data\vqa-rd-gpt4\vqa-gpt4\{mode}\*.json'
    outputs = []
    failed = []
    files = glob.glob(pat)
    print(len(files))
    for file in files:
        obj = json.load(open(file))
        if obj['GPT4VQA']['object'] == 'error':
            continue

        content = obj['GPT4VQA']['choices'][0]['message']['content']
        content = typing.cast(str, content)

        sts = [item.span()[0] for item in re.finditer('\(Human\)\s*Question\s*\d\s*:', content)]
        sts.append(len(content))
        for idx, (st, ed) in enumerate(zip(sts[:-1], sts[1:]), start=1):
            question = content[st:ed]
            try:
                if not bool(question):
                    raise ValueError(f'question is None: {idx} {content}[{st}, {ed}]')

                human, assis = re.split(r'\(Assistant\)\s*Answer\s*\d\s*:', question)
                human = re.sub(r'\(Human\)\s*Question\s*\d\s*:', '', human.strip()).strip()
                assis = assis.strip()
                human_boxes, human_sentence = format_str(human)
                assis_boxes, assis_sentence = format_str(assis)
                boxes, human_boxes_seq, assis_boxes_seq = merge_boxes(human_boxes, assis_boxes)
                boxes = [de_norm_box_xyxy(_, w=obj['width'], h=obj['height']) for _ in boxes]
                answer = re.findall(r"[tT]he answer is [\'\"](yes|no)\.?[\'\"]", assis_sentence)
                assis_sentence = assis_sentence. \
                    replace("'yes'", "yes").replace("'no'", "no"). \
                    replace('"yes"', 'yes').replace('"no"', 'no'). \
                    replace("'yes.'", "yes.").replace('"yes."', 'yes.'). \
                    replace("'no.'", "no.").replace('"no."', 'no.')
                if len(answer) < 1:
                    raise ValueError(f'no answer extracted: {answer}, {assis_sentence}')
                if len(answer) > 1:
                    raise ValueError(f'multi answer extracted: {answer}, {assis_sentence}')
                answer = answer[0]

                output = {
                    'img_id': obj['img_id'],
                    'img_path': Path(obj['img_path']).name,
                    'height': obj['height'],
                    'width': obj['width'],
                    'question': human_sentence,
                    'cot_with_ans': assis_sentence,
                    'answer': answer,
                    'boxes': boxes,
                    'question_boxes_seq': human_boxes_seq,
                    'answer_boxes_seq': assis_boxes_seq,
                }
                outputs.append(output)
            except Exception as e:
                failed.append((file, idx, (st, ed), question, e))

    print(len(failed))
    print(len(outputs))
    with open(f'{mode}_filtered_sample.jsonl', 'w') as f:
        for o in outputs:
            f.write(json.dumps(o) + '\n')