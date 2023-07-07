import json


def get_idx(boxes_list, box):
    if box in boxes_list:
        return boxes_list.index(box)
    else:
        boxes_list.append(box)
        return len(boxes_list) - 1


def get_boxes_idx(boxes_list, refs):
    idx = [get_idx(boxes_list, box) for box in refs]
    return idx


def clevr_ss_cot(obj, scene, add_ref=False):
    cot = []
    boxes = []
    seq = []

    for p in obj['program']:
        func = f"{p['function']}:{p['value_inputs'][0]}" if 'value_inputs' in p and p['value_inputs'] else p['function']
        inputs = f"[{','.join(map(str, p['inputs']))}]" if p['inputs'] else ""
        if add_ref and p['function'] in ['unique', 'intersect', 'relate', 'same_size', 'same_shape', 'same_material', 'same_color']:
            if p['ans']:
                objs = f"<boxes>"
                idx = get_boxes_idx(boxes_list=boxes, refs=[scene['objects'][_]['pixel_coords'][:2] for _ in p['ans']])
                seq.append(idx)
            else:
                objs = f"(None)"
        else:
            objs = ""
        cot.append(f"{func}{inputs}{objs}")

    ret = " -> ".join(cot)
    return ret, boxes, seq


if __name__ == '__main__':
    scene = json.load(open(r"D:\home\dataset\clevr\CLEVR_v1.0\scenes\CLEVR_val_scenes.json", 'r'))
    for line in open('CLEVR_val_questions_with_ans.jsonl', 'r'):
        obj = json.loads(line)
        ret = clevr_ss_cot(obj, scene['scenes'][obj['image_index']], add_ref=True)
        print(ret)
        _ = input()

# print(sorted(all_func))
#
# [
#     # final
#     'greater_than', 'less_than',  # num,num->bool
#     'query_color', 'query_material', 'query_shape', 'query_size',  # item->attr
#     'equal_color', 'equal_integer', 'equal_material', 'equal_shape', 'equal_size',  # item,item->bool#
#     'exist',  # list->bool
#     'count',  # list->num
#
#     # ignore
#     'scene',  # []->list
#     'filter_color[value_input]', 'filter_material[value_input]', 'filter_shape[value_input]', 'filter_size[value_input]',  # list->list
#
#     # use
#     'union', 'intersect',  # list,list->list
#     'relate[value_input]',  # item->list
#     'same_color', 'same_material', 'same_shape', 'same_size',  # item->list
#     'unique',  # list->item
# ]
