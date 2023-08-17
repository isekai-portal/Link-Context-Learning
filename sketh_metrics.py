import os
import jsonlines

def extract_ans(string: str):
    try:
        found = string.split("ASSISTANT:")[-1].split("<PAD>")[0].replace("The answer is", "")
        found = found.split('there is')[-1].replace('in the image', '').replace(".", "").strip().lower()
        return found
    except (IndexError, AttributeError):
        return None

def calculate_metric(preds, targets):
    # pd = data['pred'].split(' "target"')[0].lower()
    # gt = data['target'].split('ASSISTANT:')[-1].split('</s>')[0].lower()
    # gt_label = gt.split(' there is ')[1].split(' ')[0]
    # if gt_label in pd:
    #     correct+=1 

    correct = 0
    failed = 0
    target_failed = 0
    for pred, target in zip(preds, targets):
        # extract_pred = extract_ans(pred)
        extract_pred = pred
        extract_target = extract_ans(target)
        if extract_target is None:
            target_failed += 1
            continue
        if extract_pred is None:
            failed += 1

        if extract_target in extract_pred:
            correct += 1
    return {
        'accuracy': 1.0 * correct / len(targets),
        'target_failed': target_failed,
        'failed': failed,
    }

result_file = "/mnt/lustre/share_data/taiyan/dataset/imagenet1k/test100_otter_3s/multitest_ImageNet1k_100Class_extra_prediction.jsonl"
preds = []
targets = []
with jsonlines.open(result_file) as reader:
    for result in reader:
        preds.append(result['pred'])
        targets.append(result['target'])

metric = calculate_metric(preds, targets)
print(metric)
