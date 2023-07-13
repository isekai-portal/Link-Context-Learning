import json

obj = json.load(open(r'D:\home\dataset\GQA\questions1.2\testdev_all_questions.json', 'r', encoding='utf8'))

with open('testdev_all_questions.jsonl', 'w', encoding='utf8') as g:
    for v in obj.values():
        g.write(json.dumps(v))
        g.write('\n')
