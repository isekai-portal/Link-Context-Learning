import json

with open("D:\home\dataset\clevr\CLEVR_v1.0\scenes\CLEVR_train_scenes.json", 'r') as f, \
        open('CLEVR_train_scenes.jsonl', 'w') as g:
    obj = json.load(f)

    for o in obj['scenes']:
        g.write(json.dumps(o))
        g.write('\n')
