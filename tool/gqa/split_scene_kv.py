import json

scene = json.load(open(r"D:\home\dataset\GQA\sceneGraphs\train_sceneGraphs.json", 'r'))
indexes = {}
with open('index.json', 'w', encoding='utf8') as f, open('data.jsonl', 'w', encoding='utf8') as g:
    for idx, (k, s) in enumerate(scene.items()):
        indexes[k] = idx
        g.write(json.dumps(s))
        g.write('\n')

    json.dump(indexes, f)
