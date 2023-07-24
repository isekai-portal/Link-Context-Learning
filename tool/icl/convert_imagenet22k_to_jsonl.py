import random
import jsonlines
from tqdm import tqdm

wordnet_ids_path = '/mnt/lustre/share_data/taiyan/dataset/imagenet22k/annotations/storage.googleapis.com_bit_models_imagenet21k_wordnet_ids.txt'
wordnet_lemmas_path = '/mnt/lustre/share_data/taiyan/dataset/imagenet22k/annotations/storage.googleapis.com_bit_models_imagenet21k_wordnet_lemmas.txt'
train22k_txt = '/mnt/lustre/share_data/taiyan/dataset/imagenet22k/train22k.txt'
output_train_dir = '/mnt/lustre/share_data/taiyan/dataset/imagenet22k/icl/imagenet22k_train.jsonl'
output_test_dir = '/mnt/lustre/share_data/taiyan/dataset/imagenet22k/icl/imagenet22k_test.jsonl' 

total_clsnum = 21842
train_clsnum = 20000
test_clsnum = 1842

def read_txt(path):
    with open(path, 'r') as file:
        lines = file.readlines()
        lines = [line.rstrip('\n') for line in lines]
    return lines

def write_json(objs, path):
    with jsonlines.open(path, 'w') as writer:
        for cls_meta in tqdm(objs):
            writer.write(cls_meta)

def convert_imagenet22k_to_jsonl():
    wordnet_ids = read_txt(wordnet_ids_path)
    wordnet_lemmas = read_txt(wordnet_lemmas_path)

    # wordnet_id to lemmas
    wordnetid2lemma = {}
    for id, image_path in zip(wordnet_ids, wordnet_lemmas):
        wordnetid2lemma[id] = image_path
    
    # class_id to lemmas
    print("Reading train22k.txt, please wait.")
    train22k = read_txt(train22k_txt)
    objs = {}
    clsid2lemma = {}
    print("Organizing images, please wait.")
    for item in tqdm(train22k):
        info = item.split(' ')
        image_path = info[0]
        clsid = info[1]
        
        wordnet_id = image_path.split('/')[0]
        # image_name = image_path.split('/')[1]
        if wordnet_id not in wordnetid2lemma.keys():
            continue
        lemma = wordnetid2lemma[wordnet_id]
        lemma = [t.strip().replace("_"," ") for t in lemma.split(',')]

        if clsid not in clsid2lemma.keys():
            clsid2lemma[clsid] = lemma
            objs[clsid] = [clsid, lemma,[image_path]]
        else:
            objs[clsid][2].append(image_path)

    # split lemmas
    train_objs = []
    test_objs = []

    train_ids = random.sample(objs.keys(), train_clsnum)
    for clsid, meta in objs.items():
        if clsid in train_ids:
            train_objs.append(meta)
        else:
            test_objs.append(meta)

    write_json(train_objs, output_train_dir)
    write_json(test_objs, output_test_dir)
        
    
if __name__ == '__main__':
    convert_imagenet22k_to_jsonl()

    from tqdm import tqdm