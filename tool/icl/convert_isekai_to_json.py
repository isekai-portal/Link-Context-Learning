import os
import jsonlines

img_folder = '/mnt/lustre/share_data/taiyan/dataset/isekai/images'
jsonl = '/mnt/lustre/share_data/taiyan/dataset/isekai/test.jsonl'

support_name = ['s1.png', 's2.png']
support_warp = [support_name]

query_warp = []
imgs = os.listdir(img_folder)
querys = []
for img in imgs:
    if img not in support_name:
        querys.append(img) 
query_warp.append(querys)
folder_warp=['equimanoid']
name_warp = ['horse man']

with jsonlines.open(jsonl, 'w') as writer:
    writer.write(dict(
        Support = support_warp,
        Query = query_warp,
        Folder = folder_warp,
        Name = name_warp
        )
    )


