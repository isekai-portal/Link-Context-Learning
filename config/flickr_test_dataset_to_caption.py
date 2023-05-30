_base_ = ['_base_/dataset/flickr.py', '_base_/model/llava_v1_7b.py', '_base_/train/eval.py']

model_args = dict(
    model_name_or_path=r'/mnt/lustre/share_data/chenkeqin/llava_flickr/flickr_5_99999',
)

training_args = dict(
    do_train=False,
    do_eval=False,
    do_predict=True,
    output_dir=r'./{{fileBasenameNoExtension}}',
    per_device_eval_batch_size=16,
    # dataloader_num_workers=4,
)

data_args = dict(
    train=None,
    validation=None,
    test=dict(
        type='FlickrDataset',
        filename=r'/mnt/lustre/share_data/chenkeqin/VG/MM_Instruction/flickr30k/test.jsonl',
        image_folder=r'zz1424:s3://production-public-flickr_image/Flickr_Image/unzip/flickr30k_images/flickr30k_images',
        template_file=r'/mnt/lustre/share_data/chenkeqin/VG/MM_Instruction/flickr30k_norm01_expand2square/capbox_question_template.json',
        max_dynamic_size=1,
    ),

    gen_kwargs=dict(
        max_new_tokens=256,
        num_beams=1,
        do_sample=True,
        temperature=0.7,
    ),
)