_base_ = ['llava_eval_rec.py']

training_args = dict(
    output_dir='/mnt/lustre/share_data/chenkeqin/dummy_eval_exp_unify_mllm/{{fileBasenameNoExtension}}',
    do_eval=False,
)

model_args = dict(
    model_name_or_path=r'/mnt/lustre/share_data/chenkeqin/llava_rec/rec_train_all_1_1',
)

data_args = dict(
    validation = None,
    test = dict(
        type='RECDataset',
        filename=r'/mnt/lustre/share_data/chenkeqin/mllm_data/pretrain_data/ann/merged_REC_refcoco_test.jsonl',

        # template_file=None,
        # max_dynamic_size=1,
        # template_string="Please specify the location of <expr> using the bounding box's top-left and bottom-right coordinates. Make sure the coordinates are normalized between 0 and 1.<image>",
    ),
)
