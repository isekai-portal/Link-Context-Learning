_base_ = ['rec_ref4_reverse.py']

data_args = dict(

    train=dict(
        filename=r'/mnt/lustre/share_data/chenkeqin/VG/pretrain_data/REC/REC_ref3/train.jsonl',
    ),
    validation=dict(
        filename=r'/mnt/lustre/share_data/chenkeqin/VG/pretrain_data/REC/REC_ref3/val.jsonl',
    ),
    test=dict(
        filename=r'/mnt/lustre/share_data/chenkeqin/VG/pretrain_data/REC/REC_ref3/test.jsonl',
    ),
)
