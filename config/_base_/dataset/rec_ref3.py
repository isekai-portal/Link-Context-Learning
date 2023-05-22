_base_ = ['rec.py']

data_args = dict(

    train=dict(
        type='RECDataset',
        filename=r'/mnt/lustre/share_data/chenkeqin/VG/pretrain_data/REC/REC_ref3/train.jsonl',
    ),
    validation=dict(
        type='RECDataset',
        filename=r'/mnt/lustre/share_data/chenkeqin/VG/pretrain_data/REC/REC_ref3/val.jsonl',
    ),
    test=dict(
        type='RECDataset',
        filename=r'/mnt/lustre/share_data/chenkeqin/VG/pretrain_data/REC/REC_ref3/test.jsonl',
    ),
)
