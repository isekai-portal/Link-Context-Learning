_base_ = ['DEFAULT_TRAIN_DATASET.py']

data_args = dict(
    #
    train=dict(
        type='InterleaveDateset',
        cfgs=[
            {{_base_.DEFAULT_TRAIN_DATASET.vgcap}},
            {{_base_.DEFAULT_TRAIN_DATASET.llavacc3m}},
            {{_base_.DEFAULT_TRAIN_DATASET.llavalcs}},
            {{_base_.DEFAULT_TRAIN_DATASET.rec}},
            {{_base_.DEFAULT_TRAIN_DATASET.recvg}},
            {{_base_.DEFAULT_TRAIN_DATASET.reg}},
            {{_base_.DEFAULT_TRAIN_DATASET.gc}},
            {{_base_.DEFAULT_TRAIN_DATASET.caption}},
        ],
        probabilities=[1./8]*8,
        seed=None,
        stopping_strategy='first_exhausted',
    ),
    validation=None,
    test=None,
    
    # mix training
    use_mix=True,

    # compute_metric
    compute_metric=None,

    # padding collator kwargs
    collator_kwargs=dict(
        padding=True,
        max_length=1024,
    ),

    # generate config
    gen_kwargs=dict(
        max_new_tokens=1024,
        num_beams=1,
    ),
)
