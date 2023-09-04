_base_ = ['DEFAULT_TRAIN_DATASET.py']

#            {{_base_.DEFAULT_TRAIN_DATASET.imagenet_jigsaw_v1}},
data_args = dict(
    #
    train=dict(
        type='InterleaveDateset',
        cfgs=[
            {{_base_.DEFAULT_TRAIN_DATASET.imagenet_v13}},
            dict(
                type='SubSet',
                portion=1/20,
                do_shuffle=True,
                seed=42,
                cfg={{_base_.DEFAULT_TRAIN_DATASET.gc}},
            ),
            dict(
                type='SubSet',
                portion=1/20,
                do_shuffle=True,
                seed=43,
                cfg={{_base_.DEFAULT_TRAIN_DATASET.recvg}},
            ),
            {{_base_.DEFAULT_TRAIN_DATASET.llavacc3m}},
            {{_base_.DEFAULT_TRAIN_DATASET.llavalcs}},
            {{_base_.DEFAULT_TRAIN_DATASET.VQAv2_train}},
            {{_base_.DEFAULT_TRAIN_DATASET.VQAE_train}},
            {{_base_.DEFAULT_TRAIN_DATASET.VQAX_train}},
            {{_base_.DEFAULT_TRAIN_DATASET.caption}},
            {{_base_.DEFAULT_TRAIN_DATASET.rec}},
            {{_base_.DEFAULT_TRAIN_DATASET.reg}},
            {{_base_.DEFAULT_TRAIN_DATASET.flickr}},
            {{_base_.DEFAULT_TRAIN_DATASET.VCR_q_ra}},
            {{_base_.DEFAULT_TRAIN_DATASET.VCR_qc_rac}},
            {{_base_.DEFAULT_TRAIN_DATASET.VCR_qac_r}},
            {{_base_.DEFAULT_TRAIN_DATASET.POINT_LOCAL_b}},
            {{_base_.DEFAULT_TRAIN_DATASET.POINT_LOCAL_p}},
            {{_base_.DEFAULT_TRAIN_DATASET.POINT_TWICE_oq_bp}},
            {{_base_.DEFAULT_TRAIN_DATASET.POINT_TWICE_sq_bp}},
            {{_base_.DEFAULT_TRAIN_DATASET.POINT_TWICE_gq_bp}},
            {{_base_.DEFAULT_TRAIN_DATASET.POINT_V7W_p}},
            {{_base_.DEFAULT_TRAIN_DATASET.POINT_V7W_b}},
        ],
        probabilities=[0.5] + [1./21.*0.5]*21,
        seed=None,
        stopping_strategy='first_exhausted',
    ),
    validation=None,
    test=None,
    
    # mix training
    use_mix=True,
    icl_dataset_list=[0],

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
