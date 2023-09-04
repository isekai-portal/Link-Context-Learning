VGCAP_TRAIN_COMMON_CFG = dict(
    type='VGCapDataset',
    filename=r'/mnt/lustre/fanweichen2/Research/MLLM/vg_data/processed_vg.json',
    template_file=r"/mnt/cache/fanweichen2/Code/unify_mllm/config/_base_/dataset/template/d_cap.json",
)

DEFAULT_TRAIN_VGCAP_VARIANT = dict(
    vgcap_train=dict(**VGCAP_TRAIN_COMMON_CFG),
)
