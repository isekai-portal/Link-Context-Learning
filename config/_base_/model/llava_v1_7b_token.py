_base_ = ['llava_v1_7b.py']

model_args = dict(
    target_processor=dict(
        boxes=dict(type='TokenFormatter', num_bins=1001),
    ),
)