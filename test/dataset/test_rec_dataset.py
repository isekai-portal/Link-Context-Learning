import pytest


def test_plain_formatter(plain_formatter):
    assert plain_formatter.extract('Answer: (0.001,0.125,0.001,0.126).') == [[[0.001, 0.125, 0.001, 0.126]]]
    assert plain_formatter.extract('Answer:(0.001,0.125,0.001,0.126) .') == [[[0.001, 0.125, 0.001, 0.126]]]
    assert plain_formatter.format_box([[0.001, 0.125, 0.001, 0.126]]) == '(0.001,0.125,0.001,0.126)'


def test_rec_dataset(rec_dataset):
    from mllm.utils import draw_bounding_boxes, show
    import matplotlib.pyplot as plt
    for item in rec_dataset:
        print(item)
        res = draw_bounding_boxes(image=item['image'], boxes=item['target']['boxes'], colors='red', width=4)
        show(res)
        plt.title(item['conversations'][0]['value'])
        plt.show()
        plt.close()


@pytest.mark.parametrize('preprocessor,process_func', [
    (pytest.lazy_fixture('llava_processor'), pytest.lazy_fixture('llava_process_func')),
])
def test_dataset(preprocessor, process_func, dataset):
    from mllm.dataset import TRANSFORMS, SingleImageConvDataset
    transforms = TRANSFORMS.build(cfg=dict(type='Expand2square', ))

    train_ds = SingleImageConvDataset(
        dataset=dataset,
        preprocessor=preprocessor,
        process_func=process_func,
        transforms=transforms,
        mode='train',
    )
    for item in train_ds:
        pass

    test_ds = SingleImageConvDataset(
        dataset=dataset,
        preprocessor=preprocessor,
        process_func=process_func,
        transforms=transforms,
        mode='validation',
    )
    for item in test_ds:
        pass
