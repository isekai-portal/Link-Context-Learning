def test_plain_formatter():
    from mllm.dataset.common.box_formatter import PlainBoxFormatter

    pbf = PlainBoxFormatter()
    assert pbf.extract('Answer: (0.001,0.125,0.001,0.126).') == [[[0.001, 0.125, 0.001, 0.126]]]
    assert pbf.extract('Answer:(0.001,0.125,0.001,0.126) .') == [[[0.001, 0.125, 0.001, 0.126]]]
    assert pbf.format_box([[0.001, 0.125, 0.001, 0.126]]) == '(0.001,0.125,0.001,0.126)'
