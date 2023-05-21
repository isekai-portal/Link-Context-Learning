from matplotlib import pyplot as plt


def test_rec_dataset():
    from transformers import CLIPImageProcessor
    from mllm.dataset.conv import post_process
    from mllm.dataset.rec.rec import RECDataset
    from mllm.models.flamingo import FlamingoTokenizer

    preprocessor = dict(
        image=CLIPImageProcessor(),
        text=FlamingoTokenizer.from_pretrained(r"D:\home\code\mllm\test\llama_7b_hf"),
        multimodal_cfg=dict(
            image_token_len=256,
            is_multimodal=True,
            sep_image_conv_front=False,
            use_im_start_end=True,
        )
    )

    tokenize_kwargs = {
    }

    trainds = RECDataset(
        data_file=r'D:\home\code\mllm\test\rec_sample.jsonl',
        preprocessor=preprocessor,
        tokenize_kwargs=tokenize_kwargs,
        template_string='Can you tell me the whereabouts of <expr>? I need the spatial coordinates in the format (x1, y1, x2, y2).',
        train_mode=True,
    )
    testds = RECDataset(
        data_file=r'D:\home\code\mllm\test\rec_sample.jsonl',
        preprocessor=preprocessor,
        tokenize_kwargs=tokenize_kwargs,
        template_string='Can you tell me the whereabouts of <expr>? I need the spatial coordinates in the format (x1, y1, x2, y2).',
        train_mode=False,
    )
    trainsample = trainds[0]
    labels = post_process(preprocessor['text'], trainsample['labels'])
    assert preprocessor['text'].convert_ids_to_tokens(trainsample['input_ids']) == ['<s>', '▁A', '▁chat', '▁between', '▁a', '▁curious',
                                                                                    '▁user', '▁and', '▁an', '▁artificial', '▁intelligence',
                                                                                    '▁assistant', '.', '▁The', '▁assistant', '▁gives',
                                                                                    '▁helpful', ',', '▁detailed', ',', '▁and', '▁pol',
                                                                                    'ite', '▁answers', '▁to', '▁the', '▁user', "'", 's',
                                                                                    '▁questions', '.', '▁US', 'ER', ':', '▁Can', '▁you',
                                                                                    '▁tell', '▁me', '▁the', '▁where', 'about', 's', '▁of',
                                                                                    '▁a', '▁half', '▁dr', 'unk', '▁be', 'er', '▁in', '▁a',
                                                                                    '▁large', '▁glass', '?', '▁I', '▁need', '▁the',
                                                                                    '▁spatial', '▁coordinates', '▁in', '▁the', '▁format',
                                                                                    '▁(', 'x', '1', ',', '▁y', '1', ',', '▁x', '2', ',',
                                                                                    '▁y', '2', ').', '▁A', 'SS', 'IST', 'ANT', ':',
                                                                                    '▁Answer', ':', '▁(', '0', '.', '0', '0', '1', ',', '0',
                                                                                    '.', '1', '6', '7', ',', '0', '.', '0', '0', '1', ',',
                                                                                    '0', '.', '1', '6', '8', ')', '▁.', '</s>']
    assert preprocessor['text'].convert_ids_to_tokens(labels) == ['<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>',
                                                                  '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>',
                                                                  '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>',
                                                                  '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>',
                                                                  '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>',
                                                                  '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>',
                                                                  '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>',
                                                                  '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>',
                                                                  '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>',
                                                                  '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>',
                                                                  '▁Answer', ':', '▁(', '0', '.', '0', '0', '1', ',', '0', '.', '1', '6',
                                                                  '7', ',', '0', '.', '0', '0', '1', ',', '0', '.', '1', '6', '8', ')',
                                                                  '▁.', '</s>']
    testsample = testds[0]
    labels = post_process(preprocessor['text'], testsample['labels'])
    assert preprocessor['text'].convert_ids_to_tokens(testsample['input_ids']) == ['<s>', '▁A', '▁chat', '▁between', '▁a', '▁curious',
                                                                                   '▁user', '▁and', '▁an', '▁artificial', '▁intelligence',
                                                                                   '▁assistant', '.', '▁The', '▁assistant', '▁gives',
                                                                                   '▁helpful', ',', '▁detailed', ',', '▁and', '▁pol', 'ite',
                                                                                   '▁answers', '▁to', '▁the', '▁user', "'", 's',
                                                                                   '▁questions', '.', '▁US', 'ER', ':', '▁Can', '▁you',
                                                                                   '▁tell', '▁me', '▁the', '▁where', 'about', 's', '▁of',
                                                                                   '▁a', '▁half', '▁dr', 'unk', '▁be', 'er', '▁in', '▁a',
                                                                                   '▁large', '▁glass', '?', '▁I', '▁need', '▁the',
                                                                                   '▁spatial', '▁coordinates', '▁in', '▁the', '▁format',
                                                                                   '▁(', 'x', '1', ',', '▁y', '1', ',', '▁x', '2', ',',
                                                                                   '▁y', '2', ').', '▁A', 'SS', 'IST', 'ANT', ':']
    assert preprocessor['text'].convert_ids_to_tokens(labels) == ['▁Answer', ':', '▁(', '0', '.', '0', '0', '1', ',', '0', '.', '1', '6',
                                                                  '7', ',', '0', '.', '0', '0', '1', ',', '0', '.', '1', '6', '8', ')',
                                                                  '▁.']
    trainsample = trainds[1]
    testsample = testds[1]
    trainsample = trainds[2]
    testsample = testds[2]


def test_box():
    from mllm.dataset.common.transform import de_norm_box_xyxy
    from mllm.dataset.rec import RECRawDataset
    from mllm.dataset.common import read_img_general
    from mllm.utils import show, draw_bounding_boxes

    data_file = r'D:\home\code\unify_mllm\test\rec_sample.jsonl'
    template_string = r"Where does <expr> appear in the image? Please provide the specific coordinates."

    rec = RECRawDataset(
        data_file=data_file,
        template_string=template_string,
    )

    for i in range(5):
        raw_item = rec.get_raw_item(i)
        img_path = raw_item['img_path']
        img = read_img_general(img_path)
        box = raw_item['bbox']
        box = de_norm_box_xyxy(box, w=img.width, h=img.height)
        result = draw_bounding_boxes(img, [box], colors=['red'])
        show(result)
        plt.show()

        raw_conv = rec.get_raw_conv_with_box(i)
        img2 = raw_conv[0]['image']
        box2 = raw_conv[-1]['bboxes_seq'][0][0]
        box2 = de_norm_box_xyxy(box2, w=img2.width, h=img2.height)
        result = draw_bounding_boxes(img2, [box2], colors=['red'])
        show(result)
        plt.show()
