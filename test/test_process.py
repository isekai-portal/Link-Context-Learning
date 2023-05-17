def test_text_process():
    from mllm.models.flamingo import FlamingoTokenizer
    from mllm.conversation import get_conv_template
    from mllm.dataset.conv import Name2TextProcess
    from mllm.dataset.conv import post_process
    tk = FlamingoTokenizer.from_pretrained(r"D:\home\code\mllm\test\llama_7b_hf")
    sources = [[
        {
            "from": "human",
            'value': "What does the verbal irony in this text suggest?\nAccording to Mr. Herrera's kids, his snoring is as quiet as a jackhammer.\nContext: N/A\nOptions: (A) The snoring is loud. (B) The snoring occurs in bursts.",
        },
        {
            "from": "gpt",
            "value": "LECTURE: Figures of speech are words or phrases that use language in a nonliteral or unusual way. They can make writing more expressive.\nVerbal irony involves saying one thing but implying something very different. People often use verbal irony when they are being sarcastic.\nOlivia seems thrilled that her car keeps breaking down.\nEach breakdown is as enjoyable as a punch to the face.\nSOLUTION: The text uses verbal irony, which involves saying one thing but implying something very different.\nAs quiet as a jackhammer suggests that the snoring is loud. A jackhammer is not quiet, and neither is Mr. Herrera's snoring.\n###\nANSWER: A.",
        },
        {
            "from": "human",
            'value': "What does the verbal irony in this text suggest?\nAccording to Mr. Herrera's kids, his snoring is as quiet as a jackhammer.\nContext: N/A\nOptions: (A) The snoring is loud. (B) The snoring occurs in bursts.",
        },
        {
            "from": "gpt",
            "value": "LECTURE: Figures of speech are words or phrases that use language in a nonliteral or unusual way. They can make writing more expressive.\nVerbal irony involves saying one thing but implying something very different. People often use verbal irony when they are being sarcastic.\nOlivia seems thrilled that her car keeps breaking down.\nEach breakdown is as enjoyable as a punch to the face.\nSOLUTION: The text uses verbal irony, which involves saying one thing but implying something very different.\nAs quiet as a jackhammer suggests that the snoring is loud. A jackhammer is not quiet, and neither is Mr. Herrera's snoring.\n###\nANSWER: A.",
        },
    ], ]

    conv = get_conv_template('vicuna_v1.1')
    tpc = Name2TextProcess['llava_v1']()
    processor = {"text": tk}

    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])

    # print(conv.get_prompt())
    tk_kwargs = dict(
        return_tensors="pt",
        padding="longest",
        max_length=tk.model_max_length,
        truncation=True,
    )

    res = tpc(conv, processor, train_mode=True, **tk_kwargs)
    labels = post_process(tk, res['labels'])
    # print(tk.convert_ids_to_tokens(res['input_ids']))
    # print(tk.convert_ids_to_tokens(labels))
    assert tk.convert_ids_to_tokens(res['input_ids']) == ['<s>', '▁A', '▁chat', '▁between', '▁a', '▁curious', '▁user', '▁and', '▁an',
                                                          '▁artificial', '▁intelligence', '▁assistant', '.', '▁The', '▁assistant', '▁gives',
                                                          '▁helpful', ',', '▁detailed', ',', '▁and', '▁pol', 'ite', '▁answers', '▁to',
                                                          '▁the', '▁user', "'", 's', '▁questions', '.', '▁US', 'ER', ':', '▁What', '▁does',
                                                          '▁the', '▁ver', 'bal', '▁ir', 'ony', '▁in', '▁this', '▁text', '▁suggest', '?',
                                                          '<0x0A>', 'Acc', 'ording', '▁to', '▁Mr', '.', '▁Herr', 'era', "'", 's', '▁k',
                                                          'ids', ',', '▁his', '▁sn', 'oring', '▁is', '▁as', '▁quiet', '▁as', '▁a', '▁jack',
                                                          'ham', 'mer', '.', '<0x0A>', 'Context', ':', '▁N', '/', 'A', '<0x0A>', 'Options',
                                                          ':', '▁(', 'A', ')', '▁The', '▁sn', 'oring', '▁is', '▁loud', '.', '▁(', 'B', ')',
                                                          '▁The', '▁sn', 'oring', '▁occurs', '▁in', '▁burst', 's', '.', '▁A', 'SS', 'IST',
                                                          'ANT', ':', '▁LE', 'CT', 'URE', ':', '▁Fig', 'ures', '▁of', '▁speech', '▁are',
                                                          '▁words', '▁or', '▁phr', 'ases', '▁that', '▁use', '▁language', '▁in', '▁a',
                                                          '▁non', 'liter', 'al', '▁or', '▁unusual', '▁way', '.', '▁They', '▁can', '▁make',
                                                          '▁writing', '▁more', '▁express', 'ive', '.', '<0x0A>', 'Ver', 'bal', '▁ir', 'ony',
                                                          '▁involves', '▁saying', '▁one', '▁thing', '▁but', '▁imp', 'lying', '▁something',
                                                          '▁very', '▁different', '.', '▁People', '▁often', '▁use', '▁ver', 'bal', '▁ir',
                                                          'ony', '▁when', '▁they', '▁are', '▁being', '▁sar', 'cast', 'ic', '.', '<0x0A>',
                                                          'O', 'liv', 'ia', '▁seems', '▁thr', 'illed', '▁that', '▁her', '▁car', '▁keeps',
                                                          '▁breaking', '▁down', '.', '<0x0A>', 'Each', '▁break', 'down', '▁is', '▁as',
                                                          '▁enjoy', 'able', '▁as', '▁a', '▁p', 'unch', '▁to', '▁the', '▁face', '.',
                                                          '<0x0A>', 'S', 'OL', 'UT', 'ION', ':', '▁The', '▁text', '▁uses', '▁ver', 'bal',
                                                          '▁ir', 'ony', ',', '▁which', '▁involves', '▁saying', '▁one', '▁thing', '▁but',
                                                          '▁imp', 'lying', '▁something', '▁very', '▁different', '.', '<0x0A>', 'As',
                                                          '▁quiet', '▁as', '▁a', '▁jack', 'ham', 'mer', '▁suggests', '▁that', '▁the', '▁sn',
                                                          'oring', '▁is', '▁loud', '.', '▁A', '▁jack', 'ham', 'mer', '▁is', '▁not',
                                                          '▁quiet', ',', '▁and', '▁neither', '▁is', '▁Mr', '.', '▁Herr', 'era', "'", 's',
                                                          '▁sn', 'oring', '.', '<0x0A>', '##', '#', '<0x0A>', 'AN', 'SW', 'ER', ':', '▁A',
                                                          '.', '</s>', '▁US', 'ER', ':', '▁What', '▁does', '▁the', '▁ver', 'bal', '▁ir',
                                                          'ony', '▁in', '▁this', '▁text', '▁suggest', '?', '<0x0A>', 'Acc', 'ording', '▁to',
                                                          '▁Mr', '.', '▁Herr', 'era', "'", 's', '▁k', 'ids', ',', '▁his', '▁sn', 'oring',
                                                          '▁is', '▁as', '▁quiet', '▁as', '▁a', '▁jack', 'ham', 'mer', '.', '<0x0A>',
                                                          'Context', ':', '▁N', '/', 'A', '<0x0A>', 'Options', ':', '▁(', 'A', ')', '▁The',
                                                          '▁sn', 'oring', '▁is', '▁loud', '.', '▁(', 'B', ')', '▁The', '▁sn', 'oring',
                                                          '▁occurs', '▁in', '▁burst', 's', '.', '▁A', 'SS', 'IST', 'ANT', ':', '▁LE', 'CT',
                                                          'URE', ':', '▁Fig', 'ures', '▁of', '▁speech', '▁are', '▁words', '▁or', '▁phr',
                                                          'ases', '▁that', '▁use', '▁language', '▁in', '▁a', '▁non', 'liter', 'al', '▁or',
                                                          '▁unusual', '▁way', '.', '▁They', '▁can', '▁make', '▁writing', '▁more',
                                                          '▁express', 'ive', '.', '<0x0A>', 'Ver', 'bal', '▁ir', 'ony', '▁involves',
                                                          '▁saying', '▁one', '▁thing', '▁but', '▁imp', 'lying', '▁something', '▁very',
                                                          '▁different', '.', '▁People', '▁often', '▁use', '▁ver', 'bal', '▁ir', 'ony',
                                                          '▁when', '▁they', '▁are', '▁being', '▁sar', 'cast', 'ic', '.', '<0x0A>', 'O',
                                                          'liv', 'ia', '▁seems', '▁thr', 'illed', '▁that', '▁her', '▁car', '▁keeps',
                                                          '▁breaking', '▁down', '.', '<0x0A>', 'Each', '▁break', 'down', '▁is', '▁as',
                                                          '▁enjoy', 'able', '▁as', '▁a', '▁p', 'unch', '▁to', '▁the', '▁face', '.',
                                                          '<0x0A>', 'S', 'OL', 'UT', 'ION', ':', '▁The', '▁text', '▁uses', '▁ver', 'bal',
                                                          '▁ir', 'ony', ',', '▁which', '▁involves', '▁saying', '▁one', '▁thing', '▁but',
                                                          '▁imp', 'lying', '▁something', '▁very', '▁different', '.', '<0x0A>', 'As',
                                                          '▁quiet', '▁as', '▁a', '▁jack', 'ham', 'mer', '▁suggests', '▁that', '▁the', '▁sn',
                                                          'oring', '▁is', '▁loud', '.', '▁A', '▁jack', 'ham', 'mer', '▁is', '▁not',
                                                          '▁quiet', ',', '▁and', '▁neither', '▁is', '▁Mr', '.', '▁Herr', 'era', "'", 's',
                                                          '▁sn', 'oring', '.', '<0x0A>', '##', '#', '<0x0A>', 'AN', 'SW', 'ER', ':', '▁A',
                                                          '.', '</s>']
    assert tk.convert_ids_to_tokens(labels) == ['<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>',
                                                '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>',
                                                '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>',
                                                '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>',
                                                '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>',
                                                '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>',
                                                '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>',
                                                '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>',
                                                '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>',
                                                '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>',
                                                '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '▁LE', 'CT', 'URE', ':', '▁Fig', 'ures', '▁of',
                                                '▁speech', '▁are', '▁words', '▁or', '▁phr', 'ases', '▁that', '▁use', '▁language', '▁in',
                                                '▁a', '▁non', 'liter', 'al', '▁or', '▁unusual', '▁way', '.', '▁They', '▁can', '▁make',
                                                '▁writing', '▁more', '▁express', 'ive', '.', '<0x0A>', 'Ver', 'bal', '▁ir', 'ony',
                                                '▁involves', '▁saying', '▁one', '▁thing', '▁but', '▁imp', 'lying', '▁something', '▁very',
                                                '▁different', '.', '▁People', '▁often', '▁use', '▁ver', 'bal', '▁ir', 'ony', '▁when',
                                                '▁they', '▁are', '▁being', '▁sar', 'cast', 'ic', '.', '<0x0A>', 'O', 'liv', 'ia', '▁seems',
                                                '▁thr', 'illed', '▁that', '▁her', '▁car', '▁keeps', '▁breaking', '▁down', '.', '<0x0A>',
                                                'Each', '▁break', 'down', '▁is', '▁as', '▁enjoy', 'able', '▁as', '▁a', '▁p', 'unch', '▁to',
                                                '▁the', '▁face', '.', '<0x0A>', 'S', 'OL', 'UT', 'ION', ':', '▁The', '▁text', '▁uses',
                                                '▁ver', 'bal', '▁ir', 'ony', ',', '▁which', '▁involves', '▁saying', '▁one', '▁thing',
                                                '▁but', '▁imp', 'lying', '▁something', '▁very', '▁different', '.', '<0x0A>', 'As', '▁quiet',
                                                '▁as', '▁a', '▁jack', 'ham', 'mer', '▁suggests', '▁that', '▁the', '▁sn', 'oring', '▁is',
                                                '▁loud', '.', '▁A', '▁jack', 'ham', 'mer', '▁is', '▁not', '▁quiet', ',', '▁and', '▁neither',
                                                '▁is', '▁Mr', '.', '▁Herr', 'era', "'", 's', '▁sn', 'oring', '.', '<0x0A>', '##', '#',
                                                '<0x0A>', 'AN', 'SW', 'ER', ':', '▁A', '.', '</s>', '<PAD>', '<PAD>', '<PAD>', '<PAD>',
                                                '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>',
                                                '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>',
                                                '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>',
                                                '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>',
                                                '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>',
                                                '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>',
                                                '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>',
                                                '▁LE', 'CT', 'URE', ':', '▁Fig', 'ures', '▁of', '▁speech', '▁are', '▁words', '▁or', '▁phr',
                                                'ases', '▁that', '▁use', '▁language', '▁in', '▁a', '▁non', 'liter', 'al', '▁or', '▁unusual',
                                                '▁way', '.', '▁They', '▁can', '▁make', '▁writing', '▁more', '▁express', 'ive', '.',
                                                '<0x0A>', 'Ver', 'bal', '▁ir', 'ony', '▁involves', '▁saying', '▁one', '▁thing', '▁but',
                                                '▁imp', 'lying', '▁something', '▁very', '▁different', '.', '▁People', '▁often', '▁use',
                                                '▁ver', 'bal', '▁ir', 'ony', '▁when', '▁they', '▁are', '▁being', '▁sar', 'cast', 'ic', '.',
                                                '<0x0A>', 'O', 'liv', 'ia', '▁seems', '▁thr', 'illed', '▁that', '▁her', '▁car', '▁keeps',
                                                '▁breaking', '▁down', '.', '<0x0A>', 'Each', '▁break', 'down', '▁is', '▁as', '▁enjoy',
                                                'able', '▁as', '▁a', '▁p', 'unch', '▁to', '▁the', '▁face', '.', '<0x0A>', 'S', 'OL', 'UT',
                                                'ION', ':', '▁The', '▁text', '▁uses', '▁ver', 'bal', '▁ir', 'ony', ',', '▁which',
                                                '▁involves', '▁saying', '▁one', '▁thing', '▁but', '▁imp', 'lying', '▁something', '▁very',
                                                '▁different', '.', '<0x0A>', 'As', '▁quiet', '▁as', '▁a', '▁jack', 'ham', 'mer',
                                                '▁suggests', '▁that', '▁the', '▁sn', 'oring', '▁is', '▁loud', '.', '▁A', '▁jack', 'ham',
                                                'mer', '▁is', '▁not', '▁quiet', ',', '▁and', '▁neither', '▁is', '▁Mr', '.', '▁Herr', 'era',
                                                "'", 's', '▁sn', 'oring', '.', '<0x0A>', '##', '#', '<0x0A>', 'AN', 'SW', 'ER', ':', '▁A',
                                                '.', '</s>']

    res = tpc(conv, processor, train_mode=False, **tk_kwargs)
    labels = post_process(tk, res['labels'])
    # print(tk.convert_ids_to_tokens(res['input_ids']))
    # print(tk.convert_ids_to_tokens(labels))
    assert tk.convert_ids_to_tokens(res['input_ids']) == ['<s>', '▁A', '▁chat', '▁between', '▁a', '▁curious', '▁user', '▁and', '▁an',
                                                          '▁artificial', '▁intelligence', '▁assistant', '.', '▁The', '▁assistant', '▁gives',
                                                          '▁helpful', ',', '▁detailed', ',', '▁and', '▁pol', 'ite', '▁answers', '▁to',
                                                          '▁the', '▁user', "'", 's', '▁questions', '.', '▁US', 'ER', ':', '▁What', '▁does',
                                                          '▁the', '▁ver', 'bal', '▁ir', 'ony', '▁in', '▁this', '▁text', '▁suggest', '?',
                                                          '<0x0A>', 'Acc', 'ording', '▁to', '▁Mr', '.', '▁Herr', 'era', "'", 's', '▁k',
                                                          'ids', ',', '▁his', '▁sn', 'oring', '▁is', '▁as', '▁quiet', '▁as', '▁a', '▁jack',
                                                          'ham', 'mer', '.', '<0x0A>', 'Context', ':', '▁N', '/', 'A', '<0x0A>', 'Options',
                                                          ':', '▁(', 'A', ')', '▁The', '▁sn', 'oring', '▁is', '▁loud', '.', '▁(', 'B', ')',
                                                          '▁The', '▁sn', 'oring', '▁occurs', '▁in', '▁burst', 's', '.', '▁A', 'SS', 'IST',
                                                          'ANT', ':', '▁LE', 'CT', 'URE', ':', '▁Fig', 'ures', '▁of', '▁speech', '▁are',
                                                          '▁words', '▁or', '▁phr', 'ases', '▁that', '▁use', '▁language', '▁in', '▁a',
                                                          '▁non', 'liter', 'al', '▁or', '▁unusual', '▁way', '.', '▁They', '▁can', '▁make',
                                                          '▁writing', '▁more', '▁express', 'ive', '.', '<0x0A>', 'Ver', 'bal', '▁ir', 'ony',
                                                          '▁involves', '▁saying', '▁one', '▁thing', '▁but', '▁imp', 'lying', '▁something',
                                                          '▁very', '▁different', '.', '▁People', '▁often', '▁use', '▁ver', 'bal', '▁ir',
                                                          'ony', '▁when', '▁they', '▁are', '▁being', '▁sar', 'cast', 'ic', '.', '<0x0A>',
                                                          'O', 'liv', 'ia', '▁seems', '▁thr', 'illed', '▁that', '▁her', '▁car', '▁keeps',
                                                          '▁breaking', '▁down', '.', '<0x0A>', 'Each', '▁break', 'down', '▁is', '▁as',
                                                          '▁enjoy', 'able', '▁as', '▁a', '▁p', 'unch', '▁to', '▁the', '▁face', '.',
                                                          '<0x0A>', 'S', 'OL', 'UT', 'ION', ':', '▁The', '▁text', '▁uses', '▁ver', 'bal',
                                                          '▁ir', 'ony', ',', '▁which', '▁involves', '▁saying', '▁one', '▁thing', '▁but',
                                                          '▁imp', 'lying', '▁something', '▁very', '▁different', '.', '<0x0A>', 'As',
                                                          '▁quiet', '▁as', '▁a', '▁jack', 'ham', 'mer', '▁suggests', '▁that', '▁the', '▁sn',
                                                          'oring', '▁is', '▁loud', '.', '▁A', '▁jack', 'ham', 'mer', '▁is', '▁not',
                                                          '▁quiet', ',', '▁and', '▁neither', '▁is', '▁Mr', '.', '▁Herr', 'era', "'", 's',
                                                          '▁sn', 'oring', '.', '<0x0A>', '##', '#', '<0x0A>', 'AN', 'SW', 'ER', ':', '▁A',
                                                          '.', '</s>', '▁US', 'ER', ':', '▁What', '▁does', '▁the', '▁ver', 'bal', '▁ir',
                                                          'ony', '▁in', '▁this', '▁text', '▁suggest', '?', '<0x0A>', 'Acc', 'ording', '▁to',
                                                          '▁Mr', '.', '▁Herr', 'era', "'", 's', '▁k', 'ids', ',', '▁his', '▁sn', 'oring',
                                                          '▁is', '▁as', '▁quiet', '▁as', '▁a', '▁jack', 'ham', 'mer', '.', '<0x0A>',
                                                          'Context', ':', '▁N', '/', 'A', '<0x0A>', 'Options', ':', '▁(', 'A', ')', '▁The',
                                                          '▁sn', 'oring', '▁is', '▁loud', '.', '▁(', 'B', ')', '▁The', '▁sn', 'oring',
                                                          '▁occurs', '▁in', '▁burst', 's', '.', '▁A', 'SS', 'IST', 'ANT', ':']
    assert tk.convert_ids_to_tokens(labels) == ['▁LE', 'CT', 'URE', ':', '▁Fig', 'ures', '▁of', '▁speech', '▁are', '▁words', '▁or', '▁phr',
                                                'ases', '▁that', '▁use', '▁language', '▁in', '▁a', '▁non', 'liter', 'al', '▁or', '▁unusual',
                                                '▁way', '.', '▁They', '▁can', '▁make', '▁writing', '▁more', '▁express', 'ive', '.',
                                                '<0x0A>', 'Ver', 'bal', '▁ir', 'ony', '▁involves', '▁saying', '▁one', '▁thing', '▁but',
                                                '▁imp', 'lying', '▁something', '▁very', '▁different', '.', '▁People', '▁often', '▁use',
                                                '▁ver', 'bal', '▁ir', 'ony', '▁when', '▁they', '▁are', '▁being', '▁sar', 'cast', 'ic', '.',
                                                '<0x0A>', 'O', 'liv', 'ia', '▁seems', '▁thr', 'illed', '▁that', '▁her', '▁car', '▁keeps',
                                                '▁breaking', '▁down', '.', '<0x0A>', 'Each', '▁break', 'down', '▁is', '▁as', '▁enjoy',
                                                'able', '▁as', '▁a', '▁p', 'unch', '▁to', '▁the', '▁face', '.', '<0x0A>', 'S', 'OL', 'UT',
                                                'ION', ':', '▁The', '▁text', '▁uses', '▁ver', 'bal', '▁ir', 'ony', ',', '▁which',
                                                '▁involves', '▁saying', '▁one', '▁thing', '▁but', '▁imp', 'lying', '▁something', '▁very',
                                                '▁different', '.', '<0x0A>', 'As', '▁quiet', '▁as', '▁a', '▁jack', 'ham', 'mer',
                                                '▁suggests', '▁that', '▁the', '▁sn', 'oring', '▁is', '▁loud', '.', '▁A', '▁jack', 'ham',
                                                'mer', '▁is', '▁not', '▁quiet', ',', '▁and', '▁neither', '▁is', '▁Mr', '.', '▁Herr', 'era',
                                                "'", 's', '▁sn', 'oring', '.', '<0x0A>', '##', '#', '<0x0A>', 'AN', 'SW', 'ER', ':', '▁A',
                                                '.']


def test_conv_process():
    import re

    from mllm.conversation import get_conv_template
    from mllm.dataset.conv import (
        Name2ConvProcess,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IMAGE_PATCH_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN
    )

    cpc = Name2ConvProcess['llava_v1']()

    conv_template = get_conv_template('vicuna_v1.1')

    processor = dict(
        multimodal_cfg=dict(
            image_token_len=256,
            is_multimodal=True,
            sep_image_conv_front=False,
            use_im_start_end=True,
        )
    )

    sources = [
        {
            "from": "human",
            'value': "Can you tell me the whereabouts of the dog? <image>",
        },
        {
            "from": "gpt",
            "value": "(0.1,0.2,0.1,0.2)",
        },
    ]
    assert DEFAULT_IMAGE_TOKEN in sources[0]['value']
    res = cpc(sources, processor, conv_template)
    question = res[0]['value']
    assert DEFAULT_IMAGE_PATCH_TOKEN in question
    assert len(re.findall(DEFAULT_IMAGE_PATCH_TOKEN, question)) == processor['multimodal_cfg']['image_token_len']
    if processor['multimodal_cfg']['use_im_start_end']:
        st = len(re.findall(DEFAULT_IM_START_TOKEN, question))
        ed = len(re.findall(DEFAULT_IM_END_TOKEN, question))
        assert 0 < st == ed > 0
    print(res)


def test_image_process():
    from PIL import Image
    from transformers import CLIPImageProcessor

    from mllm.dataset.conv import Name2ImageProcess

    ipc = Name2ImageProcess['llava_v1']()

    image_processor = CLIPImageProcessor()

    processor = dict(
        image=image_processor,
    )

    images = []
    res = ipc(images, processor)
    assert res['image'].shape[-2] == processor['image'].crop_size['height']
    assert res['image'].shape[-1] == processor['image'].crop_size['width']
    print(res['image'].shape)

    images = [Image.open(r'D:/home/dataset/mscoco/images/train2014/COCO_train2014_000000570019.jpg')]
    print(images[0].height, images[0].width)
    res = ipc(images, processor)
    assert res['image'].shape[-2] == processor['image'].crop_size['height']
    assert res['image'].shape[-1] == processor['image'].crop_size['width']
    print(res['image'].shape)


def test_image_process_2():
    from PIL import Image
    from transformers import CLIPImageProcessor

    from mllm.dataset.conv import Name2ImageProcess

    ipc = Name2ImageProcess['llava_v1']()

    image_processor = CLIPImageProcessor()
    image_processor.crop_size = {'height': 300, 'width': 250}

    processor = dict(
        image=image_processor,
    )

    images = []
    res = ipc(images, processor)
    assert res['image'].shape[-2] == processor['image'].crop_size['height']
    assert res['image'].shape[-1] == processor['image'].crop_size['width']
    print(res['image'].shape)

    images = [Image.open(r'D:/home/dataset/mscoco/images/train2014/COCO_train2014_000000570019.jpg')]
    print(images[0].height, images[0].width)
    res = ipc(images, processor)
    assert res['image'].shape[-2] == processor['image'].crop_size['height']
    assert res['image'].shape[-1] == processor['image'].crop_size['width']
    print(res['image'].shape)
