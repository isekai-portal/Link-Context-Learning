def test_pad():
    from transformers import LlamaTokenizer
    from transformers import DataCollatorForSeq2Seq

    tk = LlamaTokenizer.from_pretrained(r'/test/llava_7b_tk')
    x = [
        'A chat between a curious user and an artificial intelligence assistant. The assistant',
        "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and "
    ]

    dc = DataCollatorForSeq2Seq(tk)

    r = [tk(_, return_token_type_id=False) for _ in x]
    rwl = [{**d, 'labels': d['input_ids']} for d in r]

    res = dc(rwl)
    print(res)

    tk.padding_side = 'left'
    r = [tk(_, return_token_type_id=False) for _ in x]
    rwl = [{**d, 'labels': d['input_ids']} for d in r]
    res = dc(rwl)
    print(res)
