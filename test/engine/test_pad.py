def test_pad(cfg_dir):
    from transformers import LlamaTokenizer
    from transformers import DataCollatorForSeq2Seq

    tk = LlamaTokenizer.from_pretrained(cfg_dir.LLAMA_7B_HF_PATH)

    if tk.pad_token is None:
        tk.pad_token = tk.unk_token
    x = [
        'A chat between a curious user',
        "A chat between a curious user and an artificial intelligence assistant."
    ]
    dc = DataCollatorForSeq2Seq(tk)
    IGNORE_INDEX = -100

    tk.padding_side = 'right'
    r = [tk(_, return_token_type_ids=False) for _ in x]
    rwl = [{**d, 'labels': d['input_ids']} for d in r]
    res = dc(rwl)
    assert res['input_ids'][0][0] != tk.pad_token_id == res['input_ids'][0][-1]
    assert res['attention_mask'][0][0] != 0 == res['attention_mask'][0][-1]
    assert res['labels'][0][0] != IGNORE_INDEX == res['labels'][0][-1]

    tk.padding_side = 'left'
    r = [tk(_, return_token_type_ids=False) for _ in x]
    rwl = [{**d, 'labels': d['input_ids']} for d in r]
    res = dc(rwl)
    assert res['input_ids'][0][0] == tk.pad_token_id != res['input_ids'][0][-1]
    assert res['attention_mask'][0][0] == 0 != res['attention_mask'][0][-1]
    assert res['labels'][0][0] == IGNORE_INDEX != res['labels'][0][-1]
