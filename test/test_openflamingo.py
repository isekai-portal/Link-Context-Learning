def test_tokenizer():
    from pathlib import Path
    from mllm.models.flamingo import FlamingoTokenizer
    tk = FlamingoTokenizer.from_pretrained(Path(__file__).parent / 'llama_7b_hf')
    assert tk.convert_tokens_to_ids('<|endofchunk|>') != tk.convert_tokens_to_ids('<unk>')
    assert tk.convert_tokens_to_ids('<image>') != tk.convert_tokens_to_ids('<unk>')
    assert tk.convert_tokens_to_ids('<PAD>') != tk.convert_tokens_to_ids('<unk>')
