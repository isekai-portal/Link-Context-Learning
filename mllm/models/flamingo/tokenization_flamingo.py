from transformers import LlamaTokenizer


class FlamingoTokenizer(LlamaTokenizer):
    def __init__(self, vocab_file, **kwargs):
        super().__init__(vocab_file, **kwargs)
        # add Flamingo special tokens to the tokenizer
        self.add_special_tokens(
            {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
        )
        if self.pad_token is None:
            # Issue: GPT models don't have a pad token, which we use to
            # modify labels for the loss.
            self.add_special_tokens({"pad_token": "<PAD>"})
