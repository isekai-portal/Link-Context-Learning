from fastchat.conversation import SeparatorStyle, Conversation, register_conv_template, get_conv_template

__all__ = [SeparatorStyle, Conversation, register_conv_template, get_conv_template]

if __name__ == '__main__':
    IGNORE_INDEX = -100

    from typing import Dict
    import transformers


    def preprocess_v1(
            sources,
            tokenizer: transformers.PreTrainedTokenizer,
            conv,
    ) -> Dict:
        conv = conv.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

        # Tokenize conversations
        input_ids = tokenizer(
            conversations[0],
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        targets = input_ids.clone()

        assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

        # Mask targets
        sep = conv.sep + conv.roles[1] + ": "
        conversation, target = conversations[0], targets
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

        return dict(
            input_ids=input_ids,
            labels=targets,
        )


    from mllm.models.flamingo import FlamingoTokenizer

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
    ret = preprocess_v1(sources, tk, get_conv_template('vicuna_v1.1'))
    print(ret)
