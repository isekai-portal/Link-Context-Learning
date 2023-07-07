import os
import sys
import pathlib
import logging
import time

SLURM_ENV = {k: v for k, v in os.environ.items() if 'SLURM' in k}
print(f"SLURM_ENV: {SLURM_ENV}")
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from mllm.config import prepare_args
from mllm.engine import prepare_trainer_collator
from mllm.dataset import prepare_data, smart_prepare_target_processor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)


def main():
    cfg, training_args = prepare_args(['config/dummy_llava_train.py', '--overwrite_output_dir', '--fp16_full_eval=False', '--fp16=False'])

    from transformers import CLIPImageProcessor, LlamaTokenizer
    from mllm.dataset.process_function import PlainBoxFormatter
    target_processor = {'boxes': PlainBoxFormatter()}
    LLAVA_7B_TK_PATH = r'./test/llava_7b_tk'
    tokenizer = LlamaTokenizer.from_pretrained(LLAVA_7B_TK_PATH)
    tokenizer.pad_token_id = tokenizer.unk_token_id
    preprocessor = dict(
        image=CLIPImageProcessor(),
        text=tokenizer,
        target=target_processor,
        conv=dict(
            image_token_len=256,
            is_multimodal=True,
            sep_image_conv_front=False,
            use_im_start_end=True,
        )
    )

    # Prepare data_collator
    collator_kwargs = cfg.data_args.collator_kwargs
    trainer_cls, data_collator_dict = prepare_trainer_collator(cfg.model_args, preprocessor, collator_kwargs)
    dataset, compute_metrics = prepare_data(cfg.data_args, cfg.model_args, training_args, preprocessor)

    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from mllm.utils import decode_generate_ids, show, draw_bounding_boxes
    from mllm.dataset import PlainBoxFormatter
    from matplotlib import pyplot as plt
    pbf = PlainBoxFormatter()

    st = time.time()
    print(dataset['train'])
    print(len(dataset['train']))
    for idx, item in enumerate(tqdm(dataset['train'].dataset)):
        if idx > 50:
            break
        print(item)
        pb = [item['target']['points'][0][0], item['target']['points'][0][1], item['target']['points'][0][0] + 10, item['target']['points'][0][1]+10]
        print(pb)
        show(draw_bounding_boxes(item['image'], boxes=[pb], colors='red', width=4))
        plt.savefig('./temp.jpg', dpi=300)
        _ = input()
        # print(item)
        # input_ids = decode_generate_ids(tokenizer, item['input_ids'])
        # print(input_ids)
        # labels = decode_generate_ids(tokenizer, item['labels'])
        # print(labels)
        # extracted = pbf.extract_point(labels)
        # print(extracted)
        # show(draw_bounding_boxes(item['image'], boxes=extracted))
        # plt.savefig('./temp.jpg', dpi=300)

    # dl = DataLoader(dataset['train'], batch_size=8, num_workers=4, collate_fn=data_collator_dict['train_collator'])
    # for i, batch in enumerate(tqdm(dl)):
    #     pass
    # print(f"cost {time.time() - st:.2f} s")


# noinspection PyUnusedLocal
def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
