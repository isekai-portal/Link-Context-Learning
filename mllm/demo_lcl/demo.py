import os
import sys
import logging
import argparse
import warnings
from pathlib import Path

import torch
import numpy as np
import gradio as gr
from gradio.themes.utils.sizes import Size
from PIL import Image
from PIL import ImageDraw, ImageFont
from mmengine import Config
import transformers
from transformers import BitsAndBytesConfig

# import debugpy;debugpy.connect(('10.142.4.66', 5610))

SLURM_ENV = {k: v for k, v in os.environ.items() if 'SLURM' in k}
sys.path.append(str(Path(__file__).parent.parent.parent))

from mllm.models.builder.build_llava import load_pretrained_llava
from mllm.dataset.process_function import PlainBoxFormatter
from demo_dataset import prepare_demo_dataset

log_level = logging.DEBUG
transformers.logging.set_verbosity(log_level)
transformers.logging.enable_default_handler()
transformers.logging.enable_explicit_format()

TEMP_FILE_DIR = Path(__file__).parent / 'temp'
TEMP_FILE_DIR.mkdir(parents=True, exist_ok=True)
#########################################
# mllm model init
#########################################

#region paser
parser = argparse.ArgumentParser("LCL Web Demo")
parser.add_argument('--base_model', default='llama', choices=['llama'])
parser.add_argument('--model_path', default=r'/home/taiyan/ckpt/okapis/demo_mix_1w')
parser.add_argument('--server_name', default=SLURM_ENV.get('SLURM_JOB_NODELIST', None))
parser.add_argument('--server_port', type=int, default=20488)
parser.add_argument('--load_in_8bit', action='store_true')
parser.add_argument('--load_in_4bit', action='store_true')

args = parser.parse_args()
args.cluster_mode = bool(SLURM_ENV)
if args.load_in_4bit and args.load_in_8bit:
    warnings.warn("use `load_in_4bit` and `load_in_8bit` at the same time. ignore `load_in_8bit`")
    args.load_in_8bit = False
print(args)

model_name_or_path = args.model_path
if args.cluster_mode:
    vision_tower_path = r'/mnt/lustre/share_data/chenkeqin/VG/ckpt/openai/clip-vit-large-patch14'  # offline
else:
    vision_tower_path = r'/home/chenkeqin/openai/clip-vit-large-patch14'
#endregion

#region configs
#TODO: 修改各种参数
model_args = dict(
    type='llava',
    # TODO: process version; current version use default version
    version='v1',

    # checkpoint config
    cache_dir=None,
    model_name_or_path=model_name_or_path,
    vision_tower=vision_tower_path,
    pretrain_mm_mlp_adapter=None,
    # model config
    mm_vision_select_layer=-2,
    model_max_length=2048,
    
    # finetune config
    freeze_backbone=False,
    tune_mm_mlp_adapter=False,
    freeze_mm_mlp_adapter=False,
    freeze_mm_projector=False,

    # data process config
    is_multimodal=True,
    sep_image_conv_front=False,
    image_token_len=256,
    mm_use_im_start_end=True,

    target_processor=dict(
        boxes=dict(type='PlainBoxFormatter'),
    ),

    process_func_args=dict(
        conv=dict(type='LLavaConvProcessV1'),
        target=dict(type='BoxFormatProcess'),
        text=dict(type='LlavaTextProcessV1'),
        image=dict(type='LlavaImageProcessorV1'),
    ),

    conv_args=dict(
        conv_template=['causal_v1.0', 'vicuna_v1.1'],
        transforms=dict(type='Expand2square'),
        tokenize_kwargs=dict(truncation_size=2048),
    ),

    gen_kwargs_set_pad_token_id=True,
    gen_kwargs_set_bos_token_id=True,
    gen_kwargs_set_eos_token_id=True,
)
model_args = Config(model_args)

training_args = Config(dict(
    bf16=False,
    fp16=True,
    device='cuda',
    fsdp=None,
))

if args.load_in_4bit:
    quantization_kwargs = dict(
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    )
elif args.load_in_8bit:
    quantization_kwargs = dict(
        quantization_config=BitsAndBytesConfig(
            load_in_8bit=True,
        )
    )
else:
    quantization_kwargs = dict()

#endregion



#region Load model and dataset
model, preprocessor = load_pretrained_llava(model_args, training_args, **quantization_kwargs)
preprocessor['target'] = {'boxes': PlainBoxFormatter()}
tokenizer = preprocessor['text']

if not getattr(model, 'is_quantized', False):
    model.to(dtype=torch.float16, device=torch.device('cuda'))
if not getattr(model.model.vision_tower[0], 'is_quantized', False):
    model.model.vision_tower[0].to(dtype=torch.float16, device=torch.device('cuda'))

dataset_demo = prepare_demo_dataset(model_args=model_args, preprocessor=preprocessor)

print(f"LLM device: {model.device}, is_quantized: {getattr(model, 'is_quantized', False)}, is_loaded_in_4bit: {getattr(model, 'is_loaded_in_4bit', False)}, is_loaded_in_8bit: {getattr(model, 'is_loaded_in_8bit', False)}")
print(f"vision device: {model.model.vision_tower[0].device}, is_quantized: {getattr(model.model.vision_tower[0], 'is_quantized', False)}, is_loaded_in_4bit: {getattr(model, 'is_loaded_in_4bit', False)}, is_loaded_in_8bit: {getattr(model, 'is_loaded_in_8bit', False)}")
#endregion

#########################################
# gradio utils
#########################################
def setup_gradio_warning(level=1):
    """
    level            0       1           2        3
    level          IGNORE   Weak       Strong    Error
    has Warning      _foo   Warning    Warning   Error
    no Warning       _foo    _foo      Error     Error
    """

    def _dummy_func(*args, **kwargs):
        pass

    def _raise_error(*args, **kwargs):
        raise gr.Error(*args, **kwargs)

    assert level in [0, 1, 2, 3]
    if level >= 3:
        return _raise_error
    if level <= 0:
        return _dummy_func
    if hasattr(gr, 'Warning'):
        return gr.Warning
    if level == 1:
        return _dummy_func
    return _raise_error

grWarning = setup_gradio_warning()

def predict(data_meta, history):
    dataset_demo.update_data(data_meta)
    model_inputs = dataset_demo[0]
    model_dtype = next(model.parameters()).dtype
    model_inputs['images'] = model_inputs['images'].to(model_dtype)
    # print(f"model_inputs: {model_inputs}")

    gen_kwargs = dict(
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=256,
    )

    input_ids = model_inputs['input_ids']
    with torch.inference_mode():
        with torch.autocast(dtype=torch.float16, device_type='cuda'):
            outputs = model.generate(**model_inputs, **gen_kwargs, return_dict_in_generate=True, output_scores=True)
            output_ids = outputs.sequences

            transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
            generated_tokens = outputs.sequences[:, input_ids.shape[-1]:]
            import numpy as np
            for tok, score, full_score in zip(generated_tokens[0], transition_scores[0], outputs.scores):
                # | token | token string | logits | probability
                # print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.cpu().numpy():.3f} | {np.exp(score.cpu().numpy()):.2%}", end='\t')
                full_score = full_score[0]
                topk_softmax_score, topk_index = full_score.softmax(dim=-1).topk(5)
                topk_origin_score = full_score[topk_index]
                topk_tokens = tokenizer.convert_ids_to_tokens(topk_index)
                topk_strs = [f"[{idx:5d} | {token:8s} | {oscore:.3f} | {sscore:.2%}]" for idx, token, oscore, sscore in zip(topk_index, topk_tokens, topk_origin_score, topk_softmax_score)]
                # print(",".join(topk_strs))

            # print(tokenizer.batch_decode(generated_tokens))

    input_token_len = input_ids.shape[-1]
    response = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    print(f"response: {response}")

    history = history + [(data_meta['infer_q'][0], response)]
    return data_meta, history

if __name__ == '__main__':
    text_custom = Size(
        name="text_ty",
        xxs="9px",
        xs="10px",
        sm="12px",
        md="16px",
        lg="16px",
        xl="22px",
        xxl="26px",
    )

    with gr.Blocks(title="LCL", theme=gr.themes.Base(text_size=text_custom)).queue() as demo:
        logo_file_url = f"file={os.path.join(os.path.dirname(__file__), 'examples_icl/logo.png')}"
        gr.HTML(
            f"""
            <p align="center"><img src="{logo_file_url}" alt="Logo" width="200"></p>
            <h1 align="center"><font color="#966661">Link-Context Learning for Multimodal LLMs</h1>
            <p align="center">
                <a href='https://github.com/isekai-portal/Link-Context-Learning' target='_blank'>[Project]</a>
                <a href='https://arxiv.org/abs/2308.07891' target='_blank'>[Paper]</a>
            </p>

            <p>
                <font color="#966661"><strong>Link-Context Learning(LCL)</strong></font> emphasizes "reasoning from cause and effect" to augment the learning capabilities of MLLMs.
            </p>

            <h2>User Manual</h2>
            <ul>
                <li><strong>Step 1.</strong> Upload support images.</li>
                <li><strong>Step 2.</strong> Select Question Format in <code>Task Template</code>. Task template and user input (if exists) will be assembled into final inputs to the model.</li>
                <li><strong>Notes:</strong>
                    <ul>
                    <li>We only support 2-shot(2 positive and 2 negative samples) in this demo.</li>
                    <li>Watting for supplementary.</li>
                    </ul>
                </li>
            </ul>

            """
        )

        gr.HTML(
            """ 
            <h2>Support Samples</h2>
            """
        )
        with gr.Row(variant='compact'):
            with gr.Column():
                with gr.Group():
                    with gr.Row():
                        pos_imgbox1 = gr.Image(label="Positive Support Image 1")
                        pos_imgbox2 = gr.Image(label="Positive Support Image 2(Optional)")
                    pos_q = gr.Textbox(placeholder="What is in the image?",label="Question")
                    pos_a = gr.Textbox(placeholder="The answer is ...",label="Answer")

            with gr.Column():
                with gr.Group():
                    with gr.Row():
                        neg_imgbox1 = gr.Image(label="Negative Support Image 1")
                        neg_imgbox2 = gr.Image(label="Negative Support Image 2(Optional)")
                    neg_q = gr.Textbox(placeholder="What is in the image?",label="Question")
                    neg_a = gr.Textbox(placeholder="The answer is ...",label="Answer")

        gr.HTML(
            """ 
            <h2>Query Sample</h2>
            """
        )
        
        with gr.Row():
            with gr.Column():
                infer_imgbox = gr.Image(label="Query Image")
                infer_q = gr.Textbox(placeholder="What is in the image?",label="Question")
                with gr.Row():
                    submit_btn = gr.Button('Submit')
                    clear_btn = gr.ClearButton()
            chatbot = gr.Chatbot()

        ##############################################
        #  init state
        ##############################################

        def init_state():
            return {
                'pos_img': [],
                'neg_img': [],
                'infer_img': [],
                'pos_q': [],
                'pos_a': [],
                'neg_q': [],
                'neg_a': [],
                'infer_q': []
            }

        def set_state(custom_state, pos_imgbox1, pos_imgbox2, pos_q, pos_a,\
                neg_imgbox1, neg_imgbox2, neg_q, neg_a, infer_imgbox, infer_q):
            def _convert(img):
                if img is None:
                    return
                img = Image.fromarray(img)
                return img
            
            def _append(key, value):
                if value is None:
                    return
                if '_q' in key:
                    value = value.replace('<image>', '').replace('<im_start>', '').replace('<im_end>', '')
                    value += '<image>'
                    if key == 'pos_q' or key == 'neg_q':
                        value = '[INSTRUCTION]' + value
                    elif key == 'infer_q':
                        if len(custom_state['pos_img']) != 0 or len(custom_state['neg_img']) != 0:
                            value = '[FINAL QUESTION]' + value

                custom_state[key].append(value)

            # set state
            custom_state = init_state()
            pos_imgbox1 = _convert(pos_imgbox1)
            pos_imgbox2 = _convert(pos_imgbox2)
            neg_imgbox1 = _convert(neg_imgbox1)
            neg_imgbox2 = _convert(neg_imgbox2)
            infer_imgbox = _convert(infer_imgbox)

            _append('pos_img', pos_imgbox1)
            _append('pos_img', pos_imgbox2)
            _append('neg_img', neg_imgbox1)
            _append('neg_img', neg_imgbox2)

            _append('pos_q', pos_q)
            _append('pos_a', pos_a)

            _append('neg_q', neg_q)
            _append('neg_a', neg_a)

            _append('infer_img', infer_imgbox)
            _append('infer_q', infer_q)
            
            return custom_state

        custom_state = gr.State(init_state())
        example_state = gr.State(init_state())

        ##############################################
        #  Examples
        ##############################################


        # control functions
        submit_btn.click(fn=set_state,
            inputs=[custom_state, pos_imgbox1, pos_imgbox2, pos_q, pos_a,\
                neg_imgbox1, neg_imgbox2, neg_q, neg_a, infer_imgbox, infer_q],
            outputs=[custom_state],
            show_progress=True,
            queue=True).then(fn=predict, 
                        inputs=[custom_state, chatbot],  
                        outputs=[custom_state, chatbot],  
                        show_progress=True, 
                        queue=True)

        clear_btn.add([chatbot, pos_imgbox1, pos_imgbox2, pos_q, pos_a,\
                neg_imgbox1, neg_imgbox2, neg_q, neg_a, infer_imgbox, infer_q])

    # demo.launch(server_name='0.0.0.0', server_port=args.server_port)
    demo.launch(server_name=args.server_name, server_port=args.server_port)