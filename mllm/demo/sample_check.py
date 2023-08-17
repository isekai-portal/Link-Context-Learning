import glob
import json
import os

import tempfile
from pathlib import Path
import random

import gradio as gr
from PIL import Image
import pandas as pd

import time
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))


from mllm.dataset import Expand2square, PlainBoxFormatter, BoxFormatProcess
from mllm.conversation import get_conv_template

TEMP_FILE_DIR = Path(__file__).parent / 'temp'
TEMP_FILE_DIR.mkdir(parents=True, exist_ok=True)

src_dir = r'/home/chenkeqin/unify_mllm/mllm/demo/samples'


def init_src_dir():
    '''
    get filename and creation time of this folder
    then return a pandas dataframe with the data
    '''
    df = pd.DataFrame(columns=['id', 'filename', 'creation_time'])
    filenames = glob.glob(os.path.join(src_dir, '*.json'))
    filenames.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    for i, filepath in enumerate(filenames):
        filename = os.path.basename(filepath)
        creation_time = time.strftime('%Y/%m/%d', time.localtime(os.path.getmtime(filepath)))
        df.loc[i] = [i, filename, creation_time]

    return df


def init_src_dir_yesterday_today():
    df = init_src_dir()
    df = df[df['creation_time'].isin(
        [time.strftime('%Y/%m/%d', time.localtime(time.time() - 86400)), time.strftime('%Y/%m/%d', time.localtime(time.time()))])]
    return df.values.tolist()


def init_plot():
    '''
    Obtain a dataframe object with two columns, "Date" and "Number of files created on that date in src_dir"
    '''
    df = init_src_dir()
    df = df.groupby('creation_time').count()
    df = df.rename(columns={'filename': 'count'})
    df = df.reset_index()
    return df


def get_full_num():
    '''
    get the number of files in src_dir
    '''
    filenames = os.listdir(src_dir)
    cout = 0
    for filename in filenames:
        if filename.endswith('.json'):
            cout += 1
    return [("Total number of services", str(cout))]


def load_chat(cur_file):
    import base64
    from io import BytesIO

    if cur_file is None:
        return gr.update()
    if not os.path.exists(os.path.join(src_dir, cur_file)):
        print(f'file not exist: {cur_file}')
        return gr.update()
    obj = json.load(open(os.path.join(src_dir, cur_file)))
    if 'img_base64' in obj:
        img_base64 = obj['img_base64']
        pil_image = Image.open(BytesIO(base64.b64decode(img_base64))).convert("RGB")
        obj['image'] = pil_image
        del obj['img_base64']
    else:
        pil_image = None

    def convert_raw_to_core(item):
        transforms = Expand2square()
        if 'image' in item:
            image, target = transforms(item['image'], item.get('target'))
            if target is not None:
                target['width'], target['height'] = image.width, image.height
        else:
            image, target = None, None
        preprocessor = dict(target=dict(boxes=PlainBoxFormatter()))
        process_target = BoxFormatProcess()
        conv, _ = process_target(item['conversations'], target, preprocessor)

        def build_conv(source):
            conv = get_conv_template('vicuna_v1.1')
            role_map = {"human": conv.roles[0], "gpt": conv.roles[1]}
            assert len(source) > 0
            assert source[0]['from'] == 'human'
            for sentence in source:
                role = role_map[sentence['from']]
                conv.append_message(role, sentence['value'])
            return conv

        conv = build_conv(conv)
        return conv

    def convert_one_round_message(conv, image=None):
        def parse_text(text):
            text = text.replace('<', '&lt;').replace('>', '&gt;')
            return text

        def build_boxes_image(text, image):
            if image is None:
                return text, None
            print(text, image)
            import re

            colors = ['#ed7d31', '#5b9bd5', '#70ad47', '#7030a0', '#c00000', '#ffff00', "olive", "brown", "cyan"]
            pat = re.compile(r'\[\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3}(?:;\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3})*\]')

            def extract_boxes(string):
                ret = []
                for bboxes_str in pat.findall(string):
                    bboxes = []
                    bbox_strs = bboxes_str.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(";")
                    for bbox_str in bbox_strs:
                        bbox = list(map(float, bbox_str.split(',')))
                        bboxes.append(bbox)
                    ret.append(bboxes)
                return ret

            def de_norm_box_xyxy(box, *, w, h):
                x1, y1, x2, y2 = box
                x1 = x1 * w
                x2 = x2 * w
                y1 = y1 * h
                y2 = y2 * h
                box = x1, y1, x2, y2
                return box

            def draw_bounding_boxes(
                    image,
                    boxes,
                    **kwargs,
            ):
                import PIL
                import torch
                if isinstance(image, PIL.Image.Image):
                    from torchvision.transforms import PILToTensor
                    image = PILToTensor()(image)
                assert isinstance(image, torch.Tensor), ""

                if not isinstance(boxes, torch.Tensor):
                    boxes = torch.as_tensor(boxes)
                assert isinstance(boxes, torch.Tensor)

                from torchvision.utils import draw_bounding_boxes as _draw_bounding_boxes
                return _draw_bounding_boxes(image, boxes, **kwargs)

            extract_pred = extract_boxes(text)
            boxes_to_draw = []
            color_to_draw = []
            for idx, boxes in enumerate(extract_pred):
                color = colors[idx % len(colors)]
                for box in boxes:
                    boxes_to_draw.append(de_norm_box_xyxy(box, w=image.width, h=image.height))
                    color_to_draw.append(color)
            if not boxes_to_draw:
                _, path = tempfile.mkstemp(suffix='.jpg', dir=TEMP_FILE_DIR)
                image.save(path)
                return text, path
            res = draw_bounding_boxes(image=image, boxes=boxes_to_draw, colors=color_to_draw, width=8)
            from torchvision.transforms import ToPILImage
            res = ToPILImage()(res)
            _, path = tempfile.mkstemp(suffix='.jpg', dir=TEMP_FILE_DIR)
            res.save(path)

            # post process text color
            print(text)
            location_text = text
            edit_text = list(text)
            bboxes_str = pat.findall(text)
            for idx in range(len(bboxes_str) - 1, -1, -1):
                color = colors[idx % len(colors)]
                boxes = bboxes_str[idx]
                span = location_text.rfind(boxes), location_text.rfind(boxes) + len(boxes)
                location_text = location_text[:span[0]]
                edit_text[span[0]:span[1]] = f'<span style="color:{color}; font-weight:bold;">{boxes}</span>'
            text = "".join(edit_text)
            return text, path

        text_query = parse_text(f"{conv[0][0]}: {conv[0][1]}")
        text_answer = parse_text(f"{conv[1][0]}: {conv[1][1]}")
        text_query, image_query = build_boxes_image(text_query, image)
        text_answer, image_answer = build_boxes_image(text_answer, image)

        new_chat = []
        new_chat.append([text_query, None])
        if image_query is not None:
            new_chat.append([(image_query,), None])

        new_chat.append([None, text_answer])
        if image_answer is not None:
            new_chat.append([None, (image_answer,)])
        return new_chat

    conv = convert_raw_to_core(item=obj)
    new_message = [(r, m.replace('<im_patch>', '').replace('<im_end>', '').replace('<im_start>', '<image>')) for r, m in conv.messages]
    new_message = convert_one_round_message(new_message, image=pil_image)

    return new_message


if __name__ == '__main__':
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                gr_df = gr.DataFrame(
                    # init_src_dir_yesterday_today,
                    lambda: init_src_dir().values.tolist(),
                    every=60,
                    overflow_row_behaviour='paginate',
                    headers=["ID", "Filename", "Creation Time"],
                    elem_id="gr_df",
                    label="Yesterday and Today's Samples",
                    interactive=False,
                    visible=False,
                )


                def _init_first_item(gr_df):
                    print(gr_df.value['data'])
                    if len(gr_df.value['data']) == 0 or len(gr_df.value['data'][0]) == 0:
                        return None, None
                    ret = gr_df.value['data'][0][1], gr_df.value['data'][0][2]
                    print(ret)
                    return ret


                init_file, init_time = _init_first_item(gr_df)
                chatbot = gr.Chatbot(load_chat(init_file), elem_id="chatbot", height=900)

            with gr.Column():
                full = gr.HighlightedText(get_full_num, label="Service Statistics", elem_id="full_num")
                gr_plot = gr.LinePlot(init_plot, x="creation_time", y="count", title="Daily traffic statistics", elem_id="gr_plot")
                with gr.Row():
                    Prev = gr.Button("⬅️ Prev")
                    Next = gr.Button("Next ➡️")
                with gr.Row():
                    First = gr.Button("First")
                    Rand = gr.Button("Random")
                    Last = gr.Button("Last")

                with gr.Row():
                    cur_file = gr.Textbox(value=init_file, label="Current Sample ID")
                    cur_index = gr.Number(value=0)
                    cur_time = gr.Textbox(value=init_time)


        def _next_item(gr_df, cur_file):
            if cur_file is None:
                index = 0
            else:
                try:
                    index = gr_df[(gr_df['Filename'] == cur_file)].index.tolist()[0]
                    index += 1
                    if index >= len(gr_df):
                        index = index - len(gr_df)
                except:
                    index = 0

            try:
                cur_file = gr_df.iloc[index]['Filename']
                cur_time = gr_df.iloc[index]['Creation Time']
            except:
                cur_file = None
                cur_time = None
            return cur_file, index, cur_time


        def _prev_item(gr_df, cur_file):
            if cur_file is None:
                index = 0
            else:
                try:
                    index = gr_df[(gr_df['Filename'] == cur_file)].index.tolist()[0]
                    index -= 1
                    if index < 0:
                        index = index + len(gr_df)
                except:
                    index = 0

            try:
                cur_file = gr_df.iloc[index]['Filename']
                cur_time = gr_df.iloc[index]['Creation Time']
            except:
                cur_file = None
                cur_time = None
            return cur_file, index, cur_time


        def _first_item(gr_df, cur_file):
            index = 0
            try:
                cur_file = gr_df.iloc[index]['Filename']
                cur_time = gr_df.iloc[index]['Creation Time']
            except:
                cur_file = None
                cur_time = None
            return cur_file, index, cur_time


        def _randn_item(gr_df, cur_file):
            index = random.randint(0, len(gr_df) - 1)
            try:
                cur_file = gr_df.iloc[index]['Filename']
                cur_time = gr_df.iloc[index]['Creation Time']
            except:
                cur_file = None
                cur_time = None
            return cur_file, index, cur_time


        def _last_item(gr_df, cur_file):
            index = len(gr_df) - 1
            try:
                cur_file = gr_df.iloc[index]['Filename']
                cur_time = gr_df.iloc[index]['Creation Time']
            except:
                cur_file = None
                cur_time = None
            return cur_file, index, cur_time


        def _index_input(gr_df, cur_file, cur_index):
            index = int(cur_index)
            try:
                cur_file = gr_df.iloc[index]['Filename']
                cur_time = gr_df.iloc[index]['Creation Time']
            except:
                cur_file = None
                cur_time = None
            print(cur_file, index, cur_time)
            return cur_file, index, cur_time


        Next.click(_next_item, [gr_df, cur_file], [cur_file, cur_index, cur_time])
        Prev.click(_prev_item, [gr_df, cur_file], [cur_file, cur_index, cur_time])
        First.click(_first_item, [gr_df, cur_file], [cur_file, cur_index, cur_time])
        Rand.click(_randn_item, [gr_df, cur_file], [cur_file, cur_index, cur_time])
        Last.click(_last_item, [gr_df, cur_file], [cur_file, cur_index, cur_time])
        cur_index.input(_index_input, [gr_df, cur_file, cur_index], [cur_file, cur_index, cur_time])

        cur_file.change(load_chat, [cur_file], [chatbot])


        def update_index_and_time(gr_df, cur_file):
            if cur_file is None:
                index = 0
            else:
                try:
                    index = gr_df[(gr_df['Filename'] == cur_file)].index.tolist()[0]
                except:
                    index = 0

            try:
                cur_time = gr_df.iloc[index]['Creation Time']
            except:
                cur_time = None
            return index, cur_time

        cur_file.change(update_index_and_time, [gr_df, cur_file], [cur_index, cur_time])

    print("launching...")
    demo.queue(concurrency_count=3).launch(server_name='0.0.0.0', server_port=24807)
