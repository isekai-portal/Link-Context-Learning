统一部分开源多模态LLM数据接口、训练/推理脚本，方便快速切换数据集/模型进行验证。

[TOC]

## Datasets

- [ ] ScienceQA
- [x] ref4-gnome
- [x] ref3-coco
- [x] ref_reverse(grounded caption)
- [x] flickr30k
- [x] flickr30k_reverse(multi object grounded caption)

## Models

- [x] llava_v1_7b
- [x] openflamingo(only support training in otter format. i.e. add\<answer>token)
- [x] otter

## Getting Started

### Dependence Installation

#### 1. 安装torch/torchvision/transformers

A100 参考 [LLaVA训练环境依赖](https://www.yuque.com/z_zhang/ab73nw/rwxn03tibq0kw15e). 

V100 参考 [LLaVA训练环境依赖 5. Pytorch 6. Transformers](https://www.yuque.com/z_zhang/ab73nw/rwxn03tibq0kw15e). V100不支持flash-attn, 可以不用配置.

#### 2. 安装其他依赖

```shell
pip install -r requirements.txt
```

#### 3. 配置accelerate

##### option 1 使用命令行配置

```shell
accelerate config
```

根据提示配置

##### option 2 复制默认配置

将默认配置复制到目标文件夹下

```shell
mkdir -p ~/.cache/huggingface/accelerate
cp accelerate_config/default_config.yaml ~/.cache/huggingface/accelerate/default_config.yaml
```

#### 4. 配置ceph(可选)

需要用到ceph文件系统时配置

```shell
python -m pip install setuptools==59.2.0
python -m pip install pip==21.3.1
python -m pip install -i http://pkg.sensetime.com/repository/pypi-proxy/simple/ --trusted-host pkg.sensetime.com http://10.5.41.14/packages/petrel-oss-sdk.tar.gz --user
```

测试ceph是否安装成功：

```shell
python -c 'from petrel_client.version import version; print(version)'
```

### 数据处理

需要实现Dataset类，在调用`__getitem__`时返回如下格式的item. 之后的格式转换SingleImageConvDataset会为你完成，细节见 mllm/dataset/single_image_convsation.py: class SingleImageConvDataset.

```python
item = {
    'image': None,  # PIL.Image.Image,
    'target': {
        # xmin, ymin, xmax, ymax
        'boxes': [
            [10, 10, 256, 265],  # dog1
            [24, 18, 378, 768],  # dog2
            [100, 310, 670, 653],  # man
            [278, 320, 809, 673],  # rope
        ],
        'points': [
            [100, 100],  # man1
            [200, 200],  # man2
        ]
    },

    "conversations": [
        {
            'from': 'human',
            'value': 'What is the relation between the two dogs <boxes> and the man <boxes> in the image <image> ? Is the man<points> shaking hands with the man<points>?',
            'boxes_seq': [[0, 1], [2], ],
            'points_seq': [[0,], [1,]],
        },
        {
            'from': 'gpt',
            'value': 'a rope <boxes> is connecting the left dog <boxes> with the man <boxes>. '
                     'So the man <boxes> is walking the dog <boxes>.'
                    'And the man <boxes> has no relationship with the right dog <boxes>',
            'boxes_seq': [[3], [0], [2], [2], [0], [2], [1]],
            'points_seq': None,
        }
    ]
}
```

### Training

#### 在REC上训练LLaVA

```shell
sbatch -p A10080G-share \
    -n 1 \
    -N 1 \
    --gres=gpu:4 \
    -c 64 \
    accelerate launch --num_processes 4 \
        --main_process_port 23786 \
        mllm/pipeline/finetune.py \
        config/rec_llava_v1_7b_fsdp.py
```

如果main_process_port被占用，随便换一个就好

#### 使用flash-atten进行训练（仅A100支持）

```shell
sbatch -p A10080G-share \
    -n 1 \
    -N 1 \
    --gres=gpu:4 \
    -c 64 \
    accelerate launch --num_processes 4 \
        --main_process_port 23786 \
        mllm/pipeline/finetune_mem.py \
        config/rec_llava_v1_7b_fsdp.py
```

即：将脚本从finetune.py切换为finetune_mem.py. 使用flash-atten进行验证同理.

#### 在REC上训练指定ckpt，并更改epoch learning_rate等参数

```shell
sbatch -p A10080G-share \
    -n 1 \
    -N 1 \
    --gres=gpu:4 \
    -c 64 \
    accelerate launch --num_processes 4 \
        --main_process_port 23786 \
        mllm/pipeline/finetune.py \
        config/rec_llava_v1_7b_fsdp.py \
            --cfg-options model_args.model_name_or_path='/path/to/new/ckpt' \
            --num_train_epochs 1 \
            --learning_rate 4e-5
```

支持Transformers.Seq2SeqArguments的所有参数, 可以直接--xxx  value进行更改, 这部分参数和控制训练/推理行为相关. 对于其中不经常变动的参数, 也可以写到training_args中存到文件中, 脚本会自动将这些参数合并起来送给Transformers.Seq2SeqArguments. 实现详见mllm/config/config.py

支持mmengine类似的从文件中读取参数，要修改该部分参数，要使用--cfg-options xxx1=value xxx2=value进行更改，这部分参数和模型、数据等其他行为相关

### Evaluation & Predict

#### 在REC上推理LLaVA

```shell
sbatch -p A10080G-share \
    -n 1 \
    -N 1 \
    --gres=gpu:4 \
    -c 64 \
    accelerate launch --num_processes 4 \
        --main_process_port 23786 \
        mllm/pipeline/finetune.py \
        config/rec_llava_v1_7b_fsdp_eval.py
```

本质上是由Transformers.Seq2SeqArguments的do_train, do_eval, do_predict参数分别控制是否做训练、验证、测试. 详情参考mllm/pipeline/finetune.py文件.
