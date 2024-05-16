# Link-Context Learning for Multimodal LLMs [CVPR 2024]

<p align="center" width="100%">
<img src="ISEKAI_overview.png"  width="80%" height="80%">
</p>

<div>
<div align="center">
    <a href='https://macavityt.github.io/' target='_blank'>Yan Tai<sup>*,2,3,4</sup></a>&emsp;
    <a href='https://weichenfan.github.io/Weichen/' target='_blank'>Weichen Fan<sup>*,†,3</sup></a>&emsp;
    <a href='https://zhaozhang.net/' target='_blank'>Zhao Zhang<sup>3</sup></a>&emsp;
    <a href='https://liuziwei7.github.io/' target='_blank'>Ziwei Liu<sup>&#x2709,1</sup></a>
</div>
<div>
<div align="center">
    <sup>1</sup>S-Lab, Nanyang Technological University&emsp;
    <sup>2</sup>Shanghai Jiao Tong University&emsp;
    <sup>3</sup>SenseTime Research&emsp;
    <br><sup>4</sup>Ningbo Institute of Digital Twin, Eastern Institute of Technology, Ningbo, China<br>&emsp;
    </br>
    <sup>*</sup> Equal Contribution&emsp;
    <sup>†</sup> Project Lead&emsp;
    <sup>&#x2709</sup> Corresponding Author
</div>
 
 -----------------

![](https://img.shields.io/badge/ISEKAI-v0.1-darkcyan)
![](https://img.shields.io/github/stars/isekai-portal/Link-Context-Learning)
![](https://black.readthedocs.io/en/stable/_static/license.svg)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fisekai-portal%2FLink-Context-Learning&count_bg=%23BDC4B7&title_bg=%2342C4A8&icon=octopusdeploy.svg&icon_color=%23E7E7E7&title=visitors&edge_flat=true)](https://hits.seeyoufarm.com)
[![Dataset](https://img.shields.io/badge/Dataset-Download-blue)](https://huggingface.co/ISEKAI-Portal) 
[![Generic badge](https://img.shields.io/badge/DEMO-LCL_Demo-<COLOR>.svg)](http://117.144.81.99:20488/)

Official PyTorch implementation of "[Link-Context Learning for Multimodal LLMs](https://arxiv.org/pdf/2308.07891.pdf)" [CVPR 2024].

## Updates
- **28 Feb, 2024** :boom::boom: Our paper has been accepted by CVPR 2024! 🎉
- **05 Sep, 2023**: We release the code, data, and [LCL-2WAY-WEIGHT](https://huggingface.co/ISEKAI-Portal/LCL_2WAY_WEIGHT) checkpoint.
- **24 Aug, 2023**: We release the online demo at [🔗LCL-Demo🔗](http://117.144.81.99:20488/).
- **17 Aug, 2023**: We release the two subsets of ISEKAI (ISEKAI-10 and ISEKAI-pair) at [[Hugging Face 🤗]](https://huggingface.co/ISEKAI-Portal).

---
This repository contains the **official implementation** and **dataset** of the following paper:

> **Link-Context Learning for Multimodal LLMs**<br>
> https://arxiv.org/abs/2308.07891
>
> **Abstract:** *The ability to learn from context with novel concepts, and deliver appropriate responses are essential in human conversations. Despite current Multimodal Large Language Models (MLLMs) and Large Language Models (LLMs) being trained on mega-scale datasets, recognizing unseen images or understanding novel concepts in a training-free manner remains a challenge. In-Context Learning (ICL) explores training-free few-shot learning, where models are encouraged to "learn to learn" from limited tasks and generalize to unseen tasks. In this work, we propose link-context learning (LCL), which emphasizes "reasoning from cause and effect" to augment the learning capabilities of MLLMs. LCL goes beyond traditional ICL by explicitly strengthening the causal relationship between the support set and the query set. By providing demonstrations with causal links, LCL guides the model to discern not only the analogy but also the underlying causal associations between data points, which empowers MLLMs to recognize unseen images and understand novel concepts more effectively. To facilitate the evaluation of this novel approach, we introduce the ISEKAI dataset, comprising exclusively of unseen generated image-label pairs designed for link-context learning. Extensive experiments show that our LCL-MLLM exhibits strong link-context learning capabilities to novel concepts over vanilla MLLMs.*

  
## Todo

1. [x] Release the [ISEKAI-10](https://huggingface.co/datasets/ISEKAI-Portal/ISEKAI-10) and [ISEKAI-pair](https://huggingface.co/datasets/ISEKAI-Portal/ISEKAI-pair).
2. [x] Release the dataset usage.
3. [x] Release the demo.
4. [x] Release the codes and checkpoints.
5. [ ] Release the full ISEKAI dataset.
6. [ ] Release checkpoints supporting few-shot detection and vqa tasks.


## Get Start

- [Install](#install)
- [Checkpoint](#checkpoint)
- [Dataset](#dataset)
- [Demo](#demo)

## Install

```shell
conda create -n lcl python=3.10
conda activate lcl
pip install -r requirements.txt
```

### configure accelerate

```shell
accelerate config
```
## Dataset

### ImageNet

We train the LCL setting on our rebuild ImageNet-900 set, and evaluate model on ImageNet-100 set. You can get the dataset json [here](https://github.com/isekai-portal/Link-Context-Learning/tree/main/docs).

### ISEKAI
We evaluate model on ISEKAI-10 and ISEKAI-Pair, you can download ISEKAI Dataset in [ISEKAI-10](https://huggingface.co/datasets/ISEKAI-Portal/ISEKAI-10) and [ISEKAI-pair](https://huggingface.co/datasets/ISEKAI-Portal/ISEKAI-pair).


## Checkpoint
Download our [LCL-2WAY-WEIGHT](https://huggingface.co/ISEKAI-Portal/LCL_2WAY_WEIGHT/tree/main) and [LCL-MIX](https://huggingface.co/ISEKAI-Portal/LCL-Mix) checkpoints in huggingface. 



## Demo

To launch a Gradio web demo, use the following command. Please note that the model evaluates in the torch.float16 format, which requires a GPU with at least 16GB of memory.

```shell
python ./mllm/demo/demo.py --model_path /path/to/lcl/ckpt
```

It is also possible to use it in 8-bit quantization, albeit at the expense of sacrificing some performance.

```shell
python ./mllm/demo/demo.py --model_path /path/to/lcl/ckpt --load_in_8bit
```

## Train

After preparing [data](https://github.com/shikras/shikra/blob/main/docs/data.md), you can train the model using the command:

### LCL-2Way-Weight
```shell
accelerate launch --num_processes 4 \
        --main_process_port 23786 \
        mllm/pipeline/finetune.py \
        config/lcl_train_2way_weight.py \
        --cfg-options data_args.use_icl=True \
        --cfg-options model_args.model_name_or_path=/path/to/init/checkpoint
```

### LCL-2Way-Mix
```shell
accelerate launch --num_processes 4 \
        --main_process_port 23786 \
        mllm/pipeline/finetune.py \
        config/lcl_train_mix1.py \
        --cfg-options data_args.use_icl=True \
        --cfg-options model_args.model_name_or_path=/path/to/init/checkpoint
```
## Inference

After preparing [data](#dataset), you can inference the model using the command:

### ImageNet-100
```shell
accelerate launch --num_processes 4 \
        --main_process_port 23786 \
        mllm/pipeline/finetune.py \
        config/lcl_eval_ISEKAI_10.py \
        --cfg-options data_args.use_icl=True \
        --cfg-options model_args.model_name_or_path=/path/to/checkpoint
```

mmengine style args and huggingface:Trainer args are supported. for example, you can change eval batchsize like this:

### ISEKAI
```shell
# ISEKAI10
accelerate launch --num_processes 4 \
        --main_process_port 23786 \
        mllm/pipeline/finetune.py \
        config/shikra_eval_multi_pope.py \
        --cfg-options data_args.use_icl=True \
        --cfg-options model_args.model_name_or_path=/path/to/checkpoint \
        --per_device_eval_batch_size 1

# ISEKAI-PAIR
accelerate launch --num_processes 4 \
        --main_process_port 23786 \
        mllm/pipeline/finetune.py \
        config/shikra_eval_multi_pope.py \
        --cfg-options data_args.use_icl=True \
        --cfg-options model_args.model_name_or_path=/path/to/checkpoint \
        --per_device_eval_batch_size 1
```

where `--cfg-options a=balabala b=balabala` is mmengine style argument. They will overwrite the argument predefined in config file. And `--per_device_eval_batch_size` is huggingface:Trainer argument.

the prediction result will be saved in `output_dir/multitest_xxxx_extra_prediction.jsonl`, which hold the same order as the input dataset. 

## Cite

```bibtex
@inproceedings{tai2023link,
  title={Link-Context Learning for Multimodal LLMs},
  author={Tai, Yan and Fan, Weichen and Zhang, Zhao and Liu, Ziwei},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR)},
  year={2024}
}
```
