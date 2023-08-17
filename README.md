# Link-Context Learning for Multimodal LLMs

<p align="center" width="100%">
<img src="ISEKAI_overview.png"  width="80%" height="80%">
</p>

<div>
<div align="center">
    Yan Tai<sup>*,1,2</sup></a>&emsp;
    <a href='https://weichenfan.github.io/Weichen/' target='_blank'>Weichen Fan<sup>*,†,1</sup></a>&emsp;
    <a href='https://zhaozhang.net/' target='_blank'>Zhao Zhang<sup>1</sup></a>&emsp;
    <a href='https://zhufengx.github.io/' target='_blank'>Feng Zhu<sup>1</sup></a>&emsp;
    <a href='https://scholar.google.com/citations?user=1c9oQNMAAAAJ&hl=zh-CN' target='_blank'>Rui Zhao<sup>1</sup></a>&emsp;
    <a href='https://liuziwei7.github.io/' target='_blank'>Ziwei Liu<sup>&#x2709,3</sup></a>
</div>
<div>
<div align="center">
    <sup>1</sup>SenseTime Research&emsp;
    <sup>2</sup>Institute of Automation, CAS&emsp;
    <sup>3</sup>S-Lab, Nanyang Technological University&emsp;
    </br>
    <sup>*</sup> Equal Contribution&emsp;
    <sup>†</sup> Project Lead&emsp;
    <sup>&#x2709</sup> Corresponding Author
    
</div>
 
 -----------------

![](https://img.shields.io/badge/ISEKAI-v0.1-darkcyan)
![](https://img.shields.io/github/stars/isekai-portal/Link-Context-Learning)
![](https://black.readthedocs.io/en/stable/_static/license.svg)
[![Dataset](https://img.shields.io/badge/Dataset-Download-blue)](https://huggingface.co/ISEKAI-Portal) 

This repository contains the **official implementation** and **dataset** of the following paper:

> **Link-Context Learning for Multimodal LLMs**<br>
> https://arxiv.org/abs/2308.07891
>
> **Abstract:** *The ability to learn from context with novel concepts, and deliver appropriate responses are essential in human conversations. Despite current Multimodal Large Language Models (MLLMs) and Large Language Models (LLMs) being trained on mega-scale datasets, recognizing unseen images or understanding novel concepts in a training-free manner remains a challenge. In-Context Learning (ICL) explores training-free few-shot learning, where models are encouraged to "learn to learn" from limited tasks and generalize to unseen tasks. In this work, we propose link-context learning (LCL), which emphasizes "reasoning from cause and effect" to augment the learning capabilities of MLLMs. LCL goes beyond traditional ICL by explicitly strengthening the causal relationship between the support set and the query set. By providing demonstrations with causal links, LCL guides the model to discern not only the analogy but also the underlying causal associations between data points, which empowers MLLMs to recognize unseen images and understand novel concepts more effectively. To facilitate the evaluation of this novel approach, we introduce the ISEKAI dataset, comprising exclusively of unseen generated image-label pairs designed for link-context learning. Extensive experiments show that our LCL-MLLM exhibits strong link-context learning capabilities to novel concepts over vanilla MLLMs.*

## Updates
- **17 Aug, 2023**: :boom::boom: We release the two subsets of ISEKAI (ISEKAI-10 and ISEKAI-pair) at [[HuggingFace 🤗]](https://huggingface.co/ISEKAI-Portal).

  
## Todo

1. [x] Release the [ISEKAI-10](https://huggingface.co/datasets/ISEKAI-Portal/ISEKAI-10) and [ISEKAI-pair](https://huggingface.co/datasets/ISEKAI-Portal/ISEKAI-pair).
2. [ ] Release the dataset usage.
3. [ ] Release the demo.
4. [ ] Release the codes and checkpoints.
5. [ ] Release the full ISEKAI dataset.

