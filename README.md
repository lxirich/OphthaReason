
# Bridging the Gap in Ophthalmic AI: MM-Retinal-Reason Dataset and OphthaReason Model toward Dynamic Multimodal Reasoning

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2503.13939-b31b1b.svg)](https://arxiv.org/abs/2508.16129)
[![Model](https://img.shields.io/badge/Model-OphthaReason-blue?logo=huggingface)](https://huggingface.co/lxirich/OphthaReason)

</div>

## 🔥 Overview

We introduce MM-Retinal-Reason, the first ophthalmic multimodal dataset with the full spectrum of perception and reasoning. It encompasses both basic reasoning tasks and complex reasoning tasks, aiming to enhance visual-centric fundamental reasoning capabilities and emulate realistic clinical thinking patterns. Building upon MM-Retinal-Reason, we propose OphthaReason, the first ophthalmology-specific multimodal reasoning model with step-by-step reasoning traces. To enable flexible adaptation to both basic and complex reasoning tasks, we specifically design a novel method called Uncertainty-Aware Dynamic Thinking (UADT), which estimates sample-level uncertainty via entropy and dynamically modulates the model’s exploration depth using a shaped advantage mechanism. 

<img src=./assets/model.png width="100%">
<img src=./assets/case.png width="1000%">

## 🚀 Updates
- **[2025.8.22]** We release our initial [ArXiv paper](https://arxiv.org/abs/2508.16129).
- **[2025.8.27]** We release [OphthaReason](https://huggingface.co/lxirich/OphthaReason) to Huggingface and codes for evaluation.

## 🌴 Model Inference
### 1. Setup
```bash
conda create -n OphthaReason_eval python=3.10
conda activate OphthaReason_eval

git clone https://github.com/lxirich/OphthaReason.git
cd OphthaReason
pip install -r requirements_eval.txt
```
### 2. Evaluation  
* Please change the paths `BASE64_ROOT` `DS_ROOT` `OUTPUT_DIR` in `eval.py` and the model path `path/to/the/model/` in `eval.sh` to your own.
```
bash eval.sh 
```






