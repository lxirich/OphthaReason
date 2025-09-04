
# Bridging the Gap in Ophthalmic AI: MM-Retinal-Reason Dataset and OphthaReason Model toward Dynamic Multimodal Reasoning

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2508.16129-b31b1b.svg)](https://arxiv.org/abs/2508.16129)
[![Github](https://img.shields.io/badge/Github-OphthaReason-black?logo=Github)](https://github.com/lxirich/OphthaReason)
[![Model](https://img.shields.io/badge/Model-OphthaReason-blue?logo=huggingface)](https://huggingface.co/lxirich/OphthaReason)
[![Dataset](https://img.shields.io/badge/Dataset-MM--Retinal--Reason-blue?logo=huggingface)](https://huggingface.co/datasets/lxirich/MM-Retinal-Reason)

</div>

## 🔥 Overview

We introduce MM-Retinal-Reason, the first ophthalmic multimodal dataset with the full spectrum of perception and reasoning. It encompasses both basic reasoning tasks and complex reasoning tasks, aiming to enhance visual-centric fundamental reasoning capabilities and emulate realistic clinical thinking patterns. Building upon MM-Retinal-Reason, we propose OphthaReason, the first ophthalmology-specific multimodal reasoning model with step-by-step reasoning traces. To enable flexible adaptation to both basic and complex reasoning tasks, we specifically design a novel method called Uncertainty-Aware Dynamic Thinking (UADT), which estimates sample-level uncertainty via entropy and dynamically modulates the model’s exploration depth using a shaped advantage mechanism. 

<img src="./assets/model.png" width="100%">
<img src="./assets/case.png" width="100%">

## 🌈 MM-Retinal-Reason Dataset

***For more dataset details and download links, please refer to [MM-Retinal-Reason](https://huggingface.co/datasets/lxirich/MM-Retinal-Reason).***

### 1. Data Format
The format for the JSON file:
```bash
{
  "image": ["path/to/image.jpg"],
  "conversations": [
    {"from": "human", "value": "user input"},
    {"from": "gpt", "value": "assistant output"},
  ]
  "reason": "reasoning trajectory"
  "pmcid": "PMCXXX",            # only for complex reasoning
  "title": "PMC article title"  # only for complex reasoning
  "caption": "image caption"    # only for complex reasoning
}
```

### 2. Data Source 
We gratefully acknowledge the valuable contributions of all these public datasets. 

| Subset  | Dataset Composition |
| :------ | :------ |
| **CFP** | **In-Domain:** PAPILA, PARAGUAY, ARIA, APTOS, HRF, DeepDRID, G1020, AMD, PALM, ORIGA, Drishti-GS1, CHAKSU, Cataract, FUND-OCT, <br> **Out-of-Domain:** MESSIDOR, IDRID, RFMid, STARE, ROC, Retina, SUSTech-SYSU, JICHI, EYEPACS, LAG, FIVES, E-ophta, REFUGE, DR1-2, ScarDat, ACRIMA, OIA-DDR |
| **FFA** | **In-Domain:** Angiographic <br> **Out-of-Domain:** MPOS |
| **OCT** | **In-Domain:** GOALS, GAMMA1, STAGE1, STAGE2, OIMHS, OCTA_500, Large_Dataset_of_Labeled_OCT, DUKE_DME, glaucoma_detection, RetinalOCT_C8 <br> **Out-of-Domain:** OCTDL, OCTID |
| **Complex** | PubMed Central (up to June 20, 2025) | 


## 🌴 OphthaReason Model
### 1. Pretrain Model Download
The OphthaReason model can be downloaded from [Hugging Face Link](https://huggingface.co/lxirich/OphthaReason).

### 2. Setup
```bash
# Create and activate a new conda environment
conda create -n OphthaReason_eval python=3.10
conda activate OphthaReason_eval

# Clone the repository and install dependencies
git clone https://github.com/lxirich/OphthaReason.git
cd OphthaReason
pip install -r requirements_eval.txt
```
### 3. Batch Evaluation  
1. Update the following paths in `eval.py`:
    - `BASE64_ROOT` Path to your base64 encoded images
    - `DS_ROOT`: Path to your dataset JSON files
    - `OUTPUT_DIR`: Directory for output results
2. Modify the model path in `eval.py` to point to your downloaded model
3. Run the evaluation script:
```bash
bash eval/eval.sh 
```
### 4. Single Instance VQA Inference
For Visual Question Answering with a single instance (which may include multiple images), use the following example:
```python
import base64
from vllm import LLM, SamplingParams

# Load the model
model_path = "path/to/OphthaReason/model"  # Replace with your model path
model = LLM(model=model_path, tensor_parallel_size=1, gpu_memory_utilization=0.8)
sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)

# Prepare instance image input
image_paths = [
    "path/to/retinal/image1.jpg",
    "path/to/retinal/image2.jpg",  # Additional image in the same instance
    # Add more images as needed for this instance
]

# Convert images to base64
image_contents = []
for img_path in image_paths:
    with open(img_path, "rb") as f:
        image_content = base64.b64encode(f.read()).decode('utf-8')
        image_contents.append(image_content)

# Construct prompts
system_prompt = (
    "You're a professional ophthalmologist."
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer..."
)

user_prompt = f"A 62-year-old woman presented with a one-month history of sudden painless visual loss..."

# Build message content with multiple images for this instance
content = [{"type": "text", "text": user_prompt}]
for img_content in image_contents:
    content.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_content}"})

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": system_prompt}]
    },
    {
        "role": "user",
        "content": content
    }
]

# Perform VQA inference on this instance
outputs = model.chat([messages], sampling_params)
result = outputs[0].outputs[0].text

print(result)
```





