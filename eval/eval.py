import json
from tqdm import tqdm
import os
import pandas as pd
import pickle
from vllm import LLM, SamplingParams
import ray
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

seed = 3702
def setup_seed(seed):
     import torch
     import numpy as np
     import random

     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(seed)
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


CUDA_DEVICES='0,1,2,3'
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_DEVICES


BASE64_ROOT = "path/to/the/image_base64_csv/"
DS_ROOT = "path/to/the/test_data_JSON/"
MODEL_DIR = os.environ.get("MODEL_DIR", "")
OUTPUT_DIR = "path/to/the/output_folder/"


SIGLE_DS = ['single_choice_test_data_CFP','single_choice_test_data_FFA','single_choice_test_data_OCT']
MULTI_DS = ['multi_choice_test_data_CFP']
DS_DICT = {'complex_reason': 'COMPLEX'}
DS_DICT.update({ds: 'SINGLE' for ds in SIGLE_DS})
DS_DICT.update({ds: 'MULTI' for ds in MULTI_DS})


TEST_DATASETS = ['single_choice_test_data_CFP']

SYSTEM_PROMPT = (
    "You're a professional ophthalmologist."
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

QUESTION_DICT = {
    "MULTI":
        "{Question}. First output the thinking process in English in <think> </think> tags and then output the final answer in English in <answer> </answer> tags. " \
        "The final answer must be uppercase letters separated by commas without spaces (e.g., B,E,F). " \
        "Note that this is a multiple-choice question with potentially multiple correct answers. Carefully analyze all options.",
    "SINGLE":
        "{Question}. First output the thinking process in English in <think> </think> tags and then output the final answer in English in <answer> </answer> tags. " \
        "The final answer must be uppercase letters (e.g., B). " \
        "Note that this is a single-choice question with only one correct answer. Carefully analyze all options.", 
    "COMPLEX":
        "{Question}. First output the thinking process in English in <think> </think> tags. " \
        "Then output the final answer (just the name of the disease/entity) in English in <answer> </answer> tags." ,
    }


datas = {ds: [] for ds in TEST_DATASETS}
for ds in TEST_DATASETS:
    question_key = DS_DICT[ds]
    ds_path = os.path.join(DS_ROOT, f"{ds}.json")

    data = json.load(open(ds_path, "r"))

    base64_path = os.path.join(BASE64_ROOT, f"{ds}.csv")
    base64_dict = pd.read_csv(base64_path, index_col=0).to_dict()["base64"]
    for x in data:
        question = x['conversations'][0]['value']
        answer = x['conversations'][1]['value']
        image_lst = x['image']
        if type(image_lst) != list:
            image_lst = [image_lst]
        base64_image_lst = [base64_dict[img] for img in image_lst ]

        message = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
            "role": "user",
            "content": [{"type": "text", 
                         "text": QUESTION_DICT[question_key].format(Question=question)}] + \
                        [{"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}} 
                        for base64_image in base64_image_lst]
            }]
        datas[ds].append([message, question, answer, x['image']])

print("Data Prepared")


model_name = sys.argv[1]
BSZ = 16
MODEL_PATH = os.path.join(MODEL_DIR, model_name)
model = LLM(model=MODEL_PATH, tensor_parallel_size=len(CUDA_DEVICES.split(',')), 
            gpu_memory_utilization=0.9, seed=seed, limit_mm_per_prompt={"image": 6})
print(f"{model_name} Prepared")

sampling_params = SamplingParams(temperature=0., max_tokens=8192, seed=seed)
outputs = {}
with tqdm(total=sum([len(datas[ds]) for ds in TEST_DATASETS])) as pbar:
    for ds in TEST_DATASETS:
        pkl_path =  os.path.join(OUTPUT_DIR, f"{model_name}_{ds}.pkl")
        if os.path.exists(pkl_path):
            pbar.update(len(datas[ds]))
            continue
        
        pbar.set_description(ds)
        output = []
        ds_messages = datas[ds]

        for i in range(0, len(ds_messages), BSZ):
            batch_messages = [idata[0] for idata in ds_messages[i:i + BSZ]]
            responses = model.chat(batch_messages, sampling_params, use_tqdm=False)
            output.extend([[response.outputs[0].text, idata[1], idata[2], idata[3]] 
                            for response, idata in zip(responses, ds_messages[i:i + BSZ])])
            pbar.update(len(batch_messages))


        with open(pkl_path, 'wb') as f:
            pickle.dump(output, f)

