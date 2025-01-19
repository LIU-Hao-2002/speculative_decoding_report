import time
import torch
import torch_npu
from csd import csd
from transformers import AutoTokenizer, AutoModelForCausalLM
from model import CSDraftingDecoderModel, get_mag_model
import time
from tqdm import tqdm
from loguru import logger
import pickle
device = 'npu'
draft_list = []
draft_names = ['/root/llama160m']
for draft_name in draft_names:
    hf_model = AutoModelForCausalLM.from_pretrained(draft_name)
    model = CSDraftingDecoderModel(hf_model, name=draft_name)
    model.to(device)
    draft_list.append(model)

_BIGRAM_DIR = './bigram_models/'
bi_gram_path = _BIGRAM_DIR + 'wiki_bigram_naive_bayers_greedy_llama_next_token.json'
mag_model = get_mag_model(bi_gram_path, True)
mag_model.to(device)
draft_list.append(mag_model)


LLAMA_PATH = '/root/llama7b/'

k_matrix = torch.tensor([[5, 10], [0, 10]])
LLAMA_HF_PATH = LLAMA_PATH + 'hf_7b_chat'
from transformers import LlamaForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained(LLAMA_PATH)
hf_model = LlamaForCausalLM.from_pretrained(LLAMA_PATH)

target_model = CSDraftingDecoderModel(hf_model, name='llama', vocab_size=32000)
target_model.to(device)

def test(question = 'from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"'):
    
    initial_input = tokenizer(question, truncation=True, padding=False, return_tensors="pt")['input_ids'].to(target_model.device)
    input_ids = initial_input
    start_time=time.time()
    res = csd(draft_list, target_model, initial_input, input_ids, k_matrix, max_length=200) ########
    duration=time.time()-start_time
    num_tokens=res.size()[1]-input_ids.size()[1]
    with open('./res.pkl','wb') as f:
        pickle.dump(res,f)
    generated_text = tokenizer.batch_decode(res, skip_special_tokens=True)
    print(generated_text)
    return generated_text[0],duration,num_tokens,num_tokens/duration

def human_eval(problems_path="/root/humaneval_final/human_eval_problems2.jsonl",output_dir="./"):
    import json
    import os
    with open(problems_path,'r') as f:
        problems=[json.loads(line) for line in f]
    outputs=[]
    speeds=[]
    for problem in tqdm(problems,desc="humaneval"):
        question=problem['prompt']
        task_id=problem['task_id']
        generated_text=""
        duration=""
        speed=None
        num_tokens=None
        try:
            generated_text,duration,num_tokens,speed=test(question)
        except Exception as e:
            logger.info(f"error:{e},{task_id}")
        outputs.append({'task_id':task_id,'completion':generated_text,'duration':duration,'num_tokens':num_tokens})
        if speed!=None:
            speeds.append(speed)
    dir_=os.path.join(output_dir,'cs_drafting.jsonl')
    with open(dir_,'w') as f:
        for item in outputs:
            f.write(json.dumps(item)+'\n') # 写jsonl
    with open(dir_.replace('.jsonl','.txt'),'w') as f:
        f.write(f"speed:{sum(speeds)/len(speeds)}"+'\n')

if __name__=='__main__':
    human_eval()
    #test()
# modelscope download --model shakechen/Llama-2-7b-chat-hf model-00002-of-00002.safetensors --local_dir /root/llama7b  csd 34G；naive 20G
# nohup python3 /root/CS-Drafting/get_response_csd.py > output_csd.txt &