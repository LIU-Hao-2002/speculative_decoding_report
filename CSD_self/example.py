import time
import torch
import torch_npu
from csd import csd
from transformers import AutoTokenizer, AutoModelForCausalLM
from model import CSDraftingDecoderModel, get_mag_model
import time
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


question = 'from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"'
start_time=time.time()
initial_input = tokenizer(question, truncation=True, padding=False, return_tensors="pt")['input_ids'].to(target_model.device)
input_ids = initial_input
res = csd(draft_list, target_model, initial_input, input_ids, k_matrix, max_length=1024)
generated_text = tokenizer.batch_decode(res, skip_special_tokens=True)
duration=time.time()-start_time
print(generated_text)
print(type(generated_text))
print(f'duration,{duration}')

# modelscope download --model shakechen/Llama-2-7b-chat-hf model-00002-of-00002.safetensors --local_dir /root/llama7b