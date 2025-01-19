
import torch
import torch_npu
import argparse
import contexttimer
from colorama import Fore, Style
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
from sampling import autoregressive_sampling, speculative_sampling, speculative_sampling_v2
from globals import Decoder
import time
from tqdm import tqdm
import pickle

# my local models
MODELZOO = {
    # llama-1
    # https://huggingface.co/PY007/TinyLlama-1.1B-step-50K-105b
    "llama1b": "/share_nfs/fangjiarui/root/code/hf_models/TinyLlama-1.1B-step-50K-105b",
    "llama7b": "/share_nfs/tianzhi/code/llama-7b",
    "llama30b": "/share_nfs/fangjiarui/root/code/hf_models/llama-30b-hf",
    "llama2-7b" : "/share_nfs/fangjiarui/root/code/hf_models/llama-2-7b-hf",
    "llama2-70b" : "/share_nfs/fangjiarui/root/code/hf_models/llama-2-70b-hf",
    "bloom-560m": "/share_nfs/fangjiarui/root/code/hf_models/bloom-560m",
    "bloom7b": "/share_nfs/fangjiarui/root/code/hf_models/bloomz-7b1",
    "baichuan-7b": "/share_nfs/duanqiyuan/models/source_models/hf/baichuan-7B",
    "baichuan-13b": "/share_nfs/duanqiyuan/models/source_models/hf/Baichuan-13B-Base",
    "qwen-1b5":'/root/qwen1b5',
    'qwen-0b5':'/root/qwen0b5',
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')

    parser.add_argument('--input', type=str, default="Any recommendations for my holidays in Abu Dhabi?")
    parser.add_argument('--approx_model_name', type=str, default=MODELZOO["qwen-0b5"])
    parser.add_argument('--target_model_name', type=str, default=MODELZOO["qwen-1b5"])
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='enable verbose mode')
    parser.add_argument('--seed', '-s', type=int, default=None, help='set a random seed, which can makes the result reproducible')
    parser.add_argument('--benchmark', '-b', action='store_true', default=False, help='show benchmark results.')
    parser.add_argument('--profiling', '-p', action='store_true', default=False, help='collect torch profiler results.')
    parser.add_argument('--max_tokens', '-M', type=int, default=128, help='max token number generated.')
    parser.add_argument('--gamma', '-g', type=int, default=4, help='guess time.')
    args = parser.parse_args()
    return args


def color_print(text):
    print(Fore.RED + text + Style.RESET_ALL)
    
def benchmark(fn, print_prefix, use_profiler=True, *args, **kwargs):
    TEST_TIME = 10
    profile_filename = f"./profile_logs/{print_prefix}"
    
    with contexttimer.Timer() as t:
        if use_profiler:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=0, warmup=1, active=2, repeat=1, skip_first=0),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_filename),
                record_shapes=False,
                profile_memory=False,
                # with_stack=True
            ) as prof:
                for _ in range(TEST_TIME): 
                    output = fn(*args, **kwargs)
                    prof.step()
        else:
            for _ in range(TEST_TIME): 
                output = fn(*args, **kwargs)

    print(f"\n [benchmark] {print_prefix}, tokens/sec: {len(output[0]) / t.elapsed / TEST_TIME}, {t.elapsed / TEST_TIME} sec generates {len(output[0])} tokens")

def generate(input_text, approx_model_name, target_model_name, num_tokens=20, gamma = 4,
             random_seed = None, verbose = False, use_benchmark = False, use_profiling = False):
    # NOTE() approx_model_name and target_model_name should use the same tokenizer!
    
    torch_device = 'npu' 
    
    tokenizer = AutoTokenizer.from_pretrained(approx_model_name, trust_remote_code=True)
  
    Decoder().set_tokenizer(tokenizer)
    
    print(f"begin loading models: \n {approx_model_name} \n {target_model_name}")
    small_model = AutoModelForCausalLM.from_pretrained(approx_model_name, 
                                                       torch_dtype=torch.float16,
                                                       device_map="auto",
                                                       trust_remote_code=True)
    large_model = AutoModelForCausalLM.from_pretrained(target_model_name, 
                                                       torch_dtype=torch.float16,
                                                       device_map="auto",
                                                       trust_remote_code=True)
    print("finish loading models")
    
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(torch_device)

    top_k = 1
    top_p = 0.001

    torch.manual_seed(123)
    start_large=time.time()
    output = autoregressive_sampling(input_ids, large_model, num_tokens, top_k = top_k, top_p=top_p)
    large_duration=time.time()-start_large
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    color_print(f"large (target) model autoregressive_sampling: {generated_text}")
    print(f'large_duration:{large_duration}')
    
    if use_benchmark:
        benchmark(autoregressive_sampling, "AS_large", use_profiling,
                  input_ids, large_model, num_tokens, top_k = top_k, top_p=top_p)

    #torch.manual_seed(123)
    #output = autoregressive_sampling(input_ids, small_model, num_tokens, top_k = top_k, top_p=top_p)
    #generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    #color_print(f"small (approx) model autoregressive_sampling: {generated_text}")
    
    if use_benchmark:
        benchmark(autoregressive_sampling, "AS_small", use_profiling,
                  input_ids, small_model, num_tokens, top_k = top_k, top_p=top_p)
    
    #torch.manual_seed(123)
    #output = speculative_sampling_v2(input_ids, small_model, large_model, num_tokens, top_k = top_k, top_p=top_p, random_seed = random_seed)  
    #generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    #color_print(f"deepmind's speculative_sampling: {generated_text}")   

    torch.manual_seed(123)
    spec_start=time.time()
    output,generated_tokens,accepted_count,target_sample_count,resample_count = speculative_sampling(input_ids, small_model, large_model, num_tokens, gamma = gamma, top_k = top_k, top_p=top_p, random_seed = random_seed, verbose = verbose)
    spec_duration=time.time()-spec_start
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    color_print(f"google's speculative_sampling: {generated_text}")
    print(f"generated_tokens:{generated_tokens},accepted_count:{accepted_count},target_sample_count:{target_sample_count},resample_count:{resample_count}")
    print(f'spec_duration:{spec_duration}')
    if use_benchmark:
        benchmark(speculative_sampling, "SP", use_profiling,
                  input_ids, small_model, large_model, max_len = num_tokens, gamma = gamma, top_k = top_k, top_p=top_p, random_seed = random_seed)


def generate4hmev(approx_model_name, target_model_name, num_tokens=120, gamma = 4,
             random_seed = None, verbose = False, use_benchmark = False, use_profiling = False,
             problems_path="/root/humaneval_final/human_eval_problems2.jsonl",output_dir="./"):
    # NOTE() approx_model_name and target_model_name should use the same tokenizer!
    
    torch_device = 'npu' 
    
    tokenizer = AutoTokenizer.from_pretrained(approx_model_name, trust_remote_code=True)
  
    Decoder().set_tokenizer(tokenizer)
    
    print(f"begin loading models: \n {approx_model_name} \n {target_model_name}")
    small_model = AutoModelForCausalLM.from_pretrained(approx_model_name, 
                                                       torch_dtype=torch.float16,
                                                       device_map="auto",
                                                       trust_remote_code=True)
    large_model = AutoModelForCausalLM.from_pretrained(target_model_name, 
                                                       torch_dtype=torch.float16,
                                                       device_map="auto",
                                                       trust_remote_code=True)
    print("finish loading models")
    

    with open(problems_path,'r') as f:
        problems=[json.loads(line) for line in f]
    outputs=[]
    for problem in tqdm(problems,desc="humaneval"):
        input_text=problem['prompt']
        task_id=problem['task_id']
        large_=""
        large_duration=0
        spec_=""
        spec_duration=0

        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(torch_device)

        top_k = 1  ######
        top_p = 0.001 ######


        torch.manual_seed(123)
        large_start=time.time()
        output = autoregressive_sampling(input_ids, large_model, num_tokens, top_k = top_k, top_p=top_p)
        large_duration=time.time()-large_start
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        large_=generated_text
        
        try:
            torch.manual_seed(123)
            spec_start=time.time()
            #output = speculative_sampling_v2(input_ids, small_model, large_model, num_tokens, top_k = top_k, top_p=top_p, random_seed = random_seed)
            output,generated_tokens,accepted_count,target_sample_count,resample_count = speculative_sampling(input_ids, small_model, large_model, num_tokens, gamma = gamma, top_k = top_k, top_p=top_p, random_seed = random_seed, verbose = verbose)
            spec_duration=time.time()-spec_start
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            spec_=generated_text
        except Exception as e:
            logger.info(f"error:{e},{task_id}")
        outputs.append({'task_id':task_id,'large_completion':large_,'large_duration':large_duration,'spec_completion':spec_,'spec_duration':spec_duration,'num_tokens':num_tokens,'accepted_count':accepted_count,'target_sample_count':target_sample_count,'resample_count':resample_count})
    dir_=os.path.join(output_dir,'naive_speculative.jsonl')
    with open(dir_, 'w') as file:
        for item in outputs:
            file.write(json.dumps(item) + '\n')



if __name__ == "__main__":
    args = parse_arguments()
    input="from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\""
    generate(input, args.approx_model_name, args.target_model_name, num_tokens=args.max_tokens, gamma=args.gamma,
             random_seed = args.seed, verbose=args.verbose, use_benchmark = args.benchmark) 
    #generate4hmev(args.approx_model_name, args.target_model_name, num_tokens=args.max_tokens, gamma=args.gamma,
    #         random_seed = args.seed, verbose=args.verbose, use_benchmark = args.benchmark)
# nohup python3 naive_spec_hmev.py --approx_model_name /root/llama160m --target_model_name /root/llama7b --max_tokens 200 -b > nspec_topk1.txt &
# modelscope download --model LLM-Research/Llama-3.2-1B-Instruct --local_dir /root/llama1b