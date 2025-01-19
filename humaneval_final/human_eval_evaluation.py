import json
import re
import math
import collections
from typing import List, Dict
import time
import threading

class TimeoutException(Exception): pass
#超时判断
def run_with_timeout(func, args=(), kwargs={}, timeout=2):
    result = {'value': None, 'error': None}

    def target():
        try:
            result['value'] = func(*args, **kwargs)
        except Exception as e:
            result['error'] = e

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise TimeoutException('Timeout')
    if result['error'] is not None:
        raise result['error']

    return result['value']


def evaluate_samples(sample_file: str, problem_file: str, result_file: str) -> bool:
    with open(sample_file, 'r',encoding='utf-8') as f:
        samples = [json.loads(line) for line in f]
    with open(problem_file, 'r',encoding='utf-8') as f:
        problems = [json.loads(line) for line in f]

    problems_dict = {problem['task_id']: problem for problem in problems}

    results = []
    all_passed = True
    for sample in samples:
        task_id = sample['task_id']
        completion = sample['completion']# 进一步做清洗 去除首尾的''''''和python等markdown有关的东西
        if "```python\n" in completion:
            completion = completion.split("```python\n")[1].split("```")[0]
        
        problem = problems_dict[task_id]
        tests = problem['test_list']
        if '[DONE]' in completion:
            completion=completion.replace('[DONE]','').strip()

        
        completion=completion.split('def main')[0]
        completion=completion.split('def test')[0]
        completion=completion.split('def Test')[0]
        completion=completion.split('if __name__')[0]
        completion=completion.split('# Test cases')[0].split('# Testing')[0].split('# Test')[0].split('# test')[0].split('# Example')[0].split('# example')[0]
        print('*'*20)
        print(task_id)
        print(completion)
        test_imports = problem.get('test_setup_code', [])
        if test_imports == None:
            test_imports = []
        passed = True
        error_message = ''
        globals_dict = {}
        for import_statement in test_imports:
            try:
                exec(import_statement, globals_dict)
            except Exception as e:
                passed = False
                error_message = f'Error in import: {str(e)}'
                all_passed = False
                break

        if not passed:
            continue

        for i, test in enumerate(tests):
            try:
                print(f'Starting test {i+1}')
                run_with_timeout(exec, (completion, globals_dict), timeout=2)
                run_with_timeout(exec, (test, globals_dict), timeout=2)
                print(f'Finished test {i+1}')
            except TimeoutException:
                passed = False
                error_message = f'Test case {i+1} timed out'
                all_passed = False
                break
            except Exception as e:
                passed = False
                error_message = f'Error in test case {i+1}: {str(e)}'
                all_passed = False
                break
        
        result = {
            'task_id': task_id,
            'passed': passed,
            'error_message': error_message
        }
        results.append(result)
        print(task_id)
        #time.sleep(1)

    with open(result_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    return all_passed

import json

def calculate_passed_ratio(result_file: str) -> float:
    with open(result_file, 'r') as f:
        lines = f.readlines()

    total = len(lines)
    passed_count = 0

    for line in lines:
        result = json.loads(line)
        if result['passed']:
            passed_count += 1

    return passed_count / total if total > 0 else 0

def test_large(dir_='naive_speculative.jsonl',label="large"):
    with open(dir_,'r') as f:
        completions=[json.loads(line ) for line in f ]
    res=[]
    speeds=[]
    accepted_counts=[]
    target_sample_counts=[]
    resample_counts=[]
    for completion in completions:
        if completion[f'{label}_completion']!="":
            res.append({"task_id":completion['task_id'],"completion":completion[f'{label}_completion'],"duration":completion[f'{label}_duration']})
            speeds.append(completion['num_tokens']/completion[f'{label}_duration'])
            accepted_counts.append(completion['accepted_count'])
            target_sample_counts.append(completion['target_sample_count'])
            resample_counts.append(completion['resample_count'])
    mean_speed=sum(speeds)/len(speeds)
    mean_accepted_counts=sum(accepted_counts)/len(accepted_counts)
    mean_target_sample_counts=sum(target_sample_counts)/len(target_sample_counts)
    mean_resample_counts=sum(resample_counts)/len(resample_counts)
    with open(dir_.replace('.jsonl',f'_{label}.jsonl'),'w') as f:
        for item in res:
            f.write(json.dumps(item)+'\n')
    return mean_speed,mean_accepted_counts,mean_target_sample_counts,mean_resample_counts
        

def main_spec(label='large'):
    mean_speed,mean_accepted_counts,mean_target_sample_counts,mean_resample_counts=test_large(label=label)
    #大模型回答文件、源文件、要写入的结果文件
    outcome_file=f'/root/LLMSpeculativeSampling/naive_speculative_results_{label}_metrics.jsonl'
    input=f"/root/LLMSpeculativeSampling/naive_speculative_{label}.jsonl"
    evaluate_samples(input, '/root/humaneval_final/human_eval_problems2.jsonl', outcome_file)
    # 调用函数，计算并输出Pass@1
    pass_=calculate_passed_ratio(outcome_file)
    print("Pass@1:",pass_)
    with open(outcome_file.replace('.jsonl','.txt'),'w') as f:
        f.write(f"Pass@1:{pass_}",)
        for name,item in zip(['mean_speed','mean_accepted_counts','mean_target_sample_counts','mean_resample_counts'],[mean_speed,mean_accepted_counts,mean_target_sample_counts,mean_resample_counts]):
            f.write(f"{name}:{item}"+'\n')

def main_csd():
    outcome_file='/root/CS-Drafting/cs_drafting_metrics.jsonl'
    input_file='/root/CS-Drafting/cs_drafting.jsonl'
    evaluate_samples(input_file, 'human_eval_problems2.jsonl', outcome_file)
    pass_=calculate_passed_ratio(outcome_file)
    print("Pass@1:",pass_)
    with open(input_file.replace('.jsonl','.txt'),'a') as f: ##
        f.write('pass@1:'+str(pass_)+'\n')


#main_csd()
main_spec('spec')

