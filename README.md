本代码仓记录有关speculative decoding的期末大作业.具体所作的工作在报告中有所展示，比较了CS-Drafting和LLMSpeculativeSampling两种方法在代码生成任务上的性能。所参考的论文和代码仓已清晰列出。
credit：https://github.com/lfsszd/CS-Drafting https://github.com/feifeibear/LLMSpeculativeSampling

```
.
├── CS-Drafting
│   ├── bigram_models
│   │   ├── wiki_bigram_naive_bayers_greedy_llama_next_token.json
│   │   └── wiki_bigram_naive_bayers_greedy_next_token.json
│   ├── cache
│   │   ├── FLAN_T5_xxl_cache.json
│   │   └── LLAMA_7B_cache.json
│   ├── csd_datasets.py
│   ├── csd.py
│   ├── inference.py
│   ├── kernel_meta
│   │   └── kernel_meta_temp_9876028316748965906
│   │       ├── 27737_28229_bt.log
│   │       ├── 27737_28230_bt.log
│   │       ├── 27737_28231_bt.log
│   │       ├── 27737_28232_bt.log
│   │       ├── 27737_28233_bt.log
│   │       ├── 27737_28234_bt.log
│   │       ├── 27737_28235_bt.log
│   │       └── 27737_28236_bt.log
│   ├── mag.py
│   ├── main.py
│   ├── model.py
│   ├── README.md
│   └── requirements.txt
├── CSD_self
│   ├── example.py
│   ├── get_response_csd.py
│   ├── llama13b
│   │   ├── cs_drafting.jsonl
│   │   ├── cs_drafting_metrics.jsonl
│   │   ├── cs_drafting.txt
│   │   └── output_csd_13b.txt
│   └── llama7b
│       ├── cs_drafting.jsonl
│       ├── cs_drafting_metrics.jsonl
│       ├── cs_drafting.txt
│       └── output_csd.txt
├── humaneval_final
│   ├── human_eval_evaluation.py
│   └── human_eval_problems2.jsonl
├── LLMSpeculativeSampling
│   ├── benchmark.py
│   ├── globals.py
│   ├── imgs
│   │   └── sps.jpg
│   ├── LICENSE.txt
│   ├── main.py
│   ├── README.md
│   ├── requirements.txt
│   ├── sampling
│   │   ├── autoregressive_sampling.py
│   │   ├── __init__.py
│   │   ├── kvcache_model.py
│   │   ├── speculative_sampling.py
│   │   └── utils.py
│   └── serving.py
├── README.md
├── report_1_2.pdf
└── Spec_self
    ├── llama13b_7b
    │   ├── naive_speculative.jsonl
    │   ├── naive_speculative_large.jsonl
    │   ├── naive_speculative_results_large_metrics.jsonl
    │   ├── naive_speculative_results_large_metrics.txt
    │   ├── naive_speculative_results_spec_metrics.jsonl
    │   ├── naive_speculative_results_spec_metrics.txt
    │   ├── naive_speculative_spec.jsonl
    │   └── nspec_llama13b_greedy.txt
    ├── llama7b_160m
    │   ├── naive_speculative.jsonl
    │   ├── naive_speculative_large.jsonl
    │   ├── naive_speculative_results_large_metrics.jsonl
    │   ├── naive_speculative_results_large_metrics.txt
    │   ├── naive_speculative_results_spec_metrics.jsonl
    │   ├── naive_speculative_results_spec_metrics.txt
    │   ├── naive_speculative_spec.jsonl
    │   └── nspec_topk1.txt
    └── naive_spec_hmev.py
```
