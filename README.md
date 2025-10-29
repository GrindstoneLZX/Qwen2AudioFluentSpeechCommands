## Description
This is a demo training and evaluation repo for SFT Qwen2-Audio-Instruct model for Fluent Speech Commands dataset

## Evaluation result for validation set 
| Dataset | Accuracy | WER | Keyword Accuracy| # sample | Output File |
|----------|-----------|------|------|----------|----------------|
| **Baseline(No tuning)** | 0.0013 | 4.8829 | 0.1158 | 3118 | `eval_results/baseline/valid_results.jsonl` |
| **SFT(checkpoint-1200)**  | 0.9846 | 0.0068 | 0.9875 | 3118 | `eval_results/checkpoint-1200/valid_results.jsonl` |

## Evaluation result for test set
| Dataset | Accuracy | WER | Keyword Accuracy| # sample | Output File |
|----------|-----------|------|------|----------|----------------|
| **Baseline(No tuning)** | 0.0000 | 4.7211 | 0.1331 | 3793 | `eval_results/baseline/test_results.jsonl` |
| **SFT(checkpoint-1200)**  | 0.9963 | 0.0018 | 0.9968 | 3793 | `eval_results/checkpoint-1200/test_results.jsonl` |

