## Description
This is a demo training and evaluation repo for SFT Qwen2-Audio model for Fluent Speech Commands dataset

## Evaluation result for SFT with checkpoint-1200
| Dataset | Accuracy | WER | Keyword Accuracy| # sample | Output File |
|----------|-----------|------|------|----------|----------------|
| **Valid Set** | 0.9849 | 0.0062 | 0.9878 | 3118 | `eval_results/checkpoint-1200/valid_results.jsonl` |
| **Test Set**  | 0.9963 | 0.0018 | 0.9968 | 3793 | `eval_results/checkpoint-1200/test_results.jsonl` |
