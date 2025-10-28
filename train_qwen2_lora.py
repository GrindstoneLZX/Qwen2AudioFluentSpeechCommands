import copy
import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from dataclasses import dataclass
from typing import Dict, List, Union, Any
import librosa
import soundfile as sf

model_name = "Qwen/Qwen2-Audio-7B-Instruct"
data_path = "/home/lzx/Data/fluent_speech_commands_dataset/processed/train.jsonl"
val_path = "/home/lzx/Data/fluent_speech_commands_dataset/processed/valid.jsonl"
output_dir = "lora_model"
logging_dir = "logs"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                 # 开启 4-bit 量化
    bnb_4bit_compute_dtype=torch.bfloat16,  # 用 bf16 进行计算，加速且占显存少
    bnb_4bit_quant_type="nf4",         # 使用 NF4，精度比 FP4 好
    bnb_4bit_use_double_quant=True,    # 双重量化，进一步压缩显存，几乎不影响精度
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

processor = AutoProcessor.from_pretrained(model_name, sampling_rate=16000)
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    quantization_config=bnb_config,
    device_map="auto"
)
model = get_peft_model(model, lora_config)

# === 数据加载 ===
train_dataset = load_dataset("json", data_files=data_path)["train"]
eval_dataset = load_dataset("json", data_files=val_path)["train"]

# === Collator ===
@dataclass
class Qwen2AudioSpeechCommandCollator:
    processor: Any
    padding: Union[bool, str] = "longest"
    label_pad_token_id: int = -100
    system_prompt: str = "You are a professional assistant for understanding speech commands."

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audios = []
        for f in features:
            wav, sr = librosa.load(f["audio"])
            if sr != 16000:
                wav = librosa.resample(y=wav, orig_sr=sr, target_sr=16000)
            audios.append(wav)
        targets = [f["text"].strip() for f in features] 
        
        full_prompts = [
            (
                f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n<|AUDIO|><|im_end|>\n"
                f"<|im_start|>assistant\n{t}<|im_end|>"
            )
            for t in targets
        ]

        full_inputs = self.processor(
            text=full_prompts,
            audio=audios,
            return_tensors="pt",
            padding=self.padding,
        )

        label_ids = torch.full_like(full_inputs["input_ids"], fill_value=self.label_pad_token_id)
        for i in range(len(targets)):
            input_ids_i = full_inputs["input_ids"][i]
            target_ids_i = self.processor.tokenizer(targets[i] + "<|im_end|>", return_tensors="pt", padding=False, add_special_tokens=False)["input_ids"][0]
            start_idx = self.find_subarray_from_last(input_ids_i, target_ids_i)
            label_ids[i, start_idx:start_idx + target_ids_i.size(0)] = target_ids_i

        full_inputs["labels"] = label_ids

        for k, v in full_inputs.items():
            if isinstance(v, torch.Tensor):
                full_inputs[k] = v.contiguous()

        return full_inputs
    
    @staticmethod
    def find_subarray_from_last(array_long: torch.Tensor, array_short: torch.Tensor):
        len_long = array_long.size(0)
        len_short = array_short.size(0)
        
        if len_short > len_long:
            return -1
        
        windows = array_long.unfold(0, len_short, 1)
        mask = (windows == array_short).all(dim=1)
        idxs = torch.nonzero(mask, as_tuple=False)
        if idxs.numel() == 0:
            return -1
        
        return idxs[-1].item()

data_collator = Qwen2AudioSpeechCommandCollator(processor)

# debug data_collator
print("debug data_collator")
sample = [train_dataset[i] for i in range(2)]
batch = data_collator(sample)
print(batch.keys())
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        batch[k] = v.to(model.device)
        print(k, v.shape, v.dtype, v.device)

# print("=== LoRA trainable parameters ===")
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.shape)

# === 训练参数 ===
training_args = TrainingArguments(
    output_dir=output_dir,
    logging_dir=logging_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    weight_decay=0.01,
    num_train_epochs=3,
    bf16=True,
    save_total_limit=5,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    gradient_checkpointing=False,
    report_to="tensorboard",
    remove_unused_columns=False,
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

trainer.train()
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)
print("LoRA fine-tuning finished!")
