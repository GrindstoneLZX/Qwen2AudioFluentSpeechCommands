import re
import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
    BitsAndBytesConfig
)
from peft import LoraConfig, PeftModel
from dataclasses import dataclass
from typing import Dict, List, Union, Any
import librosa
import soundfile as sf
from tqdm import tqdm
import json
from jiwer import wer

def extract_assistant_text(generated_text: str) -> str:
    match = re.search(r"\nassistant\n\s*(.*)", generated_text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    else:
        text = generated_text.strip()

    text = re.sub(r"<\|.*?\|>", "", text)
    return text.strip()


def generate(model, processor, audio_paths, system_prompt):
    audio_num = len(audio_paths)
    audios = []
    for audio_path in audio_paths:
        audio, sr = librosa.load(audio_path)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        audios.append(audio)
        
    system_prompt = "You are a professional assistant for understanding speech commands."
    full_prompts = [(
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n<|AUDIO|><|im_end|>\n"
            f"<|im_start|>assistant\n"
        ) for _ in range(audio_num)]

    full_inputs = processor(
        text=full_prompts,
        audio=audios,
        return_tensors="pt",
        padding="longest",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **full_inputs,
            max_new_tokens=128,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    assistent_text = [extract_assistant_text(text) for text in generated_text]
    return assistent_text


def evaluate(model, processor, dataset, system_prompt=None, batch_size=8, split_name="test"):
    predictions, references = [], []

    for i in tqdm(range(0, len(dataset), batch_size), desc=f"Evaluating {split_name} set"):
        batch = dataset[i : i + batch_size]
        audio_paths = batch["audio"]
        ref_texts = [text.strip() for text in batch["text"]]

        try:
            pred_texts = generate(model, processor, audio_paths, system_prompt=system_prompt)
        except Exception as e:
            print(f"[WARN] Batch failed: {e}")
            pred_texts = [""] * len(audio_paths)

        predictions.extend(pred_texts)
        references.extend(ref_texts)

    exact_match = sum(p.lower() == r.lower() for p, r in zip(predictions, references))
    acc = exact_match / len(predictions)
    wer_score = wer(references, predictions)

    print(f"\n{split_name} set evaluation completed.")
    print(f"Accuracy: {acc:.4f}")
    print(f"WER: {wer_score:.4f}")

    out_file = f"{split_name}_results.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for p, r in zip(predictions, references):
            f.write(json.dumps({"pred": p, "ref": r}, ensure_ascii=False) + "\n")
    print(f"Save output to {out_file}")

    return {"acc": acc, "wer": wer_score}

def main():
    model_name = "Qwen/Qwen2-Audio-7B-Instruct"
    val_path = "/home/lzx/Data/fluent_speech_commands_dataset/processed/valid.jsonl"
    test_path = "/home/lzx/Data/fluent_speech_commands_dataset/processed/test.jsonl"
    model_dir = "lora_model/checkpoint-1200"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                 
        bnb_4bit_compute_dtype=torch.bfloat16,  
        bnb_4bit_quant_type="nf4",         
        bnb_4bit_use_double_quant=True,    
    )

    processor = AutoProcessor.from_pretrained(model_name, sampling_rate=16000)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map="auto"
    )

    test_dataset = load_dataset("json", data_files=test_path)["train"]
    eval_dataset = load_dataset("json", data_files=val_path)["train"]

    model = PeftModel.from_pretrained(model, model_dir)
    model.eval()

    evaluate(model, processor, eval_dataset, batch_size=8, split_name="valid")
    evaluate(model, processor, test_dataset, batch_size=8, split_name="test")


if __name__ == "__main__":
    main()