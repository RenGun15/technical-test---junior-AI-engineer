import os
import json
from datasets import load_dataset
from unsloth import FastLanguageModel

# 1. Konfigurasi awal & Inisialisasi Tokenizer
max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    load_in_4bit = True,
)

# 2. Template Prompt untuk Formatting
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def format_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input_data, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input_data, output) + tokenizer.eos_token
        texts.append(text)
    return { "text" : texts }

# 3. Load & Preprocess Dataset
def prepare_data(data_path):
    print("⏳ Memuat dan memproses dataset...")
    dataset = load_dataset("json", data_files=data_path, split="train")
    dataset = dataset.map(format_prompts_func, batched=True)
    print(f"✅ Preprocessing selesai! Total data: {len(dataset)}")
    return dataset

if __name__ == "__main__":
    # Ganti path ini sesuai lokasi dataset lokal Anda
    data_path = "./data/cleaned_text.jsonl" 
    if os.path.exists(data_path):
        dataset = prepare_data(data_path)
    else:
        print(f"⚠️ File data {data_path} tidak ditemukan. Pastikan path sudah benar.")