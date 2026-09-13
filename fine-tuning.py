# 1. Unsloth WAJIB di paling atas
from unsloth import FastLanguageModel

# 2. Transformers & PyTorch
import os
import torch

# 3. TRL & Datasets
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

# Path Relatif Lokal
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_JSONL_PATH = os.path.join(DATA_DIR, "cleaned_text.jsonl")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
SAVE_PATH = os.path.join(DATA_DIR, "lora_model_1b")

# Format Prompt Alpaca
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def format_prompts(examples, tokenizer):
    texts = []

    # Hanya terima format QA yang valid (instruction, input, output)
    if "instruction" in examples and "output" in examples:
        instructions = examples["instruction"]
        inputs = examples.get("input", [""] * len(instructions))
        outputs = examples["output"]

        for inst, inp, out in zip(instructions, inputs, outputs):
            formatted_text = (
                ALPACA_PROMPT.format(inst, inp, out) + tokenizer.eos_token
            )
            texts.append(formatted_text)
    else:
        raise ValueError(
            "❌ Dataset harus memiliki kolom 'instruction' dan 'output'! "
            "Format teks mentah dengan dummy response tidak bisa digunakan untuk Q&A fine-tuning."
        )

    return {"text": texts}


def run_finetuning():
    if not os.path.exists(TRAIN_JSONL_PATH):
        print(f"❌ Error: File '{TRAIN_JSONL_PATH}' tidak ditemukan!")
        return

    max_seq_length = 2048

    print("⏳ Memuat model Unsloth Llama-3.2 1B (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-1B-bnb-4bit",
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    # Konfigurasi LoRA yang lebih ideal untuk model 1B
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    print(f"⏳ Memuat dataset dari: {TRAIN_JSONL_PATH}")
    dataset_train = load_dataset(
        "json", data_files={"train": TRAIN_JSONL_PATH}, split="train"
    )

    dataset_train = dataset_train.map(
        lambda examples: format_prompts(examples, tokenizer), batched=True
    )

    print(f"✅ Total sampel QA berhasil dimuat: {len(dataset_train)}")

    args = SFTConfig(
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=1e-4,  # Diturunkan sedikit agar lebih stabil
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        output_dir=OUTPUT_DIR,
        dataset_num_proc=1,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_train,
        processing_class=tokenizer,
        args=args,
    )

    print("🚀 Memulai proses Fine-Tuning...")
    trainer.train()

    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print(f"🎉 Fine-Tuning selesai! Model LoRA disimpan di: '{SAVE_PATH}'")


if __name__ == "__main__":
    run_finetuning()