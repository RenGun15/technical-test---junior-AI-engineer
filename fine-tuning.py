import os
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from 1_preprocessing import prepare_data  # Impor fungsi preprocessing

def run_finetuning():
    max_seq_length = 2048
    dataset_path = "./data/cleaned_text.jsonl"
    output_dir = "./outputs"

    # 1. Load Model & Tokenizer dengan LoRA
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit",
        max_seq_length = max_seq_length,
        load_in_4bit = True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    # 2. Prepare Dataset
    dataset = prepare_data(dataset_path)

    # 3. Setup Trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Bisa diset True jika sequence pendek
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 60,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = output_dir,
        ),
    )

    # 4. Mulai Fine-Tuning
    print("🚀 Memulai proses Fine-Tuning...")
    trainer.train()

    # 5. Simpan Model / Adapter LoRA
    save_path = "./lora_model"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"🎉 Fine-tuning selesai! Model disimpan di '{save_path}'")

if __name__ == "__main__":
    run_finetuning()