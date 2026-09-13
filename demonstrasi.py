import os
import random
import json
import torch
import gradio as gr
from unsloth import FastLanguageModel

# 1. Path Lokal & Konfigurasi Model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_PATH = os.path.join(DATA_DIR, "lora_model_1b")
DATASET_PATH = os.path.join(DATA_DIR, "cleaned_text.jsonl")

# Fallback path jika tidak berada di folder data
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "lora_model_1b")
if not os.path.exists(DATASET_PATH):
    DATASET_PATH = os.path.join(BASE_DIR, "cleaned_text.jsonl")

MAX_SEQ_LENGTH = 2048

# 2. Validasi & Memuat Model Unsloth LoRA
if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: Folder model '{MODEL_PATH}' tidak ditemukan!")
    print("👉 Pastikan Anda telah menjalankan '2_finetuning.py' hingga selesai.")
    exit(1)

print(f"⏳ Memuat model dari '{MODEL_PATH}'...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

# Set mode inferensi (mempercepat generasi & hemat memory GPU)
FastLanguageModel.for_inference(model)

# 3. Template Prompt Alpaca (Sama persis dengan Colab/Training)
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""

# 4. Fungsi Ambil Sampel Acak Dataset
def load_random_sample():
    if os.path.exists(DATASET_PATH):
        try:
            with open(DATASET_PATH, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
                if lines:
                    sample = json.loads(random.choice(lines))
                    return (
                        sample.get("instruction", ""),
                        sample.get("input", "")
                    )
        except Exception as e:
            print(f"⚠️ Gagal membaca sampel dataset: {e}")
    return (
        "Jelaskan atau jawab pertanyaan berikut berdasarkan dokumen yang diberikan:",
        ""
    )

# 5. Fungsi Generasi Jawaban (Diadaptasi dari versi Colab)
def generate_response(instruction, input_text):
    inst = instruction.strip()
    inp = input_text.strip()

    if not inst:
        return "⚠️ Instruksi/Pertanyaan tidak boleh kosong!"

    try:
        # Format prompt sesuai input pengguna
        prompt = ALPACA_PROMPT.format(inst, inp, "")

        # Tokenisasi input dan kirim ke GPU
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        # Evaluasi tanpa simpan gradien
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                use_cache=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Ambil hanya token baru hasil generasi (bukan prompt asli)
        input_length = inputs.input_ids.shape[-1]
        generated_tokens = outputs[0][input_length:]
        output_text = tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        ).strip()

        return output_text

    except Exception as e:
        return f"❌ Terjadi kesalahan: {e}"

# 6. Antarmuka UI Gradio
default_inst, default_inp = load_random_sample()

with gr.Blocks(title="Demo Model Permenkes No. 10 Tahun 2024") as demo:
    gr.Markdown("## 🤖 Demo Inferensi Fine-Tuned Llama-3.2 1B")
    gr.Markdown("Aplikasi inferensi lokal menggunakan adapter LoRA hasil pelatihan Unsloth.")

    with gr.Row():
        with gr.Column():
            instruction_widget = gr.Textbox(
                lines=3,
                label="Instruction:",
                placeholder="Masukkan instruksi atau pertanyaan...",
                value=default_inst,
            )
            input_widget = gr.Textbox(
                lines=5,
                label="Input/Konteks:",
                placeholder="Opsional: Tempelkan cuplikan teks/halaman Permenkes di sini jika ada...",
                value=default_inp,
            )
            btn_generate = gr.Button("🚀 Generate Jawaban", variant="primary")
            btn_sample = gr.Button("🎲 Muat Acak Sampel Dataset")

        with gr.Column():
            output_area = gr.Textbox(
                lines=12,
                label="Hasil Jawaban Model",
                interactive=False,
            )

    # Handlers Tombol
    btn_generate.click(
        fn=generate_response,
        inputs=[instruction_widget, input_widget],
        outputs=output_area,
    )

    btn_sample.click(
        fn=load_random_sample,
        inputs=[],
        outputs=[instruction_widget, input_widget],
    )

if __name__ == "__main__":
    demo.launch(share=False)