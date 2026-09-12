import os
import torch
import gradio as gr
from unsloth import FastLanguageModel

# 1. Path Relatif Lokal & Konfigurasi Model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Mengarahkan langsung ke folder hasil simpan 2_finetuning.py
MODEL_PATH = os.path.join(BASE_DIR, "data", "lora_model_1b")

# Fallback jika model disimpan di root directory
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "lora_model_1b")

MAX_SEQ_LENGTH = 2048

# 2. Validasi & Load Model
if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: Folder model '{MODEL_PATH}' tidak ditemukan!")
    print("👉 Pastikan Anda telah menjalankan '2_finetuning.py' hingga selesai.")
    exit(1)

print(f"⏳ Memuat model dari '{MODEL_PATH}' untuk inferensi...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)

# Aktifkan mode inferensi cepat khas Unsloth (2x lebih cepat & efisien VRAM)
FastLanguageModel.for_inference(model)

# 3. Template Prompt Alpaca Format
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""

# 4. Fungsi Inference
def generate_answer(instruction, input_text):
    if not instruction.strip():
        return "⚠️ Instruksi tidak boleh kosong!"

    prompt = ALPACA_PROMPT.format(instruction, input_text, "")
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        use_cache=True,
        temperature=0.7,
    )

    # Decode output dan ekstrak hanya bagian jawaban setelah '### Response:'
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    response_clean = response.split("### Response:")[-1].strip()
    return response_clean

# 5. UI Gradio Interface
demo = gr.Interface(
    fn=generate_answer,
    inputs=[
        gr.Textbox(
            lines=2,
            label="Instruction",
            value="Jelaskan atau jawab pertanyaan berikut berdasarkan dokumen yang diberikan:",
        ),
        gr.Textbox(
            lines=4,
            label="Input/Konteks (Opsional)",
            placeholder="Tempelkan cuplikan Halaman Permenkes No 10 Tahun 2024 di sini...",
        ),
    ],
    outputs=gr.Textbox(label="Hasil Jawaban Model", lines=8),
    title="🤖 Demo Model Fine-Tuned Permenkes No. 10 Tahun 2024",
    description="Masukkan pertanyaan/instruksi dan konteks dokumen untuk melihat jawaban dari model Llama 3.2 1B.",
)

if __name__ == "__main__":
    # Menjalankan Gradio di server lokal (http://127.0.0.1:7860)
    demo.launch(share=False)