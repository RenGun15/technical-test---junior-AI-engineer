import torch
import gradio as gr
from unsloth import FastLanguageModel

# 1. Load Fine-Tuned Model & Tokenizer
max_seq_length = 2048
model_path = "./lora_model"  # Path tempat menyimpan adapter lora

print("⏳ Memuat model untuk inference...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# 2. Template Prompt
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""

# 3. Fungsi Inference
def generate_answer(instruction, input_text):
    if not instruction.strip():
        return "Instruksi tidak boleh kosong!"
    
    prompt = alpaca_prompt.format(instruction, input_text, "")
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens = 512,
        use_cache = True,
        temperature = 0.7,
    )
    
    # Decode dan ambil hanya bagian response saja
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    response_clean = response.split("### Response:")[-1].strip()
    return response_clean

# 4. Interface Gradio
demo = gr.Interface(
    fn=generate_answer,
    inputs=[
        gr.Textbox(
            lines=2, 
            label="Instruction", 
            value="Jelaskan atau jawab pertanyaan berikut berdasarkan dokumen yang diberikan:"
        ),
        gr.Textbox(
            lines=4, 
            label="Input/Konteks (Opsional)", 
            placeholder="Tempelkan cuplikan dokumen di sini..."
        )
    ],
    outputs=gr.Textbox(label="Hasil Jawaban Model", lines=8),
    title="🤖 Demo Model Fine-Tuned Permenkes / JDIH",
    description="Masukkan pertanyaan/instruksi dan konteks dokumen untuk melihat jawaban dari model."
)

if __name__ == "__main__":
    demo.launch(share=False)