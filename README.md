# TECHNICAL TEST – JUNIOR AI ENGINEER

## 📌 Deskripsi Program

Program ini dikembangkan sebagai solusi *End-to-End* untuk simulasi **Fine-Tuning Large Language Model (LLM)** menggunakan teknik **PEFT (Parameter-Efficient Fine-Tuning) / QLoRA 4-bit**. Studi kasus yang digunakan berfokus pada pemrosesan dokumen regulasi kesehatan **Peraturan Menteri Kesehatan (Permenkes) Nomor 10 Tahun 2024**.

Sistem ini dirancang agar dapat berjalan secara efisien pada lingkungan dengan VRAM terbatas (seperti GPU NVIDIA Colab T4 / Local GPU) dengan memanfaatkan akselerasi dari **Unsloth**, **BitsAndBytes**, serta antarmuka interaktif berbasis **Gradio**.

## 🎯 Fitur Utama
1. **Automated PDF Preprocessing & Dataset Structuring**:
   Mengekstraksi teks mentah dari PDF regulasi kesehatan, melakukan pembersihan noise/OCR formatting, dan menyusun data ke dalam format *Instruction Tuning* (Alpaca JSONL: `instruction`, `input`, `output`).
2. **Resource-Efficient PEFT/QLoRA Fine-Tuning**:
   Melatih base model **Llama-3.2 1B (4-bit)** menggunakan modul LoRA (`q_proj`, `k_proj`, `v_proj`, `o_proj`) untuk menyesuaikan domain pengetahuan spesifik tentang regulasi kesehatan.
3. **Interactive Inference Web UI (Gradio)**:
   Antarmuka web lokal untuk menguji dan membandingkan hasil respons model *fine-tuned* secara langsung, dilengkapi fitur ekstraksi sampel dataset acak.

# 💻 Persyaratan Sistem & Dependensi
*VSCODE(Lokal)*
- OS: Windows / Linux
- GPU: NVIDIA GPU (Sangat disarankan dengan VRAM minimum 6GB - 8GB)
- Python: Versi 3.10 atau 3.11
- CUDA: Versi 12.1  
*Google Colab*
- GPU T4
- Runtime: Python 3 (GPU T4 Accelerated)
- Akses Google Drive (opsional, untuk penyimpan data/model)

# 🛠️ instalasi
## *Persiapan Eksekusi Program pada VSCODE(Lokal)*
1. Persiapan Virtual Environment  
- *Membuat environment*   
```python -m venv venv```  
- *Mengaktifkan environment (Windows PowerShell)*  
```.\venv\Scripts\activate```    
2. Instalasi Dependensi  
Pasang seluruh paket yang dibutuhkan yang terdaftar pada requirements.txt:   
```pip install -r requirements.txt```  
## *Persiapan Eksekusi Program pada Google Colab*
1. Buka Google Colab dan pastikan Hardware Accelerator diatur ke **GPU T4** (`Runtime` > `Change runtime type` > `T4 GPU`).
2. Jalankan sel pertama untuk membuat file `requirements.txt` dan menginstal seluruh dependensi secara otomatis (`!pip install -r requirements.txt`).


# 🚀 Langkah Penggunaan
## *Alur Eksekusi Program pada VSCODE(lokal)*
Step 1: Preprocessing Data  
Letakkan file PDF permenkes-no-10-tahun-2024.pdf di dalam folder data/, lalu jalankan skrip pembersihan data dengan cara mengetikkan command berikut di terminal  
```python preprocessing.py```  
pada tahap ini proses akan Membaca PDF, menghapus noise/halaman/spasi berlebih, dan mengubah isi dokumen menjadi dataset berformat instruksi Alpaca (instruction, input, output) pada data/cleaned_text.jsonl.  
Step 2: Fine-Tuning Model (LoRA)  
Eksekusi skrip fine-tuning menggunakan Unsloth:  
```python fine-tuning.py```  
pada saat melakukan prose fine-tuning.py maka akan terjadi proses berikut:
- Proses: Memuat base model unsloth/Llama-3.2-1B-bnb-4bit, memasang adapter PEFT LoRA, memproses dataset, dan melatih model selama 60 steps.
- Output: Adapter LoRA yang telah dilatih akan disimpan di folder data/lora_model_1b/.  
Step 3: Demonstrasi & Inferensi (Web UI)  
Jalankan antarmuka Gradio untuk menguji hasil fine-tuning:  
```python demonstrasi.py```  
- Buka tautan lokal yang muncul di terminal (misalnya: http://127.0.0.1:7860) melalui peramban web kamu.  
- Fitur UI mencakup:  
Generate Jawaban: Mengirim instruksi & konteks ke model.  
Muat Acak Sampel Dataset: Mengambil contoh instruksi/input secara otomatis dari dataset cleaned_text.jsonl.  
## *Alur Eksekusi Program pada Google Colab*  
Step 1: pilih opsi runtime menggunakan GPU T4 (apabila runtime menggunakan CPU)  
Step 2: setelah menggunakan GPU T4 sebagai runtime pilih opsi jalankan semua sel

# 📄 Lisensi

Proyek ini dilisensikan di bawah **MIT License**. Lihat file [LICENSE](LICENSE) untuk informasi lebih lanjut.
