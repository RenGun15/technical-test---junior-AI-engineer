import json
import os
import re
from pypdf import PdfReader

# 1. Path Relatif Lokal (Otomatis menyesuaikan lokasi folder di VS Code)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

PDF_PATH = os.path.join(DATA_DIR, "permenkes-no-10-tahun-2024.pdf")
RAW_JSONL_PATH = os.path.join(DATA_DIR, "raw_text.jsonl")
CLEANED_JSONL_PATH = os.path.join(DATA_DIR, "cleaned_text.jsonl")


# 2. Fungsi Pembersihan Teks (Regex)
def clean_text(raw_text):
    cleaned_text = raw_text

    # Menghapus nomor halaman (misal: -1-, -12-)
    cleaned_text = re.sub(r"-\s*\d+\s*-", "", cleaned_text)

    # Menghapus OCR/encoding noise
    cleaned_text = re.sub(r"[ŒДѼЖ]", "", cleaned_text)

    # Menghapus spasi dan tab berlebih
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)

    # Menghapus spasi di awal dan akhir baris
    cleaned_text = re.sub(r" *\n *", "\n", cleaned_text)

    # Menghapus spasi ganda
    cleaned_text = re.sub(r" {2,}", " ", cleaned_text)

    return cleaned_text.strip()


# 3. Ekstraksi PDF ke raw_text.jsonl
def extract_pdf():
    if not os.path.exists(PDF_PATH):
        print(f"⚠️ Error: File '{PDF_PATH}' tidak ditemukan!")
        print("👉 Pastikan Anda sudah menyimpan file PDF di folder 'data/'.")
        return False

    print(f"⏳ Membaca dan mengekstrak file PDF: {PDF_PATH}...")
    reader = PdfReader(PDF_PATH)

    with open(RAW_JSONL_PATH, "w", encoding="utf-8") as f:
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            data = {"page": page_num, "text": text.strip()}
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"✅ Ekstraksi mentah selesai! File tersimpan di: {RAW_JSONL_PATH}")
    return True


# 4. Cleaning & Formatting ke Alpaca JSONL Format
def process_and_clean_data():
    if not os.path.exists(RAW_JSONL_PATH):
        return

    print("⏳ Membersihkan teks dan menyusun dataset ke format instruction...")
    raw_char_count = 0
    cleaned_char_count = 0

    with (
        open(RAW_JSONL_PATH, "r", encoding="utf-8") as f_in,
        open(CLEANED_JSONL_PATH, "w", encoding="utf-8") as f_out,
    ):
        for line in f_in:
            data = json.loads(line)
            raw_text = data.get("text", "")
            page_num = data.get("page", 1)

            raw_char_count += len(raw_text)

            # Bersihkan teks
            cleaned_page_text = clean_text(raw_text)
            cleaned_char_count += len(cleaned_page_text)

            # Format JSONL agar sesuai dengan kebutuhan fine-tuning (Instruction-Input-Output)
            alpaca_formatted_data = {
                "instruction": f"Jelaskan ketentuan dan dokumentasi hukum yang terdapat pada Halaman {page_num} Peraturan Menteri Kesehatan Nomor 10 Tahun 2024.",
                "input": f"Halaman {page_num} Permenkes No. 10 Tahun 2024",
                "output": cleaned_page_text,
            }

            f_out.write(
                json.dumps(alpaca_formatted_data, ensure_ascii=False) + "\n"
            )

    print(f"✅ Preprocessing selesai! Saved to: {CLEANED_JSONL_PATH}\n")
    print("=== Ringkasan Statistik Preprocessing ===")
    print(f"Total Karakter Mentah   : {raw_char_count}")
    print(f"Total Karakter Cleaned  : {cleaned_char_count}")
    print(f"Karakter Terbuang       : {raw_char_count - cleaned_char_count}")


if __name__ == "__main__":
    # Jalankan alur ekstraksi dan pembersihan saja
    if extract_pdf():
        process_and_clean_data()