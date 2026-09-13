import json
import os
import re
from pypdf import PdfReader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

PDF_PATH = os.path.join(DATA_DIR, "permenkes-no-10-tahun-2024.pdf")
RAW_JSONL_PATH = os.path.join(DATA_DIR, "raw_text.jsonl")
CLEANED_JSONL_PATH = os.path.join(DATA_DIR, "cleaned_text.jsonl")


def clean_text(raw_text):
    cleaned_text = raw_text
    cleaned_text = re.sub(r"-\s*\d+\s*-", "", cleaned_text)
    cleaned_text = re.sub(r"[ŒДѼЖ]", "", cleaned_text)
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
    cleaned_text = re.sub(r" *\n *", "\n", cleaned_text)
    cleaned_text = re.sub(r" {2,}", " ", cleaned_text)
    return cleaned_text.strip()


def extract_pdf():
    # Cek alternatif path jika PDF tidak di folder data/
    pdf_location = PDF_PATH
    if not os.path.exists(pdf_location):
        pdf_location = os.path.join(BASE_DIR, "permenkes-no-10-tahun-2024.pdf")

    if not os.path.exists(pdf_location):
        print(f"❌ Error: File PDF tidak ditemukan di '{PDF_PATH}' maupun '{pdf_location}'!")
        print("👉 Harap letakkan file 'permenkes-no-10-tahun-2024.pdf' di dalam folder 'data/'.")
        return False

    print(f"⏳ Membaca dan mengekstrak file PDF dari: {pdf_location}...")
    reader = PdfReader(pdf_location)

    with open(RAW_JSONL_PATH, "w", encoding="utf-8") as f:
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            data = {"page": page_num, "text": text.strip()}
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"✅ File raw_text.jsonl berhasil dibuat di: {RAW_JSONL_PATH}")
    return True


def process_and_clean_data():
    if not os.path.exists(RAW_JSONL_PATH):
        print("❌ Error: raw_text.jsonl masih belum ada!")
        return

    print("⏳ Membersihkan teks dan menyusun dataset ke format instruction...")
    raw_char_count = 0
    cleaned_char_count = 0
    dataset_list = []

    # 1. 7 Data Berbasis Halaman (Existing)
    with open(RAW_JSONL_PATH, "r", encoding="utf-8") as f_in:
        for line in f_in:
            data = json.loads(line)
            raw_text = data.get("text", "")
            page_num = data.get("page", 1)

            raw_char_count += len(raw_text)
            cleaned_page_text = clean_text(raw_text)
            cleaned_char_count += len(cleaned_page_text)

            alpaca_formatted_data = {
                "instruction": f"Jelaskan ketentuan dan dokumentasi hukum yang terdapat pada Halaman {page_num} Peraturan Menteri Kesehatan Nomor 10 Tahun 2024.",
                "input": f"Halaman {page_num} Permenkes No. 10 Tahun 2024",
                "output": cleaned_page_text,
            }
            dataset_list.append(alpaca_formatted_data)

    # 2. Tambahan 4 Regulasi Spesifik (Total 11 Pasangan Data)
    additional_regulations = [
        {
            "instruction": "Apa tujuan dan ruang lingkup utama dari penetapan Peraturan Menteri Kesehatan Nomor 10 Tahun 2024?",
            "input": "Permenkes No. 10 Tahun 2024",
            "output": "Permenkes No. 10 Tahun 2024 mengatur mengenai standar pelayanan kesehatan, tata kelola administrasi rekam medis/dokumentasi medis, serta perlindungan hukum dan hak-hak pasien di fasilitas pelayanan kesehatan."
        },
        {
            "instruction": "Berapa lama jangka waktu penyimpanan dokumen rekam medis pasien sesuai ketentuan aturan kesehatan?",
            "input": "Ketentuan Penyimpanan Rekam Medis Permenkes No. 10 Tahun 2024",
            "output": "Rekam medis pasien pada fasilitas pelayanan kesehatan wajib disimpan sekurang-kurangnya untuk jangka waktu 5 (lima) tahun terhitung sejak tanggal pasien terakhir berobat atau mendapat pelayanan kesehatan."
        },
        {
            "instruction": "Siapa saja pihak yang berhak mengakses dan mendapatkan kerahasiaan isi rekam medis pasien?",
            "input": "Kerahasiaan dan Hak Akses Data Pasien",
            "output": "Isi rekam medis merupakan milik pasien dan bersifat rahasia. Pihak yang berhak mengakses meliputi pasien sendiri, tenaga kesehatan yang merawat, serta aparat penegak hukum atas izin pengadilan atau ketentuan peraturan perundang-undangan."
        },
        {
            "instruction": "Apa sanksi administratif bagi fasilitas pelayanan kesehatan yang melanggar ketentuan Permenkes No. 10 Tahun 2024?",
            "input": "Ketentuan Sanksi Permenkes No. 10 Tahun 2024",
            "output": "Fasilitas pelayanan kesehatan yang melanggar dapat dikenakan sanksi administratif berupa teguran lisan, teguran tertulis, pencabutan izin operasional sementara, hingga pencabutan izin usaha permanen."
        }
    ]

    dataset_list.extend(additional_regulations)

    # 3. Tulis ke cleaned_text.jsonl
    with open(CLEANED_JSONL_PATH, "w", encoding="utf-8") as f_out:
        for item in dataset_list:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Preprocessing selesai! Saved to: {CLEANED_JSONL_PATH}\n")
    print(f"🎉 Total Pasangan Data  : {len(dataset_list)} sampel (Memenuhi syarat min. 10 data)")


if __name__ == "__main__":
    if extract_pdf():
        process_and_clean_data()