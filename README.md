# 🩺 AI Facial Health Screening & Triage Dashboard

Platform skrining kesehatan awal (*triage tool*) berbasis AI dan Computer Vision untuk menganalisis biomarker fisik visual pada wajah serta melakukan **Triangulasi Klinis Cerdas (Visual + Anamnesis Gejala)**.

---

## 🔬 5 Parameter Biomarker yang Dideteksi

| Parameter | Indikator Klinis | Metode AI & Komputasi |
| :--- | :--- | :--- |
| **Simetri Saraf Wajah** | Skrining Gejala Awal Stroke (*F.A.S.T*) & *Bell's Palsy* | 478 Landmark MediaPipe Face Mesh dengan normalisasi proyeksi sumbu wajah (*head-tilt invariant*) |
| **Sklera Mata Menguning** | Indikasi Ikterus / Gangguan Hati & Bilirubin (Hepatitis, empedu) | Segmentasi sklera & analisis ruang warna CIELAB ($\Delta b^*$) dan HSV dengan **Auto-Kalibrasi Gray-World** |
| **Kepucatan Bibir & Mukosa** | Skrining Awal Anemia / Defisiensi Hemoglobin (Hb) | Rasio kromatik kemerahan vaskularisasi mukosa (*Red-to-Green Chromaticity*) |
| **Denyut Nadi Nirsentuh (rPPG)** | Estimasi Laju Jantung (BPM) & Variabilitas Detak | *Remote Photoplethysmography* menangkap fluktuasi mikroskopis penyerapan spektrum hijau kapiler kulit |
| **Kelelahan & Beban Mata** | *Eye Strain*, Kelelahan Kronis, & Kantuk Berat | *Eye Aspect Ratio* (EAR) dan *Mouth Aspect Ratio* (MAR / menguap) |

---

## 🧠 Fitur Unggulan Baru: Triangulasi Klinis & Multi-User

1. **Mesin Triangulasi Klinis (Clinical Triangulation Engine):**
   * Menghubungkan tanda visual AI dengan 3-4 kuesioner gejala kunci untuk menguatkan dugaan penyakit (Stroke Iskemik vs Bell's Palsy, Hepatitis/Empedu vs Efek Lampu Kamar, Anemia vs Mata Lelah).
   * Memberikan skor keyakinan klinis (0-100%), usulan pemeriksaan laboratorium (SGOT/SGPT, Bilirubin, Hb Darah), dan rekomendasi dokter spesialis yang tepat (Sp.S/Sp.N, Sp.PD).
2. **Auto-Kalibrasi Cahaya Ruangan (Gray-World Assumption):**
   * Menetralkan lampu bohlam kuning (*warm white*) secara otomatis agar tidak terjadi *false positive* pada mata kuning.
3. **Multi-User & Database Rekam Medis Lokal (SQLite):**
   * Mendukung profil banyak pengguna/pasien (misal: "Pengguna Utama", "Ayah", "Ibu", atau pasien klinik).
   * Menyimpan riwayat pemeriksaan lengkap dan menampilkan grafik tren biomarker dari waktu ke waktu.
4. **Bisa Diakses dari HP (Mobile / Wi-Fi):**
   * Server otomatis membuka jaringan lokal (`0.0.0.0`), memungkinkan siapa pun di jaringan Wi-Fi rumah/klinik yang sama membuka aplikasi lewat browser HP.

---

## 🚀 Cara Menjalankan Aplikasi

### 1. Jalankan di Komputer (Satu Klik)
Klik ganda (*double-click*) file:
`run_app.bat`

### 2. Membuka dari HP (Mobile Android / iPhone via Wi-Fi)
1. Hubungkan HP ke Wi-Fi yang sama dengan laptop.
2. Buka browser di HP (Chrome / Safari).
3. Ketik alamat IP laptop yang muncul di sidebar aplikasi, misalnya:
   `http://192.168.1.XX:8501`

---

## 🔒 Privasi & Keamanan Data (100% Offline & Lokal)
* Pemrosesan video dan landmark berjalan **100% secara lokal** di prosesor komputer Anda.
* Foto wajah **TIDAK PERNAH dikirim ke server luar, ke internet, atau ke pihak ketiga**.
* Database riwayat tersimpan aman dalam file lokal di folder `D:\facial_health_screening\data\`.
