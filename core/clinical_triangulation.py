from typing import Dict, List


class ClinicalTriangulationEngine:
    """Mesin Triangulasi Klinis & Diagnosa Banding.

    Menghubungkan biomarker visual objektif dari kamera dengan anamnesis gejala subjektif untuk memperkuat akurasi skrining penyakit.
    """

    @staticmethod
    def get_candidate_questions(visual_findings: dict) -> List[dict]:
        """Menghasilkan daftar pertanyaan klinis spesifik berdasarkan tanda visual yang terdeteksi."""
        questions = []

        sym = visual_findings.get("symmetry", {})
        jnd = visual_findings.get("jaundice", {})
        pal = visual_findings.get("pallor", {})
        rppg = visual_findings.get("rppg", {})

        # 1. Pertanyaan Asimetri (Jika ada deviasi sudut wajah)
        if sym.get("symmetry_score", 100) < 85.0:
            questions.append({
                "id": "asym_sudden",
                "category": "Saraf Wajah",
                "text": "Apakah kemiringan bibir/wajah ini muncul secara mendadak (kurang dari 24 jam terakhir)?",
            })
            questions.append({
                "id": "asym_limb_weakness",
                "category": "Saraf Wajah",
                "text": "Apakah disertai kelemahan/kebas pada satu sisi tangan/kaki atau kesulitan berbicara jelas (bicara pelo)?",
            })
            questions.append({
                "id": "asym_forehead_immobile",
                "category": "Saraf Wajah",
                "text": "Apakah dahi pada sisi yang turun sama sekali tidak bisa dikerutkan ke atas?",
            })

        # 2. Pertanyaan Ikterus / Sklera Kuning
        if jnd.get("jaundice_score", 0) > 30.0:
            questions.append({
                "id": "jaundice_dark_urine",
                "category": "Fungsi Hati & Empedu",
                "text": "Apakah air seni (urine) Anda beberapa hari ini berwarna gelap pekat seperti air teh?",
            })
            questions.append({
                "id": "jaundice_ruq_pain",
                "category": "Fungsi Hati & Empedu",
                "text": "Apakah ada rasa begah, mual, atau nyeri tumpul di perut bagian kanan atas?",
            })
            questions.append({
                "id": "jaundice_pale_stool",
                "category": "Fungsi Hati & Empedu",
                "text": "Apakah buang air besar (feses) berwarna lebih pucat menyerupai dempul/tanah liat?",
            })

        # 3. Pertanyaan Kepucatan / Anemia
        if pal.get("pallor_score", 0) > 35.0:
            questions.append({
                "id": "pallor_dizziness",
                "category": "Anemia & Sirkulasi",
                "text": "Apakah Anda sering merasa sangat lemas, mudah lelah, atau pusing melayang saat berdiri tiba-tiba?",
            })
            questions.append({
                "id": "pallor_cold_hands",
                "category": "Anemia & Sirkulasi",
                "text": "Apakah ujung jari tangan atau telapak kaki sering terasa dingin dan tampak pucat?",
            })

        # 4. Pertanyaan Kardiovaskular (Jika nadi terbaca tinggi/rendah)
        bpm = rppg.get("bpm")
        if bpm and (bpm > 100 or bpm < 55):
            questions.append({
                "id": "cardio_palpitation",
                "category": "Kardiovaskular",
                "text": "Apakah dada Anda terasa berdebar kencang atau detak jantung terasa tidak beraturan saat istirahat?",
            })

        return questions

    @staticmethod
    def evaluate(visual_findings: dict, user_answers: Dict[str, bool]) -> dict:
        """Mengevaluasi gabungan biomarker visual dan jawaban anamnesis untuk menghasilkan diagnosa banding klinis."""
        sym = visual_findings.get("symmetry", {})
        jnd = visual_findings.get("jaundice", {})
        pal = visual_findings.get("pallor", {})
        fat = visual_findings.get("fatigue", {})

        diagnoses = []

        # ----------------------------------------------------
        # 1. EVALUASI ASIMETRI WAJAH (STROKE vs BELL'S PALSY)
        # ----------------------------------------------------
        if sym.get("symmetry_score", 100) < 85.0:
            is_sudden = user_answers.get("asym_sudden", False)
            has_limb = user_answers.get("asym_limb_weakness", False)
            forehead_immobile = user_answers.get("asym_forehead_immobile", False)

            if is_sudden and has_limb:
                diagnoses.append({
                    "condition": "[PERINGATAN KRITIS] Suspek Stroke Iskemik Akut (Protokol F.A.S.T)",
                    "urgency": "EMERGENCY",
                    "confidence": 92.0,
                    "explanation": (
                        "Kombinasi kemiringan wajah mendadak disertai kelemahan anggota gerak atau bicara pelo adalah tanda khas stroke. "
                        "Setiap menit sangat berharga (Golden Period 3–4.5 jam) untuk mencegah kerusakan jaringan otak permanen."
                    ),
                    "action_plan": "Segera bawa ke Instalasi Gawat Darurat (IGD) Rumah Sakit terdekat dengan fasilitas CT Scan.",
                    "specialist": "Dokter Spesialis Neurologi / Saraf (Sp.N)",
                    "tests": ["CT Scan Kepala Non-Kontras", "MRI Otak", "EKG Jantung"],
                })
            elif forehead_immobile and not has_limb:
                diagnoses.append({
                    "condition": "Suspek Bell's Palsy (Paresis Saraf N.VII Perifer)",
                    "urgency": "HIGH",
                    "confidence": 85.0,
                    "explanation": (
                        "Kelumpuhan wajah unilateral yang melibatkan otot dahi tanpa disertai kelemahan tangan/kaki mengindikasikan inflamasi saraf fasialis perifer (Bell's Palsy)."
                    ),
                    "action_plan": "Kunjungi dokter spesialis saraf dalam 72 jam pertama untuk pemberian kortikosteroid antiinflamasi dan fisioterapi wajah.",
                    "specialist": "Dokter Spesialis Neurologi (Sp.N)",
                    "tests": ["Elektromiografi (EMG Saraf Wajah)", "Fisioterapi stimulasi saraf"],
                })
            else:
                diagnoses.append({
                    "condition": "Asimetri Wajah Ringan / Variasi Anatomi",
                    "urgency": "MODERATE",
                    "confidence": 65.0,
                    "explanation": "Terdapat sedikit deviasi simetri wajah tanpa gejala penyerta neurologis bermakna.",
                    "action_plan": "Observasi mandiri. Jika muncul keluhan baru seperti kebas atau sudut bibir semakin turun, periksakan ke dokter.",
                    "specialist": "Dokter Umum",
                    "tests": ["Pemeriksaan fisik saraf kranial"],
                })

        # ----------------------------------------------------
        # 2. EVALUASI SKLERA IKTERIK (HEPATITIS / EMPEDU vs LAMPU)
        # ----------------------------------------------------
        if jnd.get("jaundice_score", 0) > 30.0:
            dark_urine = user_answers.get("jaundice_dark_urine", False)
            ruq_pain = user_answers.get("jaundice_ruq_pain", False)
            pale_stool = user_answers.get("jaundice_pale_stool", False)

            if dark_urine or ruq_pain or pale_stool:
                diagnoses.append({
                    "condition": "Suspek Gangguan Hepato-Bilier (Fungsi Hati / Saluran Empedu)",
                    "urgency": "HIGH",
                    "confidence": 88.0,
                    "explanation": (
                        "Deteksi sklera menguning yang diperkuat dengan urine gelap seperti teh atau rasa begah di kanan atas merupakan indikasi kuat penumpukan bilirubin (Hepatitis, fatty liver, atau batu empedu)."
                    ),
                    "action_plan": "Konsultasikan ke Dokter Spesialis Penyakit Dalam untuk pemeriksaan darah hepar lengkap.",
                    "specialist": "Dokter Spesialis Penyakit Dalam (Sp.PD / K-GEH)",
                    "tests": ["Bilirubin Total / Direct / Indirect", "SGOT & SGPT", "USG Abdomen"],
                })
            else:
                diagnoses.append({
                    "condition": "Sklera Kuning Ringan (Kemungkinan Efek Cahaya / Kelelahan)",
                    "urgency": "LOW",
                    "confidence": 75.0,
                    "explanation": (
                        "Meskipun kamera mendeteksi sedikit warna kuning, ketiadaan gejala sistemik (urine tidak gelap, tidak ada mual) menunjukkan kemungkinan besar disebabkan oleh pencahayaan lampu ruangan atau mata lelah."
                    ),
                    "action_plan": "Periksa ulang di bawah cahaya matahari alami dan cukupi kebutuhan air minum harian.",
                    "specialist": "Dokter Umum (bila keluhan berlanjut)",
                    "tests": ["Skrining Bilirubin Darah (opsional)"],
                })

        # ----------------------------------------------------
        # 3. EVALUASI KEPUCATAN (ANEMIA)
        # ----------------------------------------------------
        if pal.get("pallor_score", 0) > 35.0:
            dizziness = user_answers.get("pallor_dizziness", False)
            cold_hands = user_answers.get("pallor_cold_hands", False)

            if dizziness or cold_hands:
                diagnoses.append({
                    "condition": "Suspek Anemia Defisiensi Besi / Hemoglobin Rendah",
                    "urgency": "MODERATE",
                    "confidence": 82.0,
                    "explanation": (
                        "Kepucatan pada mukosa bibir dan kelopak mata bawah yang disertai keluhan lemas atau pusing berkunang-kunang menguatkan indikasi kekurangan sel darah merah / hemoglobin."
                    ),
                    "action_plan": "Lakukan tes darah sederhana untuk mengukur kadar hemoglobin (Hb) dan konsumsi makanan kaya zat besi.",
                    "specialist": "Dokter Umum / Dokter Spesialis Penyakit Dalam (Sp.PD)",
                    "tests": ["Darah Rutin / Lengkap (CBC)", "Kadar Hemoglobin (Hb)", "Serum Ferritin"],
                })

        # ----------------------------------------------------
        # 4. DEFAULT: KONDISI SEHAT / NORMAL
        # ----------------------------------------------------
        if not diagnoses:
            diagnoses.append({
                "condition": "Profil Biomarker Wajah Normal (Sehat)",
                "urgency": "LOW",
                "confidence": 95.0,
                "explanation": "Simetri wajah prima, sklera mata jernih, vaskularisasi bibir baik, dan tidak ada tanda kelelahan ekstrem.",
                "action_plan": "Pertahankan pola hidup sehat, tidur teratur 7-8 jam per hari, dan hidrasi air putih yang cukup.",
                "specialist": "Tidak ada rujukan khusus yang diperlukan saat ini.",
                "tests": ["Pemeriksaan kesehatan rutin berkala (General Medical Checkup tahunan)"],
            })

        return {
            "primary_diagnosis": diagnoses[0],
            "all_differential_diagnoses": diagnoses,
        }
