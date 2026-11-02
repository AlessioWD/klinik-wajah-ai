import os
import sqlite3
import pandas as pd

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DB_PATH = os.path.join(DB_DIR, "screening_history.db")


class HistoryDatabase:
    """Mengelola penyimpanan database riwayat skrining pasien lokal berbasis SQLite."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Membuat tabel jika belum ada."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Tabel Pasien / Profil Pengguna
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # Tabel Riwayat Skrining
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS screening_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_name TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    symmetry_score REAL,
                    jaundice_score REAL,
                    pallor_score REAL,
                    fatigue_score REAL,
                    bpm REAL,
                    primary_condition TEXT,
                    urgency TEXT,
                    notes TEXT
                )
                """
            )
            # Buat profil bawaan jika kosong
            cursor.execute("SELECT COUNT(*) FROM patients")
            if cursor.fetchone()[0] == 0:
                cursor.execute("INSERT OR IGNORE INTO patients (name) VALUES (?)", ("Pengguna Utama",))
                cursor.execute("INSERT OR IGNORE INTO patients (name) VALUES (?)", ("Keluarga / Tamu",))
            conn.commit()

    def get_patients(self) -> list:
        """Mengambil daftar semua profil nama pengguna."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM patients ORDER BY id ASC")
            rows = cursor.fetchall()
            return [r[0] for r in rows]
        finally:
            conn.close()

    def add_patient(self, name: str) -> bool:
        """Menambahkan profil pengguna baru."""
        name = name.strip()
        if not name:
            return False
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO patients (name) VALUES (?)", (name,))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()

    def save_screening(
        self,
        patient_name: str,
        symmetry_score: float,
        jaundice_score: float,
        pallor_score: float,
        fatigue_score: float,
        bpm: float,
        primary_condition: str,
        urgency: str,
        notes: str = "",
    ):
        """Menyimpan satu sesi pemeriksaan skrining ke database."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO screening_records (
                    patient_name, symmetry_score, jaundice_score, pallor_score, 
                    fatigue_score, bpm, primary_condition, urgency, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    patient_name,
                    round(float(symmetry_score), 1) if symmetry_score is not None else None,
                    round(float(jaundice_score), 1) if jaundice_score is not None else None,
                    round(float(pallor_score), 1) if pallor_score is not None else None,
                    round(float(fatigue_score), 1) if fatigue_score is not None else None,
                    round(float(bpm), 1) if bpm is not None else None,
                    primary_condition,
                    urgency,
                    notes,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_patient_history(self, patient_name: str) -> pd.DataFrame:
        """Mengambil riwayat pemeriksaan pasien sebagai DataFrame pandas."""
        conn = self._get_connection()
        try:
            query = """
                SELECT id, timestamp, symmetry_score, jaundice_score, pallor_score, 
                       fatigue_score, bpm, primary_condition, urgency, notes
                FROM screening_records
                WHERE patient_name = ?
                ORDER BY timestamp DESC
            """
            df = pd.read_sql_query(query, conn, params=(patient_name,))
            return df
        finally:
            conn.close()
