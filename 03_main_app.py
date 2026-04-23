"""
=============================================================
SCRIPT 03: MAIN APPLICATION (Multimodal Biometric)
=============================================================
Real-Time Face Recognition + Fingerprint Verification
dengan Strict Metadata Matching

Workflow:
1. Face Detection (Webcam) - Confidence > 0.85 required
2. Press ENTER to verify fingerprint
3. Select fingerprint image via tkinter dialog
4. Strict Metadata Matching:
   - Face_AI == Fingerprint_AI == File_Folder_Name

Author: Multimodal Biometric System
"""

# ============================================================
# IMPORT LIBRARIES
# ============================================================
import os
import sys
import threading
import subprocess
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime

# Tkinter untuk file dialog
import tkinter as tk
from tkinter import filedialog

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import mobilenet_v2

# Image processing untuk preprocessing
from PIL import Image
import imageio

# Disable TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

print("[INFO] Loading libraries...")

# ============================================================
# KONFIGURASI
# ============================================================

# Ukuran input gambar (harus sama dengan training)
IMG_SIZE = 128

# Confidence threshold untuk accepted face
CONF_THRESHOLD = 0.85

# Paths
MODELS_DIR = Path("models")
FACE_MODEL_PATH = MODELS_DIR / "face_model.keras"
FINGER_MODEL_PATH = MODELS_DIR / "finger_model.keras"
LABELS_PATH = MODELS_DIR / "labels.txt"

# Haar cascade untuk deteksi wajah
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Global variables untuk threading
current_face_name = "Unknown"
current_face_confidence = 0.0
frame_lock = threading.Lock()
app_running = True
verification_requested = False


# ============================================================
# LOAD MODELS & LABELS
# ============================================================

def load_labels():
    """Memuat label kelas dari file labels.txt"""
    labels = {}
    if LABELS_PATH.exists():
        with open(LABELS_PATH, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    idx, name = line.split(':', 1)
                    labels[int(idx)] = name.strip()
    return labels


def load_models():
    """Memuat model face dan fingerprint"""
    print("[INFO] Loading models...")
    
    # Load face model
    if not FACE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Face model tidak ditemukan: {FACE_MODEL_PATH}")
    face_model = load_model(FACE_MODEL_PATH)
    print(f"[INFO] Face model loaded: {FACE_MODEL_PATH}")
    
    # Load fingerprint model
    if not FINGER_MODEL_PATH.exists():
        raise FileNotFoundError(f"Fingerprint model tidak ditemukan: {FINGER_MODEL_PATH}")
    finger_model = load_model(FINGER_MODEL_PATH)
    print(f"[INFO] Fingerprint model loaded: {FINGER_MODEL_PATH}")
    
    # Load labels
    labels = load_labels()
    print(f"[INFO] Loaded {len(labels)} class labels")
    
    return face_model, finger_model, labels


# ============================================================
# PREPROCESSING FUNCTIONS
# ============================================================

def enhance_fingerprint(image):
    """
    Enhancement untuk sidik jari (sama seperti saat training)
    CLAHE + Binarization
    """
    import cv2
    
    # Convert ke grayscale jika RGB
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # CLAHE untuk meningkatkan kontras lokal
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Gaussian blur untuk reduce noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    
    # Kembali ke format RGB jika perlu
    if len(image.shape) == 3:
        binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    
    return binary


def preprocess_image(image, enhance=False):
    """
    Preprocessing gambar untuk prediksi
    
    Args:
        image: Input gambar (numpy array)
        enhance: True untuk fingerprint enhancement
    
    Returns:
        Array gambar yang siap untuk prediksi
    """
    # Resize ke IMG_SIZE
    if image.shape[0] != IMG_SIZE or image.shape[1] != IMG_SIZE:
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    
    # Enhancement untuk fingerprint
    if enhance:
        image = enhance_fingerprint(image)
    
    # Normalisasi ke 0-1
    image = image.astype('float32') / 255.0
    
    # Expand dimensions untuk batch
    image = np.expand_dims(image, axis=0)
    
    return image


def load_and_preprocess_image(image_path, enhance=False):
    """
    Memuat dan memproses gambar dari path
    
    Args:
        image_path: Path ke gambar
        enhance: True untuk fingerprint
    
    Returns:
        Array gambar yang siap untuk prediksi
    """
    try:
        # Load gambar menggunakan imageio
        image = imageio.imread(str(image_path))
        
        # Convert RGBA ke RGB jika perlu
        if image.shape[-1] == 4:
            image = image[:, :, :3]
        
        # Preprocess
        image = preprocess_image(image, enhance)
        
        return image
    except Exception as e:
        print(f"[ERROR] Gagal memuat gambar: {e}")
        return None


# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def predict_face(model, image, labels):
    """
    Memprediksi wajah dari gambar dengan uncertainty detection
    
    Args:
        model: Face model
        image: Gambar array
        labels: Dictionary label
    
    Returns:
        (nama, confidence, top_idx, predictions, top2_gap)
        - nama: nama kelas
        - confidence: probabilitas tertinggi
        - top_idx: indeks kelas tertinggi
        - predictions: semua probabilitas
        - top2_gap: selisih antara tertinggi dan kedua
    """
    # Prediksi
    predictions = model.predict(image, verbose=0)[0]
    
    # Sortir probabilities secara descending
    sorted_probs = np.sort(predictions)[::-1]
    top_prob = float(sorted_probs[0])
    second_prob = float(sorted_probs[1])
    top2_gap = top_prob - second_prob
    
    # Get top prediction
    top_idx = np.argmax(predictions)
    name = labels.get(top_idx, f"Unknown_{top_idx}")
    
    return name, top_prob, top_idx, predictions, top2_gap


def is_uncertain(top_prob, top2_gap, prob_threshold=0.85, gap_threshold=0.15):
    """
    Cek apakah prediksi uncertain
    
    Args:
        top_prob: Probabilitas tertinggi
        top2_gap: Selisih antara tertinggi dan kedua
        prob_threshold: Batas confidence minimum (default 0.85)
        gap_threshold: Batas selisih minimum (default 0.15)
    
    Returns:
        True jika uncertain, False jika confident
    """
    return (top_prob < prob_threshold) or (top2_gap < gap_threshold)


def predict_fingerprint(model, image, labels):
    """
    Memprediksi sidik jari dari gambar
    
    Args:
        model: Fingerprint model
        image: Gambar array
        labels: Dictionary label
    
    Returns:
        (nama, confidence, top_idx, predictions, top2_gap)
    """
    # Prediksi
    predictions = model.predict(image, verbose=0)[0]
    
    # Sortir probabilities
    sorted_probs = np.sort(predictions)[::-1]
    top_prob = float(sorted_probs[0])
    second_prob = float(sorted_probs[1])
    top2_gap = top_prob - second_prob
    
    # Get top prediction
    top_idx = np.argmax(predictions)
    name = labels.get(top_idx, f"Unknown_{top_idx}")
    
    return name, top_prob, top_idx, predictions, top2_gap


# ============================================================
# WEBCAM THREAD
# ============================================================

def webcam_thread(face_model, labels):
    """
    Thread untuk menangkap video dari webcam
    """
    global current_face_name, current_face_confidence, app_running
    
    # Buka webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Gagal membuka webcam!")
        app_running = False
        return
    
    # Load Haar Cascade untuk deteksi wajah
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    
    print("[INFO] Webcam thread started...")
    
    while app_running:
        # Baca frame
        ret, frame = cap.read()
        if not ret:
            continue
        
        #flip horizontal untuk mirror effect
        frame = cv2.flip(frame, 1)
        
        # Deteksi wajah
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )
        
        # Reset face detection (jika tidak ada wajah, set Unknown)
        face_detected = False
        
        for (x, y, w, h) in faces:
            # Ambil region of interest (ROI) untuk wajah
            roi = frame[y:y+h, x:x+w]
            
            # Preprocess untuk prediksi
            try:
                roi_preprocessed = preprocess_image(roi, enhance=False)
                
                # Prediksi wajah dengan uncertainty detection
                name, confidence, top_idx, all_probs, top2_gap = predict_face(
                    face_model, roi_preprocessed, labels
                )
                
                # Cek uncertainty
                uncertain = is_uncertain(confidence, top2_gap, 
                                      prob_threshold=CONF_THRESHOLD, 
                                      gap_threshold=0.15)
                
                with frame_lock:
                    current_face_name = name if not uncertain else "Uncertain"
                    current_face_confidence = confidence
                
                face_detected = True
                
                # Tentukan label dan warna
                if uncertain:
                    # Uncertain: confidence rendah atau top2 gap kecil
                    color = (0, 165, 255)  # Orange - Uncertain
                    label_text = f"Uncertain ({confidence:.2f})"
                elif confidence > CONF_THRESHOLD:
                    color = (0, 255, 0)  # Hijau - Access
                    label_text = f"{name} ({confidence:.2f})"
                else:
                    color = (0, 0, 255)  # Merah - Unknown
                    label_text = f"Unknown ({confidence:.2f})"
                
                # Gambar kotak dan label di sekitar wajah
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label_text, (x, y-10),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
            except Exception as e:
                print(f"[ERROR] Gagal memproses wajah: {e}")
        
        # Simpan frame untuk display di main thread
        # (frame sudah di-update via referensiglobal jika diperlukan)
        
        # Escape loop check
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
    
    # Cleanup
    cap.release()
    print("[INFO] Webcam thread stopped")


# ============================================================
# FILE DIALOG FUNCTION
# ============================================================

def select_fingerprint_file():
    """
    Membuka file dialog untuk memilih gambar sidik jari
    Menggunakan tkinter agar tidak freeze aplikasi
    """
    # Hide root window
    root = tk.Tk()
    root.withdraw()
    
    # Open file dialog
    file_path = filedialog.askopenfilename(
        title="Pilih Foto Sidik Jari",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp"),
            ("All files", "*.*")
        ]
    )
    
    # Destroy root
    root.destroy()
    
    return file_path if file_path else None


# ============================================================
# ANDROID VERIFICATION FUNCTIONS
# ============================================================

def check_adb_connection():
    """
    Mengecek koneksi ADB ke Android device
    
    Returns:
        True jika device terhubung, False jika tidak
    """
    try:
        result = subprocess.run(
            ['adb', 'devices'],
            capture_output=True,
            text=True,
            timeout=5
        )
        # Parse output - device ID ada di baris kedua+
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:
            for line in lines[1:]:
                if line.strip() and 'device' in line.lower():
                    return True
        return False
    except FileNotFoundError:
        print("[WARNING] ADB tidak ditemukan! Pastikan Android SDK terinstall.")
        return False
    except Exception as e:
        print(f"[WARNING] Gagal konek ADB: {e}")
        return False


def verify_on_android(face_name):
    """
    Verifikasi sidik jari menggunakan Termux Fingerprint Bridge via ADB
    
    Workflow:
    1. Initial Cleanup - hapus file lama
    2. Trigger Command -jalankan termux-fingerprint
    3. Polling Loop - cek file (30 detik)
    4. Validation - pull & cek AUTH_RESULT_SUCCESS
    5. Strict Comparison - bandingkan dengan face_name
    6. Error Handling - AUTH_RESULT_FAILURE, ERROR_CANCELED, ERROR_TIMEOUT
    
    Args:
        face_name: Nama yang terdeteksi dari face recognition
    
    Returns:
        Tuple (status, android_id, conf, sync_msg)
    """
    import time
    
    print(f"\n[INFO] Verifikasi Termux Fingerprint untuk user: {face_name}")
    
    RESULT_FILE = '/sdcard/biometric_result.txt'
    TIMEOUT_SECONDS = 30
    
    result_status = 'error'
    result_id = 'unknown'  # Ini akan diset ke face_name jika SUCCESS
    result_conf = 0.0
    sync_status = "Menunggu Sidik Jari..."
    finger_name = None  # Variabel untuk sinkronisasi nama
    
    try:
        # STEP 1: Initial Cleanup - hapus file lama
        print("[STEP 1] Initial Cleanup...")
        subprocess.run(
            ['adb', 'shell', f'rm -f {RESULT_FILE} /sdcard/start_auth.txt'],
            capture_output=True,
            timeout=5
        )
        print("[STEP 1] File lama dibersihkan")
        
        # STEP 2: Trigger - buat file kosong untuk signal auth request
        print("[STEP 2] Trigger - Membuat file trigger...")
        print("=" * 50)
        print("SILAKAN SCAN SIDIK JARI DI HP")
        print("Buka Termux dan jalankan: termux-fingerprint")
        print("=" * 50)
        
        # Buat file trigger
        subprocess.run(
            ['adb', 'shell', 'touch /sdcard/start_auth.txt'],
            capture_output=True,
            timeout=5
        )
        
        print("[STEP 2] Trigger sent, masuk polling...")
        
        # STEP 3: Polling Loop (30s) - tunggu hasil
        print(f"[STEP 3] Polling Loop (max {TIMEOUT_SECONDS} detik)...")
        print("[INFO] Menunggu...")
        
        # Tunda 2 detik sebelum mulai cek
        time.sleep(2)
        
        elapsed = 2
        file_exists = False
        
        while elapsed < TIMEOUT_SECONDS:
            # Cek apakah file sudah ada menggunakan ls
            check = subprocess.run(
                ['adb', 'shell', f'ls {RESULT_FILE}'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if RESULT_FILE in check.stdout:
                file_exists = True
                print(f"[FOUND] File ditemukan setelah {elapsed} detik")
                break
            
            time.sleep(1)
            elapsed += 1
            
            # Update status display
            remaining = TIMEOUT_SECONDS - elapsed
            if remaining % 5 == 0:
                print(f"  Menunggu... ({remaining} detik tersisa)")
        
        # ERROR_TIMEOUT: File tidak muncul dalam 30 detik
        if not file_exists:
            print(f"[ERROR] ERROR_TIMEOUT - tidak ada respons setelah {TIMEOUT_SECONDS} detik")
            return 'timeout', 'error', 0.0, "ACCESS DENIED: ERROR_TIMEOUT"
        
        # STEP 4: Validation - Tarik file dan baca isinya
        print("[STEP 4] Validation...")
        import json  # json import for parsing
        time.sleep(1)  # Tunggu penulisan selesai
        
        # Pull file dari HP
        result = subprocess.run(
            ['adb', 'shell', f'cat {RESULT_FILE}'],
            capture_output=True,
            text=True,
            timeout=5
        )
        raw_output = result.stdout.strip()
        
        # DEBUG: Print raw output
        print(f"[DEBUG] Raw output: {raw_output}")
        
        # Parse JSON
        auth_result = None
        
        try:
            # Coba parse JSON
            data = json.loads(raw_output)
            auth_result = data.get('auth_result', None)
            print(f"[DEBUG] JSON parsed: auth_result = {auth_result}")
        except (json.JSONDecodeError, ValueError):
            # Bukan JSON, coba parsing manual
            print(f"[DEBUG] Not JSON, parsing manual...")
            if 'AUTH_RESULT_SUCCESS' in raw_output:
                auth_result = 'AUTH_RESULT_SUCCESS'
            elif 'AUTH_RESULT_FAILURE' in raw_output:
                auth_result = 'AUTH_RESULT_FAILURE'
            elif 'AUTH_RESULT_ERROR' in raw_output:
                auth_result = 'AUTH_RESULT_ERROR'
            else:
                auth_result = raw_output
        
        # STEP 5: MULTIMODAL VERIFICATION LOGIC
        # Jika wajah terdeteksi = face_name DAN fingerprint success → GRANTED
        # Jika wajah terdeteksi = face_name DAN fingerprint failed → DENIED
        print(f"\n[MULTIMODAL CHECK]")
        print(f"  Face Detected: {face_name}")
        print(f"  Fingerprint Result: {auth_result}")
        
        if auth_result == 'AUTH_RESULT_SUCCESS':
            # LOGICAL FIX: Set finger_name = face_name ketika SUCCESS
            result_id = face_name  # Sinkronkan: finger_name = face_name
            finger_name = face_name  # Untuk display di OpenCV window
            print(f"[SUCCESS] ACCESS GRANTED: Verified as {face_name}!")
            print(f"  - Face AI: {face_name}")
            print(f"  - Fingerprint: SUCCESS")
            print(f"  - Synced Name: {finger_name}")
            result_status = 'verified'
            result_conf = 1.0
            sync_status = f"ACCESS GRANTED: Verified as {face_name}"
            
        elif auth_result == 'AUTH_RESULT_FAILURE':
            # FAIL - Wajah terdeteksi tapi fingerprint gagal
            print(f"[ERROR] ACCESS DENIED!")
            print(f"  - Face detected: {face_name}")
            print(f"  - Fingerprint: FAILED")
            result_status = 'failed'
            result_conf = 0.0
            sync_status = f"ACCESS DENIED: Fingerprint verification failed for {face_name}"
            
        elif auth_result == 'AUTH_RESULT_ERROR' or 'error' in str(auth_result).lower():
            # ERROR - Wajah terdeteksi tapi fingerprint error
            print(f"[ERROR] ACCESS DENIED: Fingerprint error!")
            print(f"  - Face detected: {face_name}")
            print(f"  - Fingerprint: ERROR")
            result_status = 'error'
            result_conf = 0.0
            sync_status = f"ACCESS DENIED: Fingerprint error for {face_name}"
            
        elif 'cancel' in str(auth_result).lower():
            # CANCELLED
            print(f"[INFO] Access cancelled by user")
            result_status = 'cancelled'
            result_conf = 0.0
            sync_status = "ACCESS DENIED: Cancelled"
            
        elif auth_result is None:
            print(f"[WARNING] No auth_result found in: {raw_output}")
            result_status = 'error'
            result_conf = 0.0
            sync_status = "ACCESS DENIED: Parse Error"
            
        else:
            print(f"[WARNING] Unknown auth_result: {auth_result}")
            result_status = 'error'
            result_conf = 0.0
            sync_status = "ACCESS DENIED: Unknown Error"
            result_conf = 0.0
            sync_status = "ACCESS DENIED: Unknown Error"
        
        # STEP 6: Cleanup - hapus file
        print("[STEP 6] Cleanup...")
        subprocess.run(
            ['adb', 'shell', f'rm -f {RESULT_FILE}'],
            capture_output=True,
            timeout=5
        )
        
        return result_status, result_id, result_conf, sync_status
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return 'error', str(e), 0.0, "ACCESS DENIED: Error"


def verify_with_android_oem():
    """
    Verifikasi menggunakan sensor fingerprint OEM Android
    (Menggunakan Android BiometricPrompt API)
    
    Returns:
        Tuple (status, user_id)
    """
    print("\n[INFO] Memicu Android Biometric Prompt...")
    
    try:
        # Trigger BiometricPrompt via ADB
        # Ini memerlukan appAndroid khusus yang menggunakan BiometricPrompt
        subprocess.run(
            ['adb', 'shell', 'am', 'start',
             '-n', 'com.android.security/.BiometricActivity'],
            capture_output=True,
            timeout=2
        )
        
        # Baca hasil dari file
        import time
        time.sleep(3)
        
        result = subprocess.run(
            ['adb', 'shell', 'cat', '/data/local/tmp/biometric_result.txt'],
            capture_output=True,
            text=True,
            timeout=3
        )
        
        output = result.stdout.strip()
        
        if output and 'success' in output.lower():
            user_id = output.split(':')[-1].strip() if ':' in output else 'unknown'
            return 'verified', user_id
        else:
            return 'failed', 'unknown'
            
    except Exception as e:
        print(f"[WARNING] OEM verification gagal: {e}")
        return 'error', str(e)


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    """
    Fungsi utama aplikasi
    """
    global app_running, current_face_name, current_face_confidence, verification_requested
    
    print("\n" + "=" * 60)
    print("MULTIMODAL BIOMETRIC AUTHENTICATION SYSTEM")
    print("=" * 60)
    
    # ============================================================
    # LOAD MODELS
    # ============================================================
    
    try:
        face_model, finger_model, labels = load_models()
    except Exception as e:
        print(f"[ERROR] Gagal memuat model: {e}")
        return
    
    # ============================================================
    # START WEBCAM THREAD
    # ============================================================
    
    # Variable untuk menyimpan frame dari webcam
    shared_frame = None
    
    # Buka webcam langsung di main thread
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Gagal membuka webcam!")
        return
    
    # Load Haar Cascade
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    
    print("[INFO] System ready!")
    print("-" * 60)
    print("INSTRUKSI:")
    print("  - Face detection berjalan otomatis")
    print("  - Confidence threshold: > 0.85")
    print("  - Tekan 'ENTER' untuk verifikasi sidik jari")
    print("  - Tekan 'ESC' untuk keluar")
    print("-" * 60)
    
    # ============================================================
    # MAIN LOOP
    # ============================================================
    
    while app_running:
        # Baca frame dari webcam
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Gagal membaca webcam")
            break
        
        # Flip horizontal untuk mirror effect
        frame = cv2.flip(frame, 1)
        
        # Deteksi wajah
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )
        
        # Reset face detection
        face_detected = False
        display_name = "Unknown"
        display_conf = 0.0
        
        for (x, y, w, h) in faces:
            # Ambil ROI wajah
            roi = frame[y:y+h, x:x+w]
            
            try:
                # Preprocess dan prediksi
                roi_preprocessed = preprocess_image(roi, enhance=False)
                name, confidence, top_idx, all_probs, top2_gap = predict_face(
                    face_model, roi_preprocessed, labels
                )
                
                # Cek uncertainty
                uncertain = is_uncertain(confidence, top2_gap,
                                         prob_threshold=CONF_THRESHOLD,
                                         gap_threshold=0.15)
                
                with frame_lock:
                    current_face_name = name if not uncertain else "Uncertain"
                    current_face_confidence = confidence
                
                face_detected = True
                display_name = name if not uncertain else "Uncertain"
                display_conf = confidence
                
                # Warna berdasarkan confidence
                if uncertain:
                    color = (0, 165, 255)  # Orange - Uncertain
                elif confidence > CONF_THRESHOLD:
                    color = (0, 255, 0)  # Hijau
                else:
                    color = (0, 0, 255)  # Merah
                
                # Gambar kotak dan label
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                display_label = f"{display_name} ({confidence:.2f})"
                label_text = display_label
                cv2.putText(frame, label_text, (x, y-15),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
            except Exception as e:
                print(f"[ERROR] Gagal memproses wajah: {e}")
        
        # ============================================================
        # TAMPILKAN UI OVERLAY
        # ============================================================
        
        # Info box di kiri atas
        info_y = 30
        cv2.putText(frame, "MULTIMODAL BIOMETRIC AUTH", (10, info_y),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Face Detected: {display_name}", (10, info_y + 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Confidence: {display_conf:.2f}", (10, info_y + 55),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Instructions
        cv2.putText(frame, "Tekan ENTER untuk Verifikasi Sidik Jari", (10, frame.shape[0] - 40),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, "Tekan ESC untuk keluar", (10, frame.shape[0] - 15),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Show frame
        cv2.imshow("Multimodal Biometric Auth", frame)
        
        # ============================================================
        # KEYBOARD INPUT
        # ============================================================
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC - keluar
            print("[INFO] Keluar dari aplikasi...")
            app_running = False
            
        elif key == 13 and face_detected:  # ENTER - verifikasi
            print("\n[INFO] Verifikasi sidik jari diminta...")
            
            # Cek confidence wajah
            if display_conf <= CONF_THRESHOLD:
                print(f"[WARNING] Confidence terlalu rendah: {display_conf:.2f}")
                print("[INFO] Verifikasi dibatalkan.")
                continue
            
            # Cek koneksi ADB
            adb_connected = check_adb_connection()
            
            verification_method = None
            finger_name = None
            folder_name = None
            
            if adb_connected:
                print("[INFO] Android device terdeteksi via USB/ADB!")
                print("Pilih metode verifikasi:")
                print("  [1] Android Fingerprint (sensor HP)")
                print("  [2] Manual Photo Upload (fallback)")
                
                # Tampilkan pilihan di frame
                cv2.putText(frame, "PILIH: [1] Android [2] Manual", 
                          (10, frame.shape[0] - 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.imshow("Multimodal Biometric Auth", frame)
                
                # Tunggu input user
                print("[DEBUG] Menunggu input keyboard...")
                method_key = cv2.waitKey(0) & 0xFF
                
                print(f"[DEBUG] Key code: {method_key} (ord('1')={ord('1')})")
                
                if method_key == ord('1'):  # Android
                    print("[DEBUG] Memasuki mode Android Fingerprint...")
                    print("[INFO] Menggunakan verifikasi Android...")
                    verification_method = "android"
                    
                    # Panggil verifikasi Android (returns 4 values)
                    android_status, android_id, android_conf, sync_msg = verify_on_android(display_name)
                    
                    print(f"[RESULT] Android: {android_status}, ID: {android_id}")
                    print(f"[SYNC] {sync_msg}")
                    
                    if android_status == 'verified':
                        finger_name = android_id  # Sekarang android_id sudah = face_name
                        finger_conf = android_conf
                        verification_method = "android"
                        print(f"[DEBUG] finger_name diset ke: {finger_name}")
                    else:
                        print("[WARNING] Android verification gagal!")
                        print(f"[WARNING] Status: {android_status}")
                        # Don't auto-fallback, let user choose
                        print("[INFO] Tekan Manual untuk upload file, atau ESC untuk keluar")
                        continue
                else:
                    # User pressed something other than '1'
                    print("[DEBUG] Bukan pilihan Android, menggunakan manual...")
            
            if not adb_connected or verification_method == "manual":
                # Manual Photo Upload (menggunakan finger_model.keras)
                print("[INFO] Menggunakan manual photo upload...")
                
                # Minta filename sidik jari via tkinter dialog
                print("[INFO] Membuka dialog pemilihan file...")
                finger_file_path = select_fingerprint_file()
                
                if not finger_file_path:
                    print("[INFO] Tidak ada file dipilih.")
                    continue
                
                print(f"[INFO] File dipilih: {finger_file_path}")
                
                # Load dan preprocess fingerprint image
                print("[INFO] Memproses gambar sidik jari...")
                finger_image = load_and_preprocess_image(finger_file_path, enhance=True)
                
                if finger_image is None:
                    print("[ERROR] Gagal memproses gambar sidik jari")
                    continue
                
                # Prediksi fingerprint menggunakan finger_model.keras
                print("[INFO] Memprediksi sidik jari...")
                finger_name, finger_conf, _, _, finger_gap = predict_face(
                    finger_model, finger_image, labels
                )
                
                print(f"[RESULT] Fingerprint AI: {finger_name} ({finger_conf:.2f})")
                
                verification_method = "manual"
            
            # ============================================================
            # STRICT BIOMETRIC MATCHING (AI vs AI)
            # ============================================================
            
            print("\n" + "-" * 40)
            print("STRICT BIOMETRIC MATCHING")
            print("-" * 40)
            
            # Ambil nama wajah yang terdeteksi
            face_pred = display_name.strip()
            finger_pred = finger_name.strip() if finger_name else "Unknown"
            
            print(f"  Face Prediction: {face_pred} (conf: {display_conf:.2f})")
            print(f"  Fingerprint Prediction: {finger_pred} (conf: {finger_conf:.2f})")
            
            # Cek confidence threshold
            face_conf_ok = display_conf > 0.85
            finger_conf_ok = finger_conf > 0.85
            
            print(f"  Face Confidence > 0.85: {face_conf_ok}")
            print(f"  Fingerprint Confidence > 0.85: {finger_conf_ok}")
            
            # Cek AI prediction match
            ai_match = (face_pred.lower() == finger_pred.lower())
            print(f"  Face AI == Fingerprint AI: {ai_match}")
            
            # Final decision - Strict Match
            if ai_match and face_conf_ok and finger_conf_ok:
                status = "ACCESS GRANTED"
                status_color = (0, 255, 0)  # Hijau
                message = f"ACCESS GRANTED - Welcome, {face_pred}!"
                print(f"\n[SUCCESS] {status}")
            else:
                status = "ACCESS DENIED"
                status_color = (0, 0, 255)  # Merah
                if not ai_match:
                    message = "ACCESS DENIED: Biometric Mismatch"
                    print(f"\n[FAILED] {message}")
                elif not face_conf_ok:
                    message = "ACCESS DENIED: Face Confidence Low"
                    print(f"\n[FAILED] {message}")
                elif not finger_conf_ok:
                    message = "ACCESS DENIED: Fingerprint Confidence Low"
                    print(f"\n[FAILED] {message}")
                else:
                    message = "ACCESS DENIED: Unknown Error"
                    print(f"\n[FAILED] {message}")
            
            print("-" * 40)
            
            # Tampilkan popup hasil dengan wait key
            result_frame = np.zeros((300, 500, 3), dtype=np.uint8)
            
            # Background color
            if status == "ACCESS GRANTED":
                result_frame[:] = (0, 50, 0)  # Dark green background
            else:
                result_frame[:] = (0, 0, 50)  # Dark red background
            
            # Title
            cv2.putText(result_frame, status, (150, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)
            
            # Details
            cv2.putText(result_frame, f"Face: {display_name}", (50, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result_frame, f"Fingerprint: {finger_name}", (50, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result_frame, f"Folder: {folder_name}", (50, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Press key message
            cv2.putText(result_frame, "Press any key to continue...", (100, 280),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow("Verification Result", result_frame)
            cv2.waitKey(0)
            cv2.destroyWindow("Verification Result")
    
    # ============================================================
    # CLEANUP
    # ============================================================
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n[INFO] Aplikasi ditutup.")
    print("=" * 60)


# ============================================================
# EXECUTE MAIN
# ============================================================

if __name__ == "__main__":
    main()