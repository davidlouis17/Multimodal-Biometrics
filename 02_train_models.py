"""
=============================================================
SCRIPT 02: TRAINING MODEL MULTIMODAL (FACE & FINGERPRINT)
=============================================================
Sistem Biometrik Multimodal dengan Transfer Learning MobileNetV2
Untuk-face recognition & fingerprint recognition

Arsitektur:
- MobileNetV2 (Transfer Learning) sebagai base model
- GlobalAveragePooling2D -> Dense(256) -> Dropout(0.6) -> Dense(num_classes)
- Input: 128x128 RGB images

Author: Multimodal Biometric System
"""

# ============================================================
# IMPORT LIBRARIES
# ============================================================
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# OpenCV untuk preprocessing sidik jari
import cv2

# Disable warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# ============================================================
# KONFIGURASI DASAR
# ============================================================
# Ukuran input gambar
IMG_SIZE = 128

# Batch size untuk training
BATCH_SIZE = 16

# Jumlah epoch maksimal
EPOCHS = 30

# Learning rate
LEARNING_RATE = 0.0001

# Paths
DATASET_DIR = Path("dataset")
MODELS_DIR = Path("models")
FACE_DIR = DATASET_DIR / "wajah"
FINGER_DIR = DATASET_DIR / "jari"

# Confidence threshold untuk prediksi
CONF_THRESHOLD = 0.85

print("=" * 60)
print("MULTIMODAL BIOMETRIC TRAINING SYSTEM")
print("=" * 60)
print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning Rate: {LEARNING_RATE}")
print("=" * 60)


# ============================================================
# PREPROCESSING FUNCTIONS
# ============================================================

def enhance_fingerprint(image):
    """
    ============================================================
    FINGERPRINT ENHANCEMENT FUNCTION
    ============================================================
    Meningkatkan kontras pola sidik jari menggunakan:
    - Konversi ke uint8 (jika float)
    - Grayscale conversion
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Binarization (thresholding)
    
    Args:
        image: Input gambar (numpy array, bisa float32 dari ImageDataGenerator)
    
    Returns:
        Gambar yang sudah di-enhance dalam format RGB
    """
    # Konversi ke uint8 jika dalam format float
    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Deteksi jumlah channel dan konversi ke grayscale
    if len(image.shape) == 3:
        # Cek jumlah channel (3 = RGB/RGBA, 1 = Grayscale)
        if image.shape[2] == 3:
            # Konversi RGB ke Grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 4:
            # Konversi RGBA ke Grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            # Jika channel bukan 3 atau 4,ambil channel pertama
            gray = image[:, :, 0]
    elif len(image.shape) == 2:
        # Sudah grayscale
        gray = image
    else:
        # Format tidak dikenal, kembalikan original
        return image
    
    # CLAHE untuk meningkatkan kontras lokal
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Gaussian blur untuk reduce noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Adaptive thresholding untuk binarisasi
    binary = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 
        11, 
        2
    )
    
    # Konversi kembali ke RGB untuk kompatibilitas dengan model
    result = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    
    # Konversi ke float32 untuk kompatibilitas dengan ImageDataGenerator
    # Rescale 1./255 akan diterapkan setelah ini oleh ImageDataGenerator
    result = result.astype('float32')
    
    return result


def preprocess_face(image):
    """
    ============================================================
    FACE PREPROCESSING FUNCTION
    ============================================================
    Preprocessing dasar untuk wajah:
    - Rescale saja ( augmentation di ImageDataGenerator )
    
    Args:
        image: Input gambar
    
    Returns:
        Gambar yang sudah diproses
    """
    # Face tidak perlu enhancement ekstra
    # Proses ada di ImageDataGenerator
    return image


# ============================================================
# DATA AUGMENTATION CONFIGURATIONS
# ============================================================

# ImageDataGenerator untuk wajah
# Menggunakan rotasi, zoom, dan flip untuk augmentasi
face_datagen = ImageDataGenerator(
    rescale=1./255,              # Normalisasi pixel ke 0-1
    rotation_range=20,           # Rotasi acak 20 derajat
    zoom_range=0.2,              # Zoom acak 20%
    horizontal_flip=True,        # Flip horizontal
    fill_mode='nearest',         # Mode isian pixel
    validation_split=0.2         # 20% untuk validasi
)

# ImageDataGenerator untuk sidik jari
# Dengan preprocessing function untuk enhancement
finger_datagen = ImageDataGenerator(
    rescale=1./255,              # Normalisasi pixel ke 0-1
    preprocessing_function=enhance_fingerprint,  # Enhancement khusus jari
    rotation_range=15,            # Rotasi lebih kecil untuk jari
    zoom_range=0.15,              # Zoom lebih kecil
    horizontal_flip=False,       # Tidak flip untuk jari (bermakna)
    fill_mode='nearest',
    validation_split=0.2
)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(data_dir, datagen, subset='training'):
    """
    Memuat dataset dari folder menggunakan flow_from_directory
    
    Args:
        data_dir: Path ke folder dataset
        datagen: ImageDataGenerator object
        subset: 'training' atau 'validation'
    
    Returns:
        generator data, jumlah kelas, dan nama kelas
    """
    # Resolve path absolut
    abs_path = data_dir.resolve()
    
    # Cek jika folder ada
    if not data_dir.exists():
        raise FileNotFoundError(f"Folder tidak ditemukan: {abs_path}")
    
    # Hitung jumlah kelas (subdirectories)
    classes = [d.name for d in data_dir.iterdir() if d.is_dir()]
    num_classes = len(classes)
    
    if num_classes == 0:
        raise ValueError(f"Tidak ada subfolder di: {abs_path}")
    
    print(f"\n[INFO] Folder: {abs_path}")
    print(f"[INFO] - {num_classes} kelas ditemukan:")
    for c in sorted(classes):
        print(f"  - {c}")
    
    # Create generator
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset=subset,
        shuffle=True
    )
    
    # Verifikasi generator punya data
    if generator.samples == 0:
        raise ValueError(
            f"\n[ERROR] Generator kosong!\n"
            f"Path: {abs_path}\n"
            f"Subset: {subset}\n"
            f"Pastikan gambar ada di subfolder masing-masing kelas."
        )
    
    return generator, num_classes, sorted(classes)


# ============================================================
# BUILD MODEL FUNCTIONS
# ============================================================

def build_face_model(num_classes):
    """
    ============================================================
    BUILD FACE MODEL
    ============================================================
    Membangun model untuk wajah menggunakan MobileNetV2
    
    Args:
        num_classes: Jumlah kelas (orang)
    
    Returns:
        Model Keras yang sudah dikompilasi
    """
    print("\n[INFO] Membangun Face Model menggunakan MobileNetV2...")
    
    # Load MobileNetV2 sebagai base model
    # weights='imagenet' - menggunakan bobot pretrained
    # include_top=False - tidak include lapisan atas
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model (training hanya di lapisan baru)
    base_model.trainable = False
    
    # Bangun arsitektur baru di atas base model
    x = base_model.output
    
    # Global Average Pooling untuk mengurangi dimensi
    x = GlobalAveragePooling2D()(x)
    
    # Dense layer dengan ReLU activation
    x = Dense(256, activation='relu')(x)
    
    # Dropout untuk mencegah overfitting
    x = Dropout(0.6)(x)
    
    # Output layer dengan softmax untuk multiclass classification
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Buat model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"[INFO] Face Model berhasil dibuat!")
    print(f"  - Base Model: MobileNetV2 (frozen)")
    print(f"  - Trainable layers: {len(model.trainable_variables)}")
    
    return model


def build_fingerprint_model(num_classes):
    """
    ============================================================
    BUILD FINGERPRINT MODEL
    ============================================================
    Membangun model untuk sidik jari menggunakan MobileNetV2
    
    Args:
        num_classes: Jumlah kelas (orang)
    
    Returns:
        Model Keras yang sudah dikompilasi
    """
    print("\n[INFO] Membangun Fingerprint Model menggunakan MobileNetV2...")
    
    # Load MobileNetV2 sebagai base model
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Bangun arsitektur baru
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.6)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Buat model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"[INFO] Fingerprint Model berhasil dibuat!")
    print(f"  - Base Model: MobileNetV2 (frozen)")
    print(f"  - Preprocessing: CLAHE + Binarization")
    
    return model


# ============================================================
# TRAINING FUNCTION
# ============================================================

def calculate_class_weights(generator):
    """
    Menghitung class weights secara manual
    
    Args:
        generator: Data generator
    
    Returns:
        Dictionary of class weights
    """
    from collections import Counter
    
    # Cek apakah generator punya data
    if generator.samples == 0:
        raise ValueError(
            f"[ERROR] Generator tidak punya data!\n"
            f"Samples: {generator.samples}\n"
            f"Pastikan dataset sudah terisi dengan benar."
        )
    
    # Ambil semua label dari generator
    labels = generator.classes
    all_labels = list(labels)
    class_counts = Counter(all_labels)
    total_samples = len(all_labels)
    num_classes = len(class_counts)
    
    print(f"\n[INFO] Class distribution:")
    for class_idx, class_name in generator.class_indices.items():
        count = class_counts.get(class_idx, 0)
        print(f"  - {class_name}: {count} samples")
    
    # Hitung weights: total_samples / (num_classes * class_count)
    class_weights = {}
    for class_idx, count in class_counts.items():
        if count > 0:
            weight = total_samples / (num_classes * count)
            class_weights[class_idx] = weight
        else:
            class_weights[class_idx] = 1.0
    
    print(f"\n[INFO] Class weights:")
    for class_idx, class_name in generator.class_indices.items():
        w = class_weights.get(class_idx, 1.0)
        print(f"  - {class_name}: {w:.2f}")
    
    return class_weights


def train_model(model, train_generator, val_generator, model_name):
    """
    Melatih model dengan early stopping dan class weights
    
    Args:
        model: Model Keras
        train_generator: Generator untuk training data
        val_generator: Generator untuk validation data
        model_name: Nama model (untuk display)
    
    Returns:
        History object dari training
    """
    print(f"\n[INFO] Memulai training untuk {model_name}...")
    print(f"  - Training samples: {train_generator.samples}")
    print(f"  - Validation samples: {val_generator.samples}")
    print(f"  - Steps per epoch: {train_generator.samples // BATCH_SIZE}")
    
    # Hitung class weights untuk mengatasi imbalance
    class_weights = calculate_class_weights(train_generator)
    
    # Early Stopping callback
    # Monitor val_loss, berhenti jika tidak ada perbaikan selama 5 epoch
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # Training model dengan class weights
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[early_stop],
        class_weight=class_weights,
        verbose=1
    )
    
    return history


# ============================================================
# PLOTTING FUNCTION
# ============================================================

def plot_training_history(history, model_name, save_path):
    """
    Menampilkan dan menyimpan grafik training
    
    Args:
        history: History object dari model.fit()
        model_name: Nama model
        save_path: Path untuk menyimpan plot
    """
    # Buat figure dengan 2 subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Title
    fig.suptitle(f'{model_name} - Training Results', fontsize=14, fontweight='bold')
    
    # Plot Accuracy
    ax1.plot(history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=12)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Plot Loss
    ax2.plot(history.history['loss'], 'b-', label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=12)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Plot disimpan ke: {save_path}")
    
    # Show plot
    plt.show()


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():
    """
    Fungsi utama untuk melatih kedua model
    """
    print("\n" + "=" * 60)
    print("MEMULAI PROSES TRAINING")
    print("=" * 60)
    
    # ============================================================
    # CEK FOLDER DATASET
    # ============================================================
    
    # Cek folder wajah
    face_classes = [d.name for d in FACE_DIR.iterdir() if d.is_dir()]
    finger_classes = [d.name for d in FINGER_DIR.iterdir() if d.is_dir()]
    
    if not face_classes:
        raise FileNotFoundError(
            f"\n[ERROR] Folder wajah kosong!\n"
            f" Pastikan folder '{FACE_DIR}' berisi subfolder dengan nama/user.\n"
            f" Contoh: dataset/wajah/daud/"
        )
    
    if not finger_classes:
        raise FileNotFoundError(
            f"\n[ERROR] Folder sidik jari kosong!\n"
            f" Pastikan folder '{FINGER_DIR}' berisi subfolder dengan nama/user.\n"
            f" Contoh: dataset/jari/daud/"
        )
    
    # ============================================================
    # LOAD FACE DATASET
    # ============================================================
    
    print("\n" + "-" * 40)
    print("LOADING FACE DATASET")
    print("-" * 40)
    
    # Training data
    face_train, face_classes_num, face_labels = load_dataset(
        FACE_DIR, 
        face_datagen, 
        'training'
    )
    
    # Validation data
    face_val, _, _ = load_dataset(
        FACE_DIR, 
        face_datagen, 
        'validation'
    )
    
    # ============================================================
    # LOAD FINGERPRINT DATASET
    # ============================================================
    
    print("\n" + "-" * 40)
    print("LOADING FINGERPRINT DATASET")
    print("-" * 40)
    
    # Training data
    finger_train, finger_classes_num, finger_labels = load_dataset(
        FINGER_DIR, 
        finger_datagen, 
        'training'
    )
    
    # Validation data
    finger_val, _, _ = load_dataset(
        FINGER_DIR, 
        finger_datagen, 
        'validation'
    )
    
    # ============================================================
    # VALIDASI KONSISTENSI KELAS
    # ============================================================
    
    print("\n" + "-" * 40)
    print("VALIDATING CLASS CONSISTENCY")
    print("-" * 40)
    
    # Cek apakah jumlah kelas sama
    if face_classes_num != finger_classes_num:
        raise ValueError(
            f"\n[ERROR] Jumlah kelas tidak konsisten!\n"
            f"  Wajah: {face_classes_num} kelas\n"
            f"  Jari: {finger_classes_num} kelas\n"
            f" Pastikan kedua folder memiliki kelas yang sama."
        )
    
    # Cek apakah nama kelas sama
    if face_labels != finger_labels:
        print("\n[WARNING] Nama kelas tidak persis sama!")
        print(f"  Wajah: {face_labels}")
        print(f"  Jari: {finger_labels}")
        print("[INFO] Menggunakanunion dari kedua list...")
    
    # Gunakan face labels sebagai referensi
    num_classes = face_classes_num
    class_labels = face_labels
    
    print(f"\n[INFO] Total {num_classes} kelas untuk training")
    
    # ============================================================
    # SAVE CLASS LABELS
    # ============================================================
    
    labels_path = MODELS_DIR / "labels.txt"
    with open(labels_path, 'w') as f:
        for i, label in enumerate(class_labels):
            f.write(f"{i}: {label}\n")
    
    print(f"[INFO] Labels disimpan ke: {labels_path}")
    
    # ============================================================
    # BUILD MODELS
    # ============================================================
    
    print("\n" + "=" * 60)
    print("BUILDING MODELS")
    print("=" * 60)
    
    # Build Face Model
    face_model = build_face_model(num_classes)
    face_model.summary()
    
    # Build Fingerprint Model
    finger_model = build_fingerprint_model(num_classes)
    finger_model.summary()
    
    # ============================================================
    # TRAIN FACE MODEL
    # ============================================================
    
    print("\n" + "=" * 60)
    print("TRAINING FACE MODEL")
    print("=" * 60)
    
    face_history = train_model(
        face_model,
        face_train,
        face_val,
        "Face Model"
    )
    
    # Simpan Face Model
    face_model_path = MODELS_DIR / "face_model.keras"
    face_model.save(face_model_path)
    print(f"[INFO] Face Model disimpan ke: {face_model_path}")
    
    # Plot Face Model
    plot_training_history(
        face_history,
        "Face Model",
        MODELS_DIR / "face_model_history.png"
    )
    
    # ============================================================
    # TRAIN FINGERPRINT MODEL
    # ============================================================
    
    print("\n" + "=" * 60)
    print("TRAINING FINGERPRINT MODEL")
    print("=" * 60)
    
    finger_history = train_model(
        finger_model,
        finger_train,
        finger_val,
        "Fingerprint Model"
    )
    
    # Simpan Fingerprint Model
    finger_model_path = MODELS_DIR / "finger_model.keras"
    finger_model.save(finger_model_path)
    print(f"[INFO] Fingerprint Model disimpan ke: {finger_model_path}")
    
    # Plot Fingerprint Model
    plot_training_history(
        finger_history,
        "Fingerprint Model",
        MODELS_DIR / "finger_model_history.png"
    )
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n[SUCCESS] Model berhasil dilatih dan disimpan:")
    print(f"  - Face Model: {face_model_path}")
    print(f"  - Fingerprint Model: {finger_model_path}")
    print(f"  - Labels: {labels_path}")
    print(f"\n[INFO] Plots disimpan:")
    print(f"  - {MODELS_DIR / 'face_model_history.png'}")
    print(f"  - {MODELS_DIR / 'finger_model_history.png'}")
    print("\n" + "=" * 60)


# ============================================================
# EXECUTE MAIN
# ============================================================

if __name__ == "__main__":
    # Jalankan training
    main()