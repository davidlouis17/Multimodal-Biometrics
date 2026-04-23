"""
Face Dataset Collection Script
Mengambil 100 sampel foto wajah secara otomatis menggunakan OpenCV

Spesifikasi:
- Face Detection: Haar Cascade frontal face
- Preprocessing: Grayscale, crop ROI, resize 128x128
- Automation: Delay 100ms per deteksi
- Visual Feedback: Rectangle di wajah + counter
- Save: JPG format
"""

import cv2
import os
import time

def main():
    # Input nama user
    user_name = input("Masukkan nama user (contoh: daud): ").strip()
    if not user_name:
        print("Nama user tidak boleh kosong!")
        return

    # Path folder dataset
    dataset_path = os.path.join("dataset", "wajah", user_name)
    os.makedirs(dataset_path, exist_ok=True)
    print(f"Folder dataset: {dataset_path}")

    # Load Haar Cascade
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    if face_cascade.empty():
        print("Error: Haar cascade tidak ditemukan!")
        return

    # Buka webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Webcam tidak bisa dibuka!")
        return

    print("Tekan 'q' untuk keluar sebelum mencapai 100 foto")
    print("Program akan otomatis mengambil foto saat wajah terdeteksi")

    counter = 0
    last_capture_time = 0
    delay_ms = 100  # 100ms delay

    while counter < 100:
        ret, frame = cap.read()
        if not ret:
            print("Error: Gagal membaca frame!")
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

        face_detected = False

        for (x, y, w, h) in faces:
            # Gambar rectangle di wajah
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face_detected = True

            # Cek delay
            current_time = time.time() * 1000  # ms
            if current_time - last_capture_time > delay_ms:
                # Crop ROI wajah
                roi = gray[y:y+h, x:x+w]

                # Resize ke 128x128
                roi_resized = cv2.resize(roi, (128, 128))

                # Simpan gambar
                filename = f"{user_name}_{counter+1:03d}.jpg"
                filepath = os.path.join(dataset_path, filename)
                cv2.imwrite(filepath, roi_resized)

                counter += 1
                last_capture_time = current_time
                print(f"Foto {counter}/100 disimpan: {filename}")
                break  # Ambil satu wajah per frame

        # Tampilkan counter
        cv2.putText(frame, f"Samples: {counter}/100", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"User: {user_name}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if face_detected:
            cv2.putText(frame, "Face Detected - Capturing...", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Face Detected", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Tampilkan frame
        cv2.imshow("Face Dataset Collection", frame)

        # Cek key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Dihentikan oleh user")
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    print(f"\nKoleksi selesai! {counter} sampel foto disimpan di {dataset_path}")

if __name__ == "__main__":
    main()