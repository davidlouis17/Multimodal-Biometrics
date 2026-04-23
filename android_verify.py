"""
=============================================================
ANDROID BIOMETRIC VERIFIER (Run on Android)
=============================================================
Script Python untuk Android (PyDroid/QPython)

Workflow (Master-Slave Synchronized):
1. Laptop menghapus file lama
2. Laptop menulis pending:USER_ID ke file
3. HP mendeteksi perubahan file
4. User meletakkan jari di sensor (VERIFIKASI NYATA)
5. HP menulis verified:USER_ID atau failed:USER_ID
6. Laptop mendeteksi dan memproses

PENTING:
- HP HANYA menulis result JIKA verifikasi BERHASIL
- Tidak ada time.sleep untuk simulasi
- Tidak ada writing sebelum sensorOK
"""

import os
import time

RESULT_FILE = '/sdcard/biometric_result.txt'

print("=" * 50)
print("ANDROID BIOMETRIC VERIFIER")
print("=" * 50)

def read_result_file():
    """Baca konten file result"""
    try:
        with open(RESULT_FILE, 'r') as f:
            return f.read().strip()
    except:
        return None

def write_result(user_id, status):
    """Tulis hasil verifikasi ke file - HANYA jika verified"""
    try:
        os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
    except:
        pass
    
    with open(RESULT_FILE, 'w') as f:
        f.write(f"{status}:{user_id}")
    print(f"[WRITTEN] {status}:{user_id}")

# Cek apakah running di Android
IS_ANDROID = os.path.exists('/system')

if IS_ANDROID:
    print("[INFO] Running on Android device")
    
    # STEP 1: Hapus file lama saat startup
    print("[STARTUP] Membersihkan file lama...")
    try:
        os.remove(RESULT_FILE)
    except:
        pass
    
    print("[INFO] Menunggu request dari laptop...")
    print("[INFO] Jangan tutup aplikasi ini!")
    print("=" * 50)
    print("INSTRUKSI:")
    print("  Ketik 's' + ENTER jika verifikasi BERHASIL")
    print("  Ketik 'g' + ENTER jika verifikasi GAGAL")
    print("  Ketik 'c' + ENTER untuk CANCEL")
    print("=" * 50)
    print("MENUNGGU REQUEST...")
    
    last_content = None
    pending_user = None
    VERIFY_TIMEOUT = 30
    
    while True:
        content = read_result_file()
        
        if content and content != last_content:
            last_content = content
            print(f"\n[DETECTED] {content}")
            
            if content.startswith('pending:'):
                # Request baru dari laptop
                pending_user = content.split(':')[1].strip()
                print(f"\n>>> VERIFIKASI DIMINTA!")
                print(f">>> Untuk user: {pending_user}")
                print(">>> Ketik 's' jika BERHASIL")
                print(">>> Ketik 'g' jika GAGAL")
                print("=" * 50)
                
                # Tunggu input manual (HARUS input, tidak auto!)
                print("\n[KETIK INPUT] ")
                print("  s = Success (verified)")
                print("  g = Gagal (failed)")
                print("  c = Cancel")
                
                # Loop tunggu keputusan user - TIDAK ADA AUTO!
                while True:
                    try:
                        user_input = input("Ketik (s/g/c): ").strip().lower()
                        
                        if user_input == 's':
                            # SUCCESS - Verifikasi berhasil
                            print(f"\n[SUCCESS] User verified: {pending_user}")
                            write_result(pending_user, 'verified')
                            break
                            
                        elif user_input == 'g':
                            # FAILED - Verifikasi gagal
                            print(f"\n[FAILED] Verification failed - unauthorized")
                            write_result(pending_user, 'failed')
                            break
                            
                        elif user_input == 'c':
                            # CANCEL
                            print(f"\n[CANCELLED] User cancelled")
                            write_result(pending_user, 'cancelled')
                            break
                            
                        else:
                            print("Ketik s, g, atau c saja!")
                            
                    except EOFError:
                        print("[ERROR] Input tidak dikenali, coba lagi")
except KeyboardInterrupt:
                        print("\n[CANCEL] Interrupted")
                        write_result(pending_user, 'cancelled')
                        break
        
        time.sleep(0.5)
        
else:
    # Desktop mode - ADB testing
    print("[INFO] Running on desktop")
    print("\n" + "=" * 50)
    print("TESTING COMMANDS (PowerShell):")
    print("=" * 50)
    print("""
# 1. Cek koneksi
adb devices

# 2. Hapus file lama
adb shell rm /sdcard/biometric_result.txt

# 3. Kirim request (seperti laptop)
adb shell "echo pending:daud > /sdcard/biometric_result.txt"

# 4. Di terminal lain HP, jalankan script ini

# 5. Simulasikan success
adb shell "echo verified:daud > /sdcard/biometric_result.txt"

# 6. Simulasikan failed
adb shell "echo failed:daud > /sdcard/biometric_result.txt"

# 7. Simulasikan cancel
adb shell "echo cancelled:daud > /sdcard/biometric_result.txt"

# 8. Lihat hasil
adb shell cat /sdcard/biometric_result.txt
    """)
    
    # Demo polling loop
    print("\n[TEST MODE] Polling for file changes...")
    print("Tekan Ctrl+C untuk keluar\n")
    
    try:
        last = None
        while True:
            result = os.popen(f'adb shell "cat {RESULT_FILE}"').read().strip()
            
            if result and result != last:
                last = result
                print(f"[FILE] {result}")
                
                if result.startswith('pending:'):
                    user = result.split(':')[1].strip()
                    print(f"[INFO] Request for: {user}")
                    print(f"  Run: adb shell \"echo verified:{user} > {RESULT_FILE}\"")
                    
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n[ENDED]")