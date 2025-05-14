import subprocess
from datetime import datetime

def otomatik_git_push():
    # Commit mesajı
    zaman = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mesaj = f"Otomatik commit: {zaman}"

    try:
        # Git komutlarını çalıştır
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", mesaj], check=True)
        subprocess.run(["git", "push"], check=True)
        print("✅ Değişiklikler otomatik olarak gönderildi.")
    except subprocess.CalledProcessError as e:
        print("❌ Bir hata oluştu:", e)

if __name__ == "__main__":
    otomatik_git_push()
