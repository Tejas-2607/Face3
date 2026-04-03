# download_model.py
# Run this once: python download_model.py

from huggingface_hub import hf_hub_download
import shutil, os

print("Downloading yolov11n-face.pt from Hugging Face...")
path = hf_hub_download(
    repo_id="AdamCodd/YOLOv11n-face-detection",
    filename="model.pt"
)
dest = os.path.join(os.path.dirname(__file__), "yolov11n-face.pt")
shutil.copy(path, dest)
print(f"Saved to: {dest}")