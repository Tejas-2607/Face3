import requests, os

url = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcface-resnet100-8.onnx"
dest = "arcface_r100.onnx"

print("Downloading arcface_r100.onnx ...")
with requests.get(url, stream=True) as r:
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    done = 0
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done//1024//1024}/{total//1024//1024} MB", end="")
print(f"\nDone! {os.path.getsize(dest)//1024//1024} MB saved to {dest}")