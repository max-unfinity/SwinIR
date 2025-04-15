# FastAPI service: SwinIR Upscaling


## Start

build image:
```bash
docker build -t fastapi-swinir .
```

run container:
```bash
docker run -d -p 8000:8000 fastapi-swinir 
```


## Input data

### Args:
     
     task: str
     scale: int
     large_model: bool
     model_path : str
     tile: int
     tile_overlap: int
### Files:
     list[UploadFile] - bytes format

## Output data

     batch of bytes
     
## Inference

```python
import requests

url = "http://localhost:8000/upscale"
files = {"files": open("input.png", "rb")}
response = requests.post(url, files=files)

boundary = response.headers["Content-Type"].split("boundary=")[-1]
parts = response.content.split(f"--{boundary}".encode())

for idx, part in enumerate(parts):
    if b"Content-Type: image/png" in part:
        img_data = part.split(b"\r\n\r\n")[1].strip()
        with open(f"output_{idx}.png", "wb") as f:
            f.write(img_data)

```

