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

## Inference

```
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

## Input data


## Output data


<!-- | Training Set | quality factor | PSNR (RGB) | PSNR-B (RGB) | SSIM (RGB) |
|:-------------|:--------------:|:----------:|:------------:|:----------:|
| LIVE1        |       10       |   28.06    |    27.76     |   0.8089   |
| LIVE1        |       20       |   30.45    |    29.97     |   0.8741   |
| LIVE1        |       30       |   31.82    |    31.24     |   0.9018   |
| LIVE1        |       40       |   32.75    |    32.12     |   0.9174   |
</details> -->

