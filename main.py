import os
import torch
import requests

from app_utils import define_model, inference

from fastapi import FastAPI, Request, UploadFile, HTTPException
from fastapi.responses import Response

import uvicorn
import uuid

from logger import log
from pydantic import BaseModel

app = FastAPI()

class ArgsModel(BaseModel):
    task: str = 'classical_sr'
    scale: int = 2
    training_patch_size: int = 48
    large_model: bool = True
    model_path : str = 'model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth'
    tile: int = None
    tile_overlap: int = 32
    
args = ArgsModel()

class UnicornException(Exception):
    def __init__(self, response: dict):
        self.response = response

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if os.path.exists(args.model_path):
    print(f'loading model from {args.model_path}')
else:
    print(f'Downloading model {args.model_path}')
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}'.format(os.path.basename(args.model_path))
    r = requests.get(url, allow_redirects=True)
    open(args.model_path, 'wb').write(r.content)

model = define_model(args)
model.eval()
model = model.to(device)


def get_response(client_addr, request_id, args, files):

    status_code, results, success, time, error = inference(files, args, model, device)
    log_type = 'INFO' if success else 'ERROR'
    logs = log(log_type, status_code, client_addr, request_id, success, time, error)
    return results, success, logs
    

@app.post('/upscale', tags=['Upscaling'])
async def run_model(args: ArgsModel, files: list[UploadFile], request: Request):
    #request.client.host
    results, success, logs = get_response('foo', str(uuid.uuid4()), args, files)
    
    if not success:
        raise UnicornException({'logs': logs})

    boundary = "my_boundary"
    multipart_body = b""

    for idx, img_bytes in enumerate(results):
        headers = (
            f"--{boundary}\r\n"
            f"Content-Type: image/png\r\n"
            f"Content-Disposition: inline; filename=image_{idx}.png\r\n\r\n"
        )
        multipart_body += headers.encode() + img_bytes + b"\r\n"

    multipart_body += f"--{boundary}--\r\n".encode()

    return Response(content=multipart_body, media_type=f"multipart/mixed; boundary={boundary}")