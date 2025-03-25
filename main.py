import cv2
import glob
import numpy as np
import os
import torch
import requests
import base64
import time

from models.network_swinir import SwinIR as net
from utils import util_calculate_psnr_ssim as util
from main_test_swinir import get_image_pair, test
from app_utils import define_model, setup

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

import uvicorn
import uuid

from logger import log
from pydantic import BaseModel

app = FastAPI()

class ArgsModel(BaseModel):
    task: str = 'classical_sr'
    scale: int = 2
    noise: int = 15
    jpeg: int = 40
    training_patch_size: int = 48
    large_model: bool = True
    model_path : str = 'model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth'
    folder_lq: str = 'testsets/Set5/LR_bicubic/X2'
    folder_gt: str = 'testsets/Set5/HR'
    tile: int = None
    tile_overlap: int = 32
    save_jpg: bool = True
    
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

def inference(args):
    
    start = time.time()
    image_results = []
    code = 200
    try:
        folder, save_dir, border, window_size = setup(args)
        for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
            image_result = {}
            # read image
            imgname, img_lq, img_gt = get_image_pair(args, path)  # image to HWC-BGR, float32
            img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
            img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

            # inference
            with torch.no_grad():
                # pad input image to be a multiple of window_size
                _, _, h_old, w_old = img_lq.size()
                h_pad = (h_old // window_size + 1) * window_size - h_old
                w_pad = (w_old // window_size + 1) * window_size - w_old
                img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
                img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
                output = test(img_lq, model, args, window_size)
                output = output[..., :h_old * args.scale, :w_old * args.scale]

            # save image
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            if output.ndim == 3:
                output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
            output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        
            _, buffer = cv2.imencode(".jpg", output)
            img_base64 = base64.b64encode(buffer).decode("utf-8")

            image_result['index'] = idx
            image_result['base64'] = img_base64
            
            image_results.append(image_result)
        delay = time.time() - start
        return code, image_results, True, delay, ''
    except Exception as e:
        code = 500
        delay = time.time() - start
        return code, None, False, delay, str(e)
        


def get_response(client_addr, request_id, input):

    status_code, results, success, time, error = inference(input)
    log_type = 'INFO' if success else 'ERROR'
    logs = log(log_type, status_code, client_addr, request_id, success, time, error)
    return {
        'results': results,
        'logs': logs
    }


@app.exception_handler(UnicornException)
async def unicorn_exception_handler(request: Request, exc: UnicornException):
    return JSONResponse(
        status_code=exc.response['logs']['status_code'],
        content=exc.response,
    )
    

@app.post('/run', tags=['Run model'])
async def run_model(args: ArgsModel, request: Request):
    response = get_response(request.client.host, str(uuid.uuid4()), args)
    raise UnicornException(response)