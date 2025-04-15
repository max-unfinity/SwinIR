import os
import io
import zipfile
import torch
import requests
from typing import List, Optional
from pathlib import Path

from app_utils import define_model, inference

from fastapi import FastAPI, Request, UploadFile, HTTPException, Depends, File
from fastapi.responses import Response, JSONResponse

import uuid

from logger import log
from pydantic import BaseModel, Field

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = FastAPI()

class UpscaleArgs(BaseModel):
    """Configuration parameters for the upscaling model"""
    task: str = Field(default="classical_sr", description="Type of super-resolution task")
    scale: int = Field(default=2, description="Upscaling factor", ge=1, le=4)
    training_patch_size: int = Field(default=48, description="Training patch size")
    large_model: bool = Field(default=True, description="Whether to use the large model variant")
    model_path: str = Field(
        default="model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth", 
        description="Path to the model weights"
    )
    tile: Optional[int] = Field(default=None, description="Tile size for processing large images")
    tile_overlap: int = Field(default=32, description="Overlap size when using tiles")

def initialize_model(args: UpscaleArgs):
    """Initialize the upscaling model, downloading it if necessary"""
    model_path = Path(args.model_path)
    
    if not model_path.exists():
        print(f'Downloading model {args.model_path}')
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        url = f'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{model_path.name}'
        r = requests.get(url, allow_redirects=True)
        with open(model_path, 'wb') as f:
            f.write(r.content)
    else:
        print(f'Loading model from {args.model_path}')
    
    model = define_model(args)
    model.eval()
    return model.to(device)

class UpscaleError(Exception):
    """Custom exception for upscaling errors"""
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail

@app.exception_handler(UpscaleError)
async def upscale_exception_handler(request: Request, exc: UpscaleError):
    """Handler for upscale errors"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.get("/health", tags=["System"])
async def health_check():
    """Endpoint to check if the server is up and running"""
    return {"status": "ok"}

async def extract_images_from_zip(zip_file: UploadFile) -> List[UploadFile]:
    """Extract images from a zip file and return them as a list of UploadFile objects"""
    content = await zip_file.read()
    zip_bytes = io.BytesIO(content)
    
    extracted_files = []
    with zipfile.ZipFile(zip_bytes) as zip_ref:
        for file_name in zip_ref.namelist():
            ext = os.path.splitext(file_name)[1].lower()
            if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                # Get content type based on extension
                content_type = "image/png"
                if ext == '.jpg' or ext == '.jpeg':
                    content_type = "image/jpeg"
                elif ext == '.bmp':
                    content_type = "image/bmp"
                elif ext == '.tiff':
                    content_type = "image/tiff"
                
                # Create an UploadFile object
                file_content = zip_ref.read(file_name)
                file = UploadFile(
                    filename=os.path.basename(file_name),
                    file=io.BytesIO(file_content),
                    content_type=content_type
                )
                extracted_files.append(file)
    
    if not extracted_files:
        raise HTTPException(status_code=400, detail="No valid images found in zip file")
    
    return extracted_files

@app.post("/upscale/zip", tags=["Upscaling"])
async def upscale_from_zip(
    request: Request,
    zip_file: UploadFile = File(..., description="Zip file containing images to upscale"),
    args: UpscaleArgs = Depends()
):
    """
    Upscale multiple images from a zip file using the SwinIR model.
    
    - Upload a zip file containing multiple images
    - Returns a zip file containing all the upscaled images
    """
    if not zip_file.filename.lower().endswith('.zip'):
        raise HTTPException(status_code=400, detail="Uploaded file must be a zip file")
    
    request_id = str(uuid.uuid4())
    client_addr = request.client.host if request.client else "unknown"

    # Initialize model
    model = initialize_model(args)
    
    try:
        # Extract images from zip
        files = await extract_images_from_zip(zip_file)
        
        # Process the images
        status_code, results, success, time_taken, error = inference(files, args, model, device)
        
        # Log the result
        log_type = "INFO" if success else "ERROR"
        logs = log(log_type, status_code, client_addr, request_id, success, time_taken, error)
        
        if not success:
            raise UpscaleError(status_code=status_code, detail=error)
        
        # Create a zip file with the results
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as result_zip_file:
            for idx, img_bytes in enumerate(results):
                # Use original filename if available
                original_name = os.path.splitext(files[idx].filename)[0] if idx < len(files) else f"image_{idx}"
                filename = f"{original_name}_upscaled.png"
                result_zip_file.writestr(filename, img_bytes)
        
        zip_buffer.seek(0)
        
        return Response(
            content=zip_buffer.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename=upscaled_images.zip"}
        )
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid zip file format")

@app.post("/upscale/multipart", tags=["Upscaling"])
async def upscale_multipart(
    files: List[UploadFile], 
    request: Request,
    args: UpscaleArgs = Depends()
):
    """
    Upscale multiple individual image files using the SwinIR model.
    Returns the upscaled images as a multipart response.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No image files provided")
    
    request_id = str(uuid.uuid4())
    client_addr = request.client.host if request.client else "unknown"
    
    # Initialize model
    model = initialize_model(args)
    
    # Process the images
    status_code, results, success, time_taken, error = inference(files, args, model, device)
    
    # Log the result
    log_type = "INFO" if success else "ERROR"
    logs = log(log_type, status_code, client_addr, request_id, success, time_taken, error)
    
    if not success:
        raise UpscaleError(status_code=status_code, detail=error)

    # Prepare multipart response
    boundary = "image_boundary"
    multipart_body = b""

    for idx, img_bytes in enumerate(results):
        # Use original filename if available, otherwise generate one
        original_name = os.path.splitext(files[idx].filename)[0] if idx < len(files) else f"image_{idx}"
        filename = f"{original_name}_upscaled.png"
        
        headers = (
            f"--{boundary}\r\n"
            f"Content-Type: image/png\r\n"
            f"Content-Disposition: inline; filename={filename}\r\n\r\n"
        )
        multipart_body += headers.encode() + img_bytes + b"\r\n"

    multipart_body += f"--{boundary}--\r\n".encode()

    return Response(
        content=multipart_body, 
        media_type=f"multipart/mixed; boundary={boundary}"
    )
