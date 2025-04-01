from models.network_swinir import SwinIR as net
import torch
import time
import cv2
import numpy as np
from main_test_swinir import test

def define_model(args):
    # 001 classical image sr
    if args.task == 'classical_sr':
        model = net(upscale=args.scale, in_chans=3, img_size=args.training_patch_size, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
        param_key_g = 'params'

    # 002 lightweight image sr
    # use 'pixelshuffledirect' to save parameters
    elif args.task == 'lightweight_sr':
        model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
        param_key_g = 'params'

    # 003 real-world image sr
    elif args.task == 'real_sr':
        if not args.large_model:
            # use 'nearest+conv' to avoid block artifacts
            model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
        else:
            # larger model size; use '3conv' to save parameters and memory; use ema for GAN training
            model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
        param_key_g = 'params_ema'

    # 004 grayscale image denoising
    elif args.task == 'gray_dn':
        model = net(upscale=1, in_chans=1, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    # 005 color image denoising
    elif args.task == 'color_dn':
        model = net(upscale=1, in_chans=3, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    # 006 grayscale JPEG compression artifact reduction
    # use window_size=7 because JPEG encoding uses 8x8; use img_range=255 because it's sligtly better than 1
    elif args.task == 'jpeg_car':
        model = net(upscale=1, in_chans=1, img_size=126, window_size=7,
                    img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    # 006 color JPEG compression artifact reduction
    # use window_size=7 because JPEG encoding uses 8x8; use img_range=255 because it's sligtly better than 1
    elif args.task == 'color_jpeg_car':
        model = net(upscale=1, in_chans=3, img_size=126, window_size=7,
                    img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

    return model

def inference(files, args, model, device):
    
    start = time.time()
    image_results = []
    code = 200
    try:
        window_size = 8
        for file in files:
            # read image
            file_bytes = file.file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            img_lq = cv2.imdecode(nparr, cv2.IMREAD_COLOR).astype(np.float32) / 255
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
            img_bytes = buffer.tobytes()
            
            image_results.append(img_bytes)
        delay = time.time() - start
        return code, image_results, True, delay, ''
    except Exception as e:
        code = 500
        delay = time.time() - start
        return code, None, False, delay, str(e)
        