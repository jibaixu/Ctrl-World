import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from typing import Optional

import torch
import PIL.Image
from diffusers import AutoencoderKLWan
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image
from diffusers.models.attention_dispatch import attention_backend
from torch.nn.attention import SDPBackend, sdpa_kernel

from models.pipeline_wan_vace import WanVACEPipeline
# from models.flow_match import FlowMatchScheduler
from models.scheduler_flow_match import FlowMatchScheduler


def prepare_video_and_mask(first_img: PIL.Image.Image, last_img: Optional[PIL.Image.Image], height: int, width: int, num_frames: int):
    if first_img:
        first_img = first_img.resize((width, height))
    if last_img:
        last_img = last_img.resize((width, height))
    frames = []
    if first_img:
        frames.append(first_img)
    # Ideally, this should be 127.5 to match original code, but they perform computation on numpy arrays
    # whereas we are passing PIL images. If you choose to pass numpy arrays, you can set it to 127.5 to
    # match the original code.
    num_pad_frames = num_frames
    if last_img:
        num_pad_frames -= 1
    if first_img:
        num_pad_frames -= 1
    frames.extend([PIL.Image.new("RGB", (width, height), (128, 128, 128))] * num_pad_frames)
    mask_black = PIL.Image.new("L", (width, height), 0)
    mask_white = PIL.Image.new("L", (width, height), 255)
    if first_img:
        mask = [mask_black, *[mask_white] * num_pad_frames]
    else:
        mask = [*[mask_white] * num_pad_frames]
    if last_img:
        frames.append(last_img)
        mask.append(mask_black)
    return frames, mask


# Available checkpoints: Wan-AI/Wan2.1-VACE-1.3B-diffusers, Wan-AI/Wan2.1-VACE-14B-diffusers
model_id = "/data2/jibaixu/Models/Wan-AI/Wan2.1-VACE-1.3B-diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanVACEPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
flow_shift = 3.0  # 5.0 for 720P, 3.0 for 480P
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
pipe.scheduler = FlowMatchScheduler(shift=flow_shift, sigma_min=0.0, extra_one_step=True)
pipe.to("cuda")

prompt = "This is a Tom and Jerry cartoon. The cat lets go of the mouse's tail, and the mouse is flung away."
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
first_frame = load_image(
    "image.png"
)

# size = [192*320, 480*832]
height = 480
width = 832
num_frames = 81
video, mask = prepare_video_and_mask(None, None, height, width, num_frames)

# with attention_backend("flash"):
# with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]):
output = pipe(
    video=video,
    mask=mask,
    reference_images=[first_frame],
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height,
    width=width,
    num_frames=num_frames,
    num_inference_steps=30,
    guidance_scale=5.0,
    generator=torch.Generator().manual_seed(42),
).frames[0]

import numpy as np
import mediapy
gen_video = (output*255).astype(np.uint8)
mediapy.write_video(f"output_cat_{height}_{width}_with_ref_img.mp4", gen_video, fps=16)
