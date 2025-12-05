import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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
    first_img = first_img.resize((width, height))
    if last_img:
        last_img = last_img.resize((width, height))
    frames = []
    frames.append(first_img)
    # Ideally, this should be 127.5 to match original code, but they perform computation on numpy arrays
    # whereas we are passing PIL images. If you choose to pass numpy arrays, you can set it to 127.5 to
    # match the original code.
    num_pad_frames = num_frames - (2 if last_img else 1)
    frames.extend([PIL.Image.new("RGB", (width, height), (128, 128, 128))] * num_pad_frames)
    mask_black = PIL.Image.new("L", (width, height), 0)
    mask_white = PIL.Image.new("L", (width, height), 255)
    mask = [mask_black, *[mask_white] * num_pad_frames]
    if last_img:
        frames.append(last_img)
        mask.append(mask_black)
    return frames, mask


# Available checkpoints: Wan-AI/Wan2.1-VACE-1.3B-diffusers, Wan-AI/Wan2.1-VACE-14B-diffusers
model_id = "/home/jibaixu/Models/Wan/Wan2.1-VACE-1.3B-diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanVACEPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
flow_shift = 3.0  # 5.0 for 720P, 3.0 for 480P
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
pipe.scheduler = FlowMatchScheduler(shift=flow_shift, sigma_min=0.0, extra_one_step=True)
pipe.to("cuda")

prompt = "This is a robotic arm operation scenario. On the left is a Panda model robotic arm, which moves to the table on the right to grab a banana."
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
first_frame = load_image(
    "output.png"
)
last_frame = None

# size = [192*320, 480*832]
height = 480
width = 832
num_frames = 81
video, mask = prepare_video_and_mask(first_frame, last_frame, height, width, num_frames)

# with attention_backend("flash"):
# with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]):
output = pipe(
    video=video,
    mask=mask,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height,
    width=width,
    num_frames=num_frames,
    num_inference_steps=30,
    guidance_scale=5.0,
    generator=torch.Generator().manual_seed(42),
).frames[0]

export_to_video(output, f"output_robot_{height}_{width}.mp4", fps=16)
