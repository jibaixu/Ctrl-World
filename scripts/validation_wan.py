import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from typing import Optional
import random

import torch
import numpy as np
import PIL.Image
from diffusers import AutoencoderKLWan
import mediapy

from models.pipeline_wan_vace import WanVACEPipeline
from models.transformer_wan_vace import WanVACETransformer3DModel
from models.scheduler_flow_match import FlowMatchScheduler
from dataset.dataset_droid_vace import Dataset_mix
from config_vace import vace_args


RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BOLD = '\033[1m'
RESET = '\033[0m'

args = vace_args()
args.output_dir = "output/droid_vace_5hz_192_320_latent_full_lr_1e-05_grad_4_gray_with_ref_img"
ckpt_step = "2500"
val_num_videos = 3
output_dir = f"output/val/{args.output_dir.split('/')[-1]}_{ckpt_step}"

height = 192
width = 320
num_frames = 21

os.makedirs(output_dir, exist_ok=True)
################################## Val Dataset ###################################
mode = 'train'
ds = Dataset_mix(args, mode=mode)

dataset_path = ds.dataset_path_all[0]
val_samples = [random.choice(ds.samples_all[0]) for _ in range(val_num_videos)]
val_prompts = [ds.total_anno_dict[sample]['texts'][0] for sample in val_samples]
val_video_dirs = [os.path.join(dataset_path, "videos", mode, sample) for sample in val_samples]

################################## Model ###################################
vae = AutoencoderKLWan.from_pretrained(args.vace_model_path, subfolder="vae", torch_dtype=torch.float32)
pipe = WanVACEPipeline.from_pretrained(args.vace_model_path, vae=vae, torch_dtype=torch.bfloat16)
pipe.transformer = WanVACETransformer3DModel.from_pretrained(os.path.join(args.output_dir, ckpt_step), torch_dtype=torch.bfloat16)

shift = 3.0
pipe.scheduler = FlowMatchScheduler(shift=shift, sigma_min=0.0, extra_one_step=True)
pipe.to("cuda")


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


for i, video_dir in enumerate(val_video_dirs):
    sample = val_samples[i]
    prompt = val_prompts[i]
    # negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    negative_prompt = None
    vps = [os.path.join(video_dir, "0.mp4")]
    for vp in vps:
        origin_video = mediapy.read_video(vp)
        first_frame = PIL.Image.fromarray(origin_video[0])
        last_frame = None
        video, mask = prepare_video_and_mask(first_frame, last_frame, height, width, num_frames)

        output = pipe(
            video=video,
            mask=mask,
            reference_images=[first_frame],
            prompt=prompt,
            negative_prompt=negative_prompt,
            conditioning_scale=1.0,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=30,
            # guidance_scale=1.0,     # disable classifier-free guidance
            generator=torch.Generator().manual_seed(42),
        ).frames[0]
        gen_video = (output*255).astype(np.uint8)
        concat_video = np.concatenate([origin_video[:num_frames], gen_video], axis=1)

        cam_id = os.path.splitext(os.path.basename(vp))[0]
        output_path = os.path.join(output_dir, f"{sample}_cam{cam_id}_{mode}_shift_{shift}.mp4")
        mediapy.write_video(output_path, concat_video, fps=5)
        print(f"{GREEN}Saved video to {output_path}{RESET}")
