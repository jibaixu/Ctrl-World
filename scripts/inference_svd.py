import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "/home/jibaixu/Codes/Ctrl-World/ckpts/svd", torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()

# Load the conditioning image
image = load_image("output.png")
image = image.resize((320, 192))

generator = torch.manual_seed(42)
frames = pipe(image, num_frames=35, decode_chunk_size=8, generator=generator).frames[0]

export_to_video(frames, "generated.mp4", fps=7)
