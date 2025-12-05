import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
from typing import Union, List

import torch
import torch.nn as nn
from diffusers.models import AutoencoderKLWan

from models.pipeline_wan_vace import WanVACEPipeline
from models.transformer_wan_vace import WanVACETransformer3DModel
from models.flow_match import FlowMatchScheduler


class WanVACE(nn.Module):
    def __init__(
            self,
            vace_model_path: str,
            num_training_timesteps: int = 1000,
            seed: int = 42,
            use_lora: bool = True,
            transformer_dtype: torch.dtype = torch.bfloat16
        ):
        super().__init__()

        vae = AutoencoderKLWan.from_pretrained(vace_model_path, subfolder="vae", torch_dtype=torch.float32)
        self.pipeline = WanVACEPipeline.from_pretrained(vace_model_path, vae=vae, torch_dtype=transformer_dtype)
        print("Replace the transformer of WanVACEPipeline ...")
        self.pipeline.transformer = WanVACETransformer3DModel.from_pretrained(os.path.join(vace_model_path, "transformer"), torch_dtype=transformer_dtype)
        
        self.vae = self.pipeline.vae    # 1014MB
        self.text_encoder = self.pipeline.text_encoder  # 11342MB
        self.tokenizer = self.pipeline.tokenizer
        self.transformer = self.pipeline.transformer    # 4972MB
        self.scheduler = FlowMatchScheduler(sigma_min=0.0, extra_one_step=True)

        # Freeze VAE and text encoder
        self.vae.requires_grad(False)
        self.text_encoder.requires_grad(False)
        if not use_lora:
            self.transformer.requires_grad_(True)
        else:
            self._enable_transformer_lora()

        # Training Parameters
        self.num_training_timesteps = num_training_timesteps
        self.training_weights = self._get_training_weights_offline(num_training_timesteps)

        self.generator = torch.Generator().manual_seed(seed)

    def _enable_transformer_lora(
            self,
            lora_rank: int = 32,
            lora_alpha: int = 32,
            lora_target_modules: Union[str, list[str]] = ".*vace_blocks.*\.(to_q|to_k|to_v|to_out\.0|ffn\.net\.0\.proj|ffn\.net\.2)$",
        ):
        """
        lora_target_modules: 当输入str允许正则化匹配; 当输入list[str]只判定是否以该精确字符串为结尾
        """
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise ImportError(
                "LoRA finetuning requires the `peft` package. Install it with `pip install peft`."
            ) from exc

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            bias="none",
        )

        self.transformer.requires_grad_(False)
        self.transformer = get_peft_model(self.transformer, lora_config)

        print(f"Trainable parameters in VACE Transformer:")
        self.transformer.print_trainable_parameters()

    def _get_training_weights_offline(self, num_training_timesteps):
        # Construct training weight offline
        sigmas = torch.linspace(1.0, 0.0, num_training_timesteps + 1)[:-1]
        timesteps = sigmas * num_training_timesteps

        # Calculate the training weights on the timesteps according to the Gaussian distribution,
        # with higher weights in the middle and lower weights at the ends.
        x = timesteps
        y = torch.exp(-2 * ((x - num_training_timesteps / 2) / num_training_timesteps) ** 2)
        y_shifted = y - y.min()
        linear_timesteps_weights = y_shifted * (num_training_timesteps / y_shifted.sum())
        return linear_timesteps_weights

    def forward(
            self,
            batch,
            conditioning_scale: Union[float, List[float], torch.Tensor] = 1.0,
        ):
        device = self.transformer.device
        transformer_dtype = self.transformer.dtype

        latents = batch['latents'].to(dtype=transformer_dtype, device=device)
        conditioning_latents = batch['condition_latents'].to(dtype=transformer_dtype, device=device)   # [B, 32+patch_h*patch_w, num_ref_imgs+T/4, H/8, W/8]
        prompt_embeds = batch['prompt_embeds'].to(dtype=transformer_dtype, device=device)
        
        # Construct conditioning_scale dtype and device
        if isinstance(conditioning_scale, (int, float)):
            conditioning_scale = [conditioning_scale] * len(self.transformer.config.vace_layers)
        if isinstance(conditioning_scale, list):
            if len(conditioning_scale) != len(self.transformer.config.vace_layers):
                raise ValueError(
                    f"Length of `conditioning_scale` {len(conditioning_scale)} does not match number of layers {len(self.transformer.config.vace_layers)}."
                )
            conditioning_scale = torch.tensor(conditioning_scale)
        if isinstance(conditioning_scale, torch.Tensor):
            if conditioning_scale.size(0) != len(self.transformer.config.vace_layers):
                raise ValueError(
                    f"Length of `conditioning_scale` {conditioning_scale.size(0)} does not match number of layers {len(self.transformer.config.vace_layers)}."
                )
            conditioning_scale = conditioning_scale.to(device=device, dtype=transformer_dtype)

        # Construct noisy latents
        sigma = torch.rand(latents.shape[0], 1, 1, 1, 1, generator=self.generator, dtype=transformer_dtype).to(device)
        noise = torch.randn(latents.shape, generator=self.generator, dtype=transformer_dtype).to(device)
        noisy_latents = (1 - sigma) * latents + sigma * noise
        timestep = (sigma * self.num_training_timesteps).reshape(sigma.shape[0]).to(torch.long)

        noise_pred = self.transformer(
            hidden_states=noisy_latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            control_hidden_states=conditioning_latents,
            control_hidden_states_scale=conditioning_scale,
            return_dict=False,
        )[0]

        target = noise - latents

        if self.training_weights.device is not device:
            self.training_weights = self.training_weights.to(device)
        training_weight = self.training_weights[timestep].reshape(sigma.shape[0], 1, 1, 1, 1)
        loss = (training_weight * (noise_pred - target) ** 2).mean()
        return loss


if __name__ == "__main__":
    model = WanVACE(
        vace_model_path = "/home/jibaixu/Models/Wan/Wan2.1-VACE-1.3B-diffusers",
        use_lora=True,
    )
    model.pipeline.save_pretrained(
        "model_ckpt/doird_subset",
        max_shard_size="5GB",
    )
    latents = torch.randn(8, 16, 21, 64, 64)
    condition_latents = torch.randn(8, 96, 21, 64, 64)
    prompt_embeds = torch.randn(8, 512, 4096)
    batch = {
        'latents': latents,
        'condition_latents': condition_latents,
        'prompt_embeds': prompt_embeds,
    }
    model.transformer.to('cuda:0')
    loss = model(batch)
    print(loss)
