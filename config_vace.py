from typing import List
from dataclasses import dataclass, field


@dataclass
class vace_args:
    # -- model
    vace_model_path: str = "/data2/jibaixu/Models/Wan-AI/Wan2.1-VACE-1.3B-diffusers"
    use_lora: bool = False

    # -- dataset
    dataset_root_path: str = "/data2/jibaixu/Codes/Ctrl-World/dataset_example"
    dataset_names: str = "droid_vace_5hz_192_320_latent"
    probs: List[float] = field(default_factory=lambda: [1.0])  # probability to sample from each dataset, should sum to 1.0
    shuffle: bool = True
    num_cams: int = 2   # The number of cameras in one trajectory.

    # -- training
    learning_rate: float = 1e-4
    train_batch_size: int = 1
    max_train_steps: int = 50_000
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    checkpointing_steps: int = 2_000
    validation_steps: int = 250

    # -- log
    sft_tag = "lora" if use_lora else "full"
    output_dir: str = f"output/{dataset_names}_{sft_tag}_lr_{str(learning_rate)}_grad_{gradient_accumulation_steps}"
    wandb_project_name: str = "diffusers_wan_vace"
    wandb_run_name: str = f"{dataset_names}_{sft_tag}_lr_{str(learning_rate)}_grad_{gradient_accumulation_steps}"
