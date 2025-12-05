from typing import List
from dataclasses import dataclass


@dataclass
class vace_args:
    # -- model
    vace_model_path: str = "/home/jibaixu/Models/Wan/Wan2.1-VACE-1.3B-diffusers"
    use_lora: bool = True

    # -- dataset
    dataset_root_path: str = "/home/jibaixu/Codes/Ctrl-World/dataset_example"
    dataset_names: str = "droid_test_vace"
    probs: List[float] = [1.0]  # probability to sample from each dataset, should sum to 1.0
    shuffle: bool = True
    num_cams: int = 2   # The number of cameras in one trajectory.

    # -- training
    learning_rate: float = 1e-4
    num_train_epochs: int = 100
    train_batch_size: int = 1
    max_train_steps: int = 500_000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    mixed_precision: str = "fp16"
    checkpointing_steps: int = 20_000
    validation_steps: int = 2_500

    # -- log
    output_dir: str = "output/droid_subset"
    wandb_project_name: str = "diffusers_wan_vace"
    wandb_run_name: str = "droid_subset"
