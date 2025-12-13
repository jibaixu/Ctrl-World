from typing import List
from dataclasses import dataclass, field


@dataclass
class vace_args:
    # -- model
    vace_model_path: str = "/data2/jibaixu/Models/Wan-AI/Wan2.1-VACE-1.3B-diffusers"
    use_lora: bool = False
    trainable_module: str = "all"  # all / vace

    # -- dataset
    dataset_root_path: str = "/data2/jibaixu/Codes/Ctrl-World/dataset_example"
    dataset_names: str = "droid_vace_5hz_192_320_latent"
    probs: List[float] = field(default_factory=lambda: [1.0])  # probability to sample from each dataset, should sum to 1.0
    shuffle: bool = True
    num_cams: int = 2   # The number of cameras in one trajectory.
    filter_length: bool = True  # Whether to filter the video samples by length
    min_len: int = 21   # If 'filter_length' is True, filter the video samples whose length is in [min_len, max_len]
    max_len: int = 25

    # -- training
    learning_rate: float = 1e-5
    weight_decay: float = 1e-2
    train_batch_size: int = 1
    val_batch_size: int = 1
    max_train_steps: int = 6_000
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    checkpointing_steps: int = 300
    validation_steps: int = 100

    # -- log
    sft_tag = "lora" if use_lora else "full"
    wandb_project_name: str = "diffusers_wan_vace_gray"
    wandb_run_name: str = f"{dataset_names}_{sft_tag}_trainable_module_{trainable_module}_lr_{str(learning_rate)}_grad_{gradient_accumulation_steps}_gray_with_ref_img"
    output_dir: str = f"output/{wandb_run_name}"

    def __post_init__(self):
        assert self.trainable_module in ["all", "vace"], "trainable_module should be 'all' or 'vace'"
