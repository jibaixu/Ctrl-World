import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
from typing import List

import torch
import numpy as np
import json
from tqdm import tqdm


class Dataset_mix(torch.utils.data.Dataset):
    def __init__(
            self,
            args,
            mode = 'val',
    ):
        """
        Dataset Stucture:
            dataset_root_path/dataset_name/annotation_name/mode/traj
            dataset_root_path/dataset_name/video/mode/traj
            dataset_root_path/dataset_name/latent_video/mode/traj
        """
        super().__init__()
        self.args = args
        self.mode = mode

        # samples:{'ann_file':xxx, 'frame_idx':xxx, 'dataset_name':xxx}

        # prepare all datasets path
        self.dataset_path_all = []
        self.samples_all = []
        self.samples_len = []
        self.norm_all = []

        dataset_root_path = args.dataset_root_path
        dataset_names = args.dataset_names.split(',')
        self.probs = args.probs
        for dataset_name in dataset_names:
            dataset_path = os.path.join(dataset_root_path, dataset_name)
            self.dataset_path_all.append(dataset_path)

            samples_path = os.path.join(dataset_path, "latent_videos", mode)
            samples = os.listdir(samples_path)

            self.samples_all.append(samples)
            self.samples_len.append(len(samples))
        
        self.max_id = max(self.samples_len)
        print('samples_len:',self.samples_len, 'max_id:',self.max_id)

    def __len__(self):
        return self.max_id

    def __getitem__(self, index):
        # first sample the dataset id, than sample the data from the dataset
        dataset_id = np.random.choice(len(self.samples_all), p=self.probs)

        dataset_path = self.dataset_path_all[dataset_id]
        samples = self.samples_all[dataset_id]
        index = index % len(samples)
        sample = samples[index]

        latent_video_dir = os.path.join(dataset_path, "latent_videos", self.mode)
        cam_id = random.choice(range(self.args.num_cams))
        latent_video_path = os.path.join(latent_video_dir, sample, f"{cam_id}.pt")
        
        with open(latent_video_path,'rb') as file:
            data_dict = torch.load(file)
        
        return_dict = dict()

        for k, v in data_dict.items():
            # Close grad
            v.requires_grad = False
            # Randomly choice a sample from batch
            B = v.shape[0]
            v = v[np.random.choice(B)]
            # Remove 'batch' dim
            v = v.squeeze()

            return_dict[k] = v

        """
        return_dict = {
            "prompt_embeds"     # [num_tokens, dim]
            "latents"   # [C, T, H, W]
            "condition_latents" # [C, T, H, W]
        }
        """
        return return_dict


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    # model parameters
    parser.add_argument('--vace_model_path', type=str, default="/home/jibaixu/Models/Wan/Wan2.1-VACE-1.3B-diffusers")
    parser.add_argument('--use_lora', type=bool, default=True)

    # dataset parameters
    parser.add_argument('--dataset_root_path', type=str, default="/home/jibaixu/Codes/Ctrl-World/dataset_example")
    parser.add_argument('--dataset_names', type=str, default="droid_test_vace")
    parser.add_argument('--probs', type=List[float], default=[1.0], help="probability to sample from each dataset, should sum to 1.0")
    parser.add_argument('--num_cams', type=int, default=2, help="The number of cameras in one trajectory.")

    # training parameters
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_train_epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--max_train_steps', type=int, default=500000)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--mixed_precision', type=str, default='fp16')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--checkpointing_steps', type=int, default=20000)
    parser.add_argument('--validation_steps', type=int, default=2500)

    # log parameters
    parser.add_argument('--output_dir', type=str, default="output/droid_subset")
    parser.add_argument('--wandb_project_name', type=str, default="diffusers_wan_vace")
    parser.add_argument('--wandb_run_name', type=str, default="droid_subset")
    
    args = parser.parse_args()

    train_dataset = Dataset_mix(args, mode="train")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True
    )
    for data in tqdm(train_loader, total=len(train_loader)):
        print(data)
