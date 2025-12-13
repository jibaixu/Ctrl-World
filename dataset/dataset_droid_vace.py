import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import json
from tqdm import tqdm
import mediapy
import PIL.Image


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

            total_anno_path = os.path.join(dataset_path, "annotation", f"{mode}.json")

            with open(total_anno_path, 'r', encoding="utf-8") as f:
                self.total_anno_dict = json.load(f)

            # Filter "video_length" less than 100
            if self.args.filter_length:
                samples = sorted([k for k, v in self.total_anno_dict.items() if v['video_length'] >= self.args.min_len and v['video_length'] <= self.args.max_len])

            self.samples_all.append(samples)
            self.samples_len.append(len(samples))
        
        self.max_id = max(self.samples_len)
        print(f'\033[1;32m{mode} dataset: samples_len:{self.samples_len}, max_id: {self.max_id}\033[0m')

    def __len__(self):
        return self.max_id

    def __getitem__(self, index):
        # first sample the dataset id, than sample the data from the dataset
        dataset_id = np.random.choice(len(self.samples_all), p=self.probs)

        dataset_path = self.dataset_path_all[dataset_id]
        samples = self.samples_all[dataset_id]
        index = index % len(samples)
        sample = samples[index]

        video_dir = os.path.join(dataset_path, "videos", self.mode)
        latent_video_dir = os.path.join(dataset_path, "latent_videos", self.mode)
        # cam_id = random.choice(range(self.args.num_cams))
        cam_id = 0
        video_path = os.path.join(video_dir, sample, f"{cam_id}.mp4")
        latent_video_path = os.path.join(latent_video_dir, sample, f"{cam_id}.pt")
        prompt_embed_path = os.path.join(latent_video_dir, sample, "prompt_embeds.pt")
        
        video = mediapy.read_video(video_path)
        video = torch.tensor(video)
        with open(latent_video_path, 'rb') as file:
            latent_dict = torch.load(file)
        with open(prompt_embed_path, 'rb') as file:
            prompt_embeds = torch.load(file)
        latent_dict["prompt_embeds"] = prompt_embeds

        return_dict = dict()
        return_dict["videos"] = video

        for k, v in latent_dict.items():
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
            "videos"    # [T, H, W, C] torch.tensor(uint8)
            "prompt_embeds"     # [num_tokens, dim]
            "latents"   # [C, T, H, W]
            "condition_latents" # [C, T, H, W]
        }
        """
        return return_dict


if __name__ == "__main__":

    from config_vace import vace_args

    train_dataset = Dataset_mix(vace_args(), mode="train")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=vace_args.train_batch_size,
        shuffle=True
    )
    for data in tqdm(train_loader, total=len(train_loader)):
        print(data)
