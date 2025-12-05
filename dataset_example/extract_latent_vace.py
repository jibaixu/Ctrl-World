import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import math
import shutil

import numpy as np
import torch
from torch.utils.data import Dataset
import mediapy
import json
import pandas as pd
from diffusers.models import AutoencoderKLWan
from accelerate import Accelerator

from models.pipeline_wan_vace import WanVACEPipeline


class EncodeLatentDataset(Dataset):
    def __init__(self, old_path, new_path, model_path, size=(192, 320), rgb_skip=3, num_frames=81, batch_size=3, extract_latent=True, resume=False, resume_traj_id=None, device="cuda"):
        self.old_path = old_path
        self.new_path = new_path
        self.size = size
        self.skip = rgb_skip
        self.num_frames = num_frames
        self.batch_size = batch_size    # 在extract latent时，每一个video选择的num_frames视频片段的个数
        self.extract_latent = extract_latent

        vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
        self.pipe = WanVACEPipeline.from_pretrained(model_path, vae=vae, torch_dtype=torch.bfloat16)
        self.pipe.text_encoder.to(device)
        self.pipe.vae.to(device)
        self.generator = torch.Generator().manual_seed(42)

        def load_json_file(file_path):
            data = []
            with open(file_path, "r") as f:
                for line in f:
                    data.append(json.loads(line))  # 使用 json.loads() 解析单行
            return data

        self.data = load_json_file(f'{old_path}/meta/episodes.jsonl')

        # Remove trajectories which 'tasks' is empty
        self.data = [traj for traj in self.data if len(traj['tasks'][0]) > 0]
        # Remove trajectories which 'length' is too short
        self.data = [traj for traj in self.data if traj['length'] >= 20*rgb_skip]
        if resume:
            self.data = [traj for traj in self.data if traj['episode_index'] >= resume_traj_id]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        traj_data = self.data[idx]
        instruction = traj_data['tasks'][0]
        traj_id = traj_data['episode_index']
        chunk_id = int(traj_id/1000)

        mode = 'val' if traj_id % 50 == 49 else 'train'

        file_path = f'{self.old_path}/data/chunk-{chunk_id:03d}/episode_{traj_id:06d}.parquet'
        df = pd.read_parquet(file_path)
        length = len(df['observation.state.cartesian_position'])

        obs_car = []
        obs_joint =[]
        obs_gripper = []
        action_car = []
        action_joint = []
        action_gripper = []
        action_joint_vel = []

        for i in range(length):
            obs_car.append(df['observation.state.cartesian_position'][i].tolist())
            obs_joint.append(df['observation.state.joint_position'][i].tolist())
            obs_gripper.append(df['observation.state.gripper_position'][i].tolist())
            action_car.append(df['action.cartesian_position'][i].tolist())
            action_joint.append(df['action.joint_position'][i].tolist())
            action_gripper.append(df['action.gripper_position'][i].tolist())
            action_joint_vel.append(df['action.joint_velocity'][i].tolist())
        success = df['is_episode_successful'][0]
        video_paths = [
            f'{self.old_path}/videos/chunk-{chunk_id:03d}/observation.images.exterior_1_left/episode_{traj_id:06d}.mp4',
            f'{self.old_path}/videos/chunk-{chunk_id:03d}/observation.images.exterior_2_left/episode_{traj_id:06d}.mp4',
            # f'{self.old_path}/videos/chunk-{chunk_id:03d}/observation.images.wrist_left/episode_{traj_id:06d}.mp4'
        ]
        traj_info = {
            'success': success,
            'observation.state.cartesian_position': obs_car,
            'observation.state.joint_position': obs_joint,
            'observation.state.gripper_position': obs_gripper,
            'action.cartesian_position': action_car,
            'action.joint_position': action_joint,
            'action.gripper_position': action_gripper,
            'action.joint_velocity': action_joint_vel,
        }
        
        if self.check_traj_file(traj_id, mode):
            print(f"\033[1;33mTraj {traj_id} has existed, continue ...\033[0m")
            return 0
        self.process_traj(
            video_paths,
            traj_info,
            instruction,
            self.new_path,
            traj_id=traj_id,
            mode=mode,
            device=self.pipe.vae.device
        )
        print(f"\033[1;32mTraj {traj_id} finish!\033[0m")
        return 0

    @torch.no_grad()
    def process_traj(self, video_paths, traj_info, instruction, save_root, traj_id=0, mode='val', device='cuda'):
        ##################### construct prompt embed #####################
        prompt_embeds, _ = self.pipe.encode_prompt(
            prompt=instruction,
            negative_prompt=None,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            max_sequence_length=512,
            device=device,
        )
        prompt_embeds = prompt_embeds.cpu()

        os.makedirs(f"{save_root}/latent_videos/{mode}/{traj_id}", exist_ok=True)
        torch.save(prompt_embeds, f"{save_root}/latent_videos/{mode}/{traj_id}/prompt_embeds.pt")

        ##################### construct video and latent #####################
        latents_starts_info = dict()
        for video_id, video_path in enumerate(video_paths):
            # load and resize video and save
            try:
                video = mediapy.read_video(video_path)
            except:
                print(f"### Video path is not exist: {video_path}")
                continue
            frames = torch.tensor(video).permute(0, 3, 1, 2).float() / 255.0*2-1
            frames = frames[::self.skip]  # Skip frames to save memory here!!!
            resized_frames = torch.nn.functional.interpolate(frames, size=self.size, mode='bilinear', align_corners=False)
            resized_frames = ((resized_frames / 2.0 + 0.5).clamp(0, 1)*255)
            resized_frames = resized_frames.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            
            os.makedirs(f"{save_root}/videos/{mode}/{traj_id}", exist_ok=True)
            mediapy.write_video(f"{save_root}/videos/{mode}/{traj_id}/{video_id}.mp4", resized_frames, fps=5)

            ##################### construct wan latent #####################
            # construct video and mask
            if not self.extract_latent:
                continue

            from PIL import Image
            total_video = [Image.fromarray(v) for v in resized_frames]
            #! 尾帧重复
            if len(total_video) < self.num_frames:
                last_frame = total_video[-1]
                total_video = total_video + [last_frame] * (self.num_frames - len(total_video))

            latents, condition_latents = [], []
            #! 每个视频仅构造一个batch
            starts = np.random.randint(0, len(total_video)-self.num_frames+1, size=self.batch_size)
            latents_starts_info[video_id] = starts.tolist()
            for start in starts:
                video = total_video[start : start + self.num_frames]

                latent, condition_latent = self.pipe.construct_latents_from_video(
                    video,
                    generator=self.generator,
                    device=device,
                )
                latents.append(latent.cpu())
                condition_latents.append(condition_latent.cpu())

            latents_dict = {
                "latents": torch.cat(latents, dim=0), # [B, C, T, H, W]
                "condition_latents": torch.cat(condition_latents, dim=0), # [B, C, T, H, W]
            }

            os.makedirs(f"{save_root}/latent_videos/{mode}/{traj_id}", exist_ok=True)
            torch.save(latents_dict, f"{save_root}/latent_videos/{mode}/{traj_id}/{video_id}.pt")
        
        ##################### construct annotation #####################
        # record cartesain aligned with video frames
        cartesian_pose = np.array(traj_info['observation.state.cartesian_position'])
        cartesian_gripper = np.array(traj_info['observation.state.gripper_position'])[:,None]
        # print(cartesian_pose.shape, cartesian_gripper.shape)
        cartesian_states = np.concatenate((cartesian_pose, cartesian_gripper),axis=-1)[::self.skip].tolist()
        if len(cartesian_states) < self.num_frames:
            last_state = cartesian_states[-1]
            cartesian_states = cartesian_states + [last_state] * (self.num_frames - len(cartesian_states))
        
        info = {
            "texts": [instruction],
            "episode_id": traj_id,
            "success": int(traj_info['success']),
            "video_length": frames.shape[0],
            "state_length": len(cartesian_states),
            "raw_length": len(traj_info['observation.state.cartesian_position']),
            "videos": [
                {"video_path": f"videos/{mode}/{traj_id}/0.mp4"},
                {"video_path": f"videos/{mode}/{traj_id}/1.mp4"},
                {"video_path": f"videos/{mode}/{traj_id}/2.mp4"}
            ],
            "latent_videos": [
                {"latent_video_path": f"latent_videos/{mode}/{traj_id}/0.pt"},
                {"latent_video_path": f"latent_videos/{mode}/{traj_id}/1.pt"},
                {"latent_video_path": f"latent_videos/{mode}/{traj_id}/2.pt"}
            ],
            "latents_starts_info": latents_starts_info,
            'states': cartesian_states,
            'observation.state.cartesian_position': traj_info['observation.state.cartesian_position'],
            'observation.state.joint_position': traj_info['observation.state.joint_position'],
            'observation.state.gripper_position': traj_info['observation.state.gripper_position'],
            'action.cartesian_position': traj_info['action.cartesian_position'],
            'action.joint_position': traj_info['action.joint_position'],
            'action.gripper_position': traj_info['action.gripper_position'],
            'action.joint_velocity': traj_info['action.joint_velocity'],
            }
        os.makedirs(f"{save_root}/annotation/{mode}", exist_ok=True)
        with open(f"{save_root}/annotation/{mode}/{traj_id}.json", "w") as f:
            json.dump(info, f, indent=2)

    def check_traj_file(self, sample, mode):
        annotation_path = os.path.join(self.new_path, "annotation", mode, f"{sample}.json")
        latent_path = os.path.join(self.new_path, "latent_videos", mode, sample)
        video_path = os.path.join(self.new_path, "videos", mode, sample)

        if not os.path.exists(annotation_path) or \
            not os.path.exists(latent_path) or \
            not os.path.exists(video_path):
            print(f"Sample {sample} miss annotation or latent or video!")
            return False
        
        files = os.listdir(latent_path)
        if "0.pt" not in files or \
            "1.pt" not in files or \
            "prompt_embeds.pt" not in files:
            print(f"Sample {sample} miss latent file!")
            return False

        return True

    def delete_miss_file_traj(self):
        """
        用于在数据处理完成后进行过滤
        """
        modes = ['train', 'val']
        for mode in modes:
            del_samples = []
            annotation_dir = os.path.join(self.new_path, "annotation", mode)
            latent_videos_dir = os.path.join(self.new_path, "latent_videos", mode)
            videos_dir = os.path.join(self.new_path, "videos", mode)

            samples = os.listdir(latent_videos_dir)
            for sample in samples:
                if self.check_traj_file(sample, mode):
                    del_samples.append(sample)
            print(del_samples)

            # Delete
            def safe_delete(path):
                if not os.path.exists(path):
                    return
                if os.path.isfile(path):
                    os.remove(path)
                else:                       # 是目录
                    shutil.rmtree(path)

            # for sample in del_samples:
            #     annotation_path = os.path.join(annotation_dir, f"{sample}.json")
            #     latent_path = os.path.join(latent_videos_dir, sample)
            #     video_path = os.path.join(videos_dir, sample)
            #     safe_delete(annotation_path)
            #     safe_delete(latent_path)
            #     safe_delete(video_path)


if __name__ == "__main__":

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--droid_hf_path', type=str, default='/home/jibaixu/Datasets/droid_1.0.1')
    parser.add_argument('--droid_output_path', type=str, default='dataset_example/droid_vace_dynamic_hz_192_320')
    parser.add_argument('--model_path', type=str, default='/home/jibaixu/Models/Wan/Wan2.1-VACE-1.3B-diffusers')
    # debug
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    accelerator = Accelerator()
    dataset = EncodeLatentDataset(
        old_path=args.droid_hf_path,
        new_path=args.droid_output_path,
        model_path=args.model_path,
        size=(192, 320),
        rgb_skip=3, #  to downsample 15hz video to 5hz video
        num_frames=81,
        batch_size=1,
        extract_latent=True,   # 是否进行潜变量压缩
        resume=True,
        resume_traj_id=25577,
        device=accelerator.device,
    )
    # dataset.delete_miss_file_traj()
    tmp_data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
    )

    from tqdm import tqdm
    tmp_data_loader = accelerator.prepare_data_loader(tmp_data_loader)
    for idx, _ in enumerate(tqdm(tmp_data_loader, desc="Processing")):
        if idx == 40 and args.debug:
            break

# accelerate launch --num_processes 8 --main_process_port 29500 --gpu_ids 0,1,2,3 dataset_example/extract_latent_vace.py
