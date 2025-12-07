# from diffusers import StableVideoDiffusionPipeline
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import math

import numpy as np
import torch
import einops
from accelerate import Accelerator
import datetime
from accelerate.logging import get_logger
from tqdm.auto import tqdm
import swanlab
import mediapy

from models.pipeline_wan_vace import WanVACEPipeline
from models.wan_vace import WanVACE


RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BOLD = '\033[1m'
RESET = '\033[0m'


def init_logging():
    logger = get_logger(__name__)
    if logger.logger.hasHandlers():
        return logger

    accelerator = Accelerator()
    if accelerator.is_main_process:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)

        base_logger = logger.logger
        base_logger.addHandler(handler)
        base_logger.setLevel(logging.INFO)
        base_logger.propagate = False

    return logger


def main(args):
    logger = init_logging()
    swanlab.sync_wandb()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with='wandb',
        project_dir=args.output_dir
    )

    # model and optimizer
    model = WanVACE(vace_model_path=args.vace_model_path, use_lora=args.use_lora)
    model.to(accelerator.device)
    model.train()
    optimizer = torch.optim.AdamW(model.transformer.parameters(), lr=args.learning_rate)

    # logs
    if accelerator.is_main_process:
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        run_name = f"train_{now}_{args.dataset_names}"
        accelerator.init_trackers(args.wandb_project_name, config={}, init_kwargs={"wandb":{"name":run_name}})
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        num_params = sum(p.numel() for p in model.transformer.parameters())
        logger.info(f"Number of parameters in the transformer: {num_params/1000000:.2f}M")

    # train and val datasets
    from dataset.dataset_droid_vace import Dataset_mix
    train_dataset = Dataset_mix(args, mode='train')
    val_dataset = Dataset_mix(args, mode='val')
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=args.shuffle
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.train_batch_size,
        shuffle=args.shuffle
    )

    # Prepare everything with our accelerator
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    ############################ training ##############################
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_train_epochs = math.ceil(args.max_train_steps * total_batch_size / len(train_dataset))

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs (equivalent) = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  checkpointing_steps = {args.checkpointing_steps}")
    logger.info(f"  validation_steps = {args.validation_steps}")
    global_step = 0
    train_loss = 0.0
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    logger.info(f"{GREEN}{BOLD}Total train epochs (equivalent): {num_train_epochs}{RESET}")

    for epoch in range(num_train_epochs):
        if global_step >= args.max_train_steps:
            break
        for step, batch in enumerate(train_dataloader):
            if global_step >= args.max_train_steps:
                break
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    loss_gen = model(batch)
                avg_loss = accelerator.gather(loss_gen.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                accelerator.backward(loss_gen)

                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

            # 只有同步梯度才算一次“优化步”
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # log
                if global_step % 100 == 0:
                    progress_bar.set_postfix({"loss": train_loss})
                    accelerator.log({"train_loss": train_loss / 100}, step=global_step)
                    train_loss = 0.0

                # checkpoint
                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"Transformer-{global_step}")
                    accelerator.unwrap_model(model).transformer.save_pretrained(save_path)
                    logger.info(f"Saved checkpoint to {save_path}")

                # validation（每 N 步做一次，这里改成 !=0 即可）
                # if global_step % args.validation_steps == 0 and global_step != 0 and accelerator.is_main_process:
                #     # TODO: 真正验证代码写这里
                #     model.eval()
                #     with accelerator.autocast():
                #         for id in range(args.video_num):
                #             validate_video_generation(model, val_dataset, args,global_step, args.output_dir, id, accelerator)
                #     model.train()


# def main_val(args):
#     accelerator = Accelerator()
#     model = CrtlWorld(args)
#     # load form val_model_path
#     print("load from val_model_path",args.val_model_path)
#     model.load_state_dict(torch.load(args.val_model_path))
#     model.to(accelerator.device)
#     model.eval()
#     validate_video_generation(model, None, args, 0, 'output', 0, accelerator, load_from_dataset=False)


def validate_video_generation(model, val_dataset, args, train_steps, videos_dir, id, accelerator, load_from_dataset=True):
    device = accelerator.device
    pipeline = model.module.pipeline if accelerator.num_processes > 1 else model.pipeline
    videos_row = args.video_num if not args.debug else 1
    videos_col = 2

    # sample from val dataset
    batch_id = list(range(0,len(val_dataset),int(len(val_dataset)/videos_row/videos_col)))
    batch_id = batch_id[int(id*(videos_col)):int((id+1)*(videos_col))]
    batch_list = [val_dataset.__getitem__(id) for id in batch_id]
    video_gt = torch.cat([t['latent'].unsqueeze(0) for i,t in enumerate(batch_list)],dim=0).to(device, non_blocking=True)
    text = [t['text'] for i,t in enumerate(batch_list)]
    actions = torch.cat([t['action'].unsqueeze(0) for i,t in enumerate(batch_list)],dim=0).to(device, non_blocking=True)
    his_latent_gt, future_latent_ft = video_gt[:,:args.num_history], video_gt[:,args.num_history:]
    current_latent = future_latent_ft[:,0]
    print("image",current_latent.shape, 'action', actions.shape)
    assert current_latent.shape[1:] == (4, 72, 40)
    assert actions.shape[1:] == (int(args.num_frames+args.num_history), args.action_dim)

    # start generate
    with torch.no_grad():
        bsz = actions.shape[0]
        action_latent = model.module.action_encoder(actions, text, model.module.tokenizer, model.module.text_encoder, args.frame_level_cond) if accelerator.num_processes > 1 else model.action_encoder(actions, text, model.tokenizer, model.text_encoder,args.frame_level_cond) # (8, 1, 1024)
        print("action_latent",action_latent.shape)

        _, pred_latents = WanVACEPipeline.__call__(
            pipeline,
            image=current_latent,
            text=action_latent,
            width=args.width,
            height=int(3*args.height),
            num_frames=args.num_frames,
            history=his_latent_gt,
            num_inference_steps=args.num_inference_steps,
            decode_chunk_size=args.decode_chunk_size,
            max_guidance_scale=args.guidance_scale,
            fps=args.fps,
            motion_bucket_id=args.motion_bucket_id,
            mask=None,
            output_type='latent',
            return_dict=False,
            frame_level_cond=args.frame_level_cond,
            his_cond_zero=args.his_cond_zero,
        )
    
    pred_latents = einops.rearrange(pred_latents, 'b f c (m h) (n w) -> (b m n) f c h w', m=3,n=1) # (B, 8, 4, 32,32)
    video_gt = torch.cat([his_latent_gt, future_latent_ft], dim=1) # (B, 8, 4, 32,32)
    video_gt = einops.rearrange(video_gt, 'b f c (m h) (n w) -> (b m n) f c h w', m=3,n=1) # (B, 8, 4, 32,32)
    
    # decode latent
    if video_gt.shape[2] != 3:
        decoded_video = []
        bsz,frame_num = video_gt.shape[:2]
        video_gt = video_gt.flatten(0,1)
        decode_kwargs = {}
        for i in range(0,video_gt.shape[0],args.decode_chunk_size):
            chunk = video_gt[i:i+args.decode_chunk_size]/pipeline.vae.config.scaling_factor
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        video_gt = torch.cat(decoded_video,dim=0)
        video_gt = video_gt.reshape(bsz,frame_num,*video_gt.shape[1:])
        
        decoded_video = []
        bsz,frame_num = pred_latents.shape[:2]
        pred_latents = pred_latents.flatten(0,1)
        decode_kwargs = {}
        for i in range(0,pred_latents.shape[0],args.decode_chunk_size):
            chunk = pred_latents[i:i+args.decode_chunk_size]/pipeline.vae.config.scaling_factor
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        videos = torch.cat(decoded_video,dim=0)
        videos = videos.reshape(bsz,frame_num,*videos.shape[1:])

    video_gt = ((video_gt / 2.0 + 0.5).clamp(0, 1)*255)
    video_gt = video_gt.to(pipeline.unet.dtype).detach().cpu().numpy().transpose(0,1,3,4,2).astype(np.uint8)
    videos = ((videos / 2.0 + 0.5).clamp(0, 1)*255)
    videos = videos.to(pipeline.unet.dtype).detach().cpu().numpy().transpose(0,1,3,4,2).astype(np.uint8) #(2,16,256,256,3)
    videos = np.concatenate([video_gt[:, :args.num_history],videos],axis=1) #(2,16,512,256,3)
    videos = np.concatenate([video_gt,videos],axis=-3) #(2,16,512,256,3)
    videos = np.concatenate([video for video in videos],axis=-2).astype(np.uint8) # (16,512,256*batch,3)
    
    os.makedirs(f"{videos_dir}/samples", exist_ok=True)
    filename = f"{videos_dir}/samples/train_steps_{train_steps}_{id}.mp4"
    mediapy.write_video(filename, videos, fps=2)
    return


if __name__ == "__main__":
    from config_vace import vace_args

    main(vace_args())

    # accelerate launch --num_processes 2 --main_process_port 29500 --gpu_ids 6,7 scripts/train_vace.py
