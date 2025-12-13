# from diffusers import StableVideoDiffusionPipeline
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["WANDB_MODE"] = "disabled"
# os.environ["SWANLAB_MODE"] = "disabled"
import math
import json

import numpy as np
import torch
import einops
from accelerate import Accelerator
import datetime
from accelerate.logging import get_logger
import logging
from tqdm.auto import tqdm
import swanlab
import mediapy
from dataclasses import asdict

from models.pipeline_wan_vace import WanVACEPipeline
from models.wan_vace import WanVACE


RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BOLD = '\033[1m'
RESET = '\033[0m'


def init_logging(accelerator):
    logger = get_logger(__name__)
    if logger.logger.hasHandlers():
        return logger

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
    # accelerate and logger
    swanlab.sync_wandb()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with='wandb',
        project_dir=args.output_dir
    )
    logger = init_logging(accelerator)

    # model and optimizer
    model = WanVACE(
        vace_model_path=args.vace_model_path,
        trainable_module=args.trainable_module,
        use_lora=args.use_lora
    )
    model.pipeline.to(accelerator.device)
    model.transformer.train()
    trainable_params = [p for p in model.transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    # logs
    if accelerator.is_main_process:
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        run_name = f"train_{args.wandb_run_name}_{now}"
        accelerator.init_trackers(args.wandb_project_name, config={}, init_kwargs={"wandb":{"name":run_name}})
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        num_params = sum(p.numel() for p in model.transformer.parameters())
        logger.info(f"Number of parameters in the transformer: {num_params/1000000:.2f}M")
        logger.info(f"Number of trainable parameters in the transformer: {sum(p.numel() for p in trainable_params)/1000000:.2f}M")
        logger.info(f"***** VACE Config Args *****")
        logger.info(f"{json.dumps(asdict(args), ensure_ascii=False, indent=2)}")

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
    model, optimizer, scheduler, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader, val_dataloader
    )

    ############################ training ##############################
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_train_epochs = math.ceil(args.max_train_steps * total_batch_size / len(train_dataset))

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs (equivalent) = {GREEN}{BOLD}{num_train_epochs}{RESET}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  checkpointing_steps = {args.checkpointing_steps}")
    logger.info(f"  validation_steps = {args.validation_steps}")
    global_step = 0
    train_loss = 0.0
    val_loss = 0.0
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(num_train_epochs):
        if global_step >= args.max_train_steps:
            break
        for step, batch in enumerate(train_dataloader):
            if global_step >= args.max_train_steps:
                break
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    loss_gen = model(batch)

                train_loss += loss_gen.item()

                accelerator.backward(loss_gen)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # 只有同步梯度才算一次“优化步”
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                avg_loss_local = train_loss / args.gradient_accumulation_steps
                # log
                if global_step % 10 == 0:
                    loss_tensor = torch.tensor(avg_loss_local, device=accelerator.device)
                    gathered_loss = accelerator.gather(loss_tensor)
                    global_avg_loss = gathered_loss.mean().item()

                    progress_bar.set_postfix({"loss": global_avg_loss})
                    accelerator.log({"train_loss": global_avg_loss}, step=global_step)

                # checkpoint
                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"{global_step}")
                    accelerator.unwrap_model(model).save_model(save_path)
                    logger.info(f"Saved checkpoint to {save_path}")

                train_loss = 0.0

                # if global_step % args.validation_steps == 0 and accelerator.is_main_process:
                #     model.eval()

                #     with torch.no_grad():
                #         for batch in val_dataloader:
                #             with accelerator.autocast():
                #                 loss_gen = model(batch)
                #             # Multi process average loss
                #             avg_loss = accelerator.gather(loss_gen.repeat(args.val_batch_size)).mean()
                #             # 训练阶段达到梯度累计值时进行反向传播，所以要平均梯度累计；验证阶段是计算整个验证集的平均loss
                #             val_loss += avg_loss.item() / len(val_dataset)
                #     accelerator.log({"val_loss": val_loss}, step=global_step)
                #     logger.info(f"{YELLOW}{BOLD}Validation loss: {val_loss:.4f}{RESET}")
                #     val_loss = 0.0

                    # model.train()


if __name__ == "__main__":
    from config_vace import vace_args

    main(vace_args())

    # accelerate launch --num_processes 2 --main_process_port 29500 --gpu_ids 4,7 scripts/train_vace.py
