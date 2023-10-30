# ------------------------------------------------------------------------------
# TCL
# Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# ------------------------------------------------------------------------------
# Modified from GroupViT (https://github.com/NVlabs/GroupViT)
# Copyright (c) 2021-22, NVIDIA Corporation & affiliates. All Rights Reserved.
# ------------------------------------------------------------------------------
import argparse
import datetime
import os
import os.path as osp
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
from torch.distributed.distributed_c10d import _get_default_group
import numpy as np
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmcv.utils import collect_env, get_git_hash
from mmseg.apis import multi_gpu_test

from datasets import build_loader, build_text_transform
from models import build_model
from omegaconf import OmegaConf, read_write
from segmentation.evaluation import build_seg_dataloader, build_seg_dataset, build_seg_inference
from timm.utils import AverageMeter
from torchvision.utils import make_grid
from utils import (
    build_optimizer,
    build_scheduler,
    get_config,
    get_grad_norm,
    get_logger,
    load_checkpoint,
    parse_losses,
    save_checkpoint,
    CheckpointManager,
    load_config
)
import us


def cyclize(loader):
    while True:
        for i in loader:
            yield i


def get_argparser():
    parser = argparse.ArgumentParser("TCL training and evaluation script")
    parser.add_argument("--cfg", type=str, help="path to config file")
    parser.add_argument(
        "--opts", help="Modify config options by adding 'KEY=VALUE' list. ", default=None, nargs="+"
    )

    # easy config modification
    parser.add_argument("--batch-size", type=int, help="batch size for single GPU")
    parser.add_argument("--resume", help="resume from checkpoint")
    parser.add_argument(
        "--output",
        type=str,
        help="root of output folder, " "the full path is <output>/<model_name>/<tag>",
    )
    parser.add_argument("--tag", type=str, help="tag of experiment")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--wandb", action="store_true", help="Use W&B to log experiments")

    return parser


def train(cfg):
    # if not cfg.model.clip_model.lower().startswith("vit"):
    #     raise ValueError("Current TCL only supports ViT backbone of CLIP.")

    if cfg.wandb and dist.get_rank() == 0:
        import wandb

        wandb.init(
            project="tcl",
            name=osp.join(cfg.model_name, cfg.tag),
            dir=cfg.output,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume=False,
        )
    else:
        wandb = None
    # waiting wandb init
    dist.barrier()

    # build datasets
    dataset_train, data_loader_train = build_loader(cfg.data)

    # build validation loaders
    val_loaders = {}
    # for key in cfg.evaluate.task:
    #     if key == "cls":
    #         continue

    #     loader = build_seg_dataloader(build_seg_dataset(cfg.evaluate.get(key)))
    #     val_loaders[key] = loader

    logger = get_logger()

    # build model & optimizer
    logger.info(f"Creating model:{cfg.model.type}/{cfg.model_name}")
    model = build_model(cfg.model)
    model.class_freq = dataset_train.class_freq
    model.cuda()

    model.set_train(decoder_only=(cfg.train.ust_steps > 0), config=cfg)
    optimizer = build_optimizer(cfg.train, model)
    model = MMDistributedDataParallel(
        model,
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False, find_unused_parameters=True
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters} ({n_parameters/1000/1000:.1f}M)")
    lr_scheduler = build_scheduler(cfg.train, optimizer)

    # fp16 compression
    logger.info(us.dist_info())
    if cfg.train.fp16 and cfg.train.fp16_comm:
        if int(os.getenv("LOCAL_WORLD_SIZE")) < int(os.getenv("WORLD_SIZE")):
            pg = _get_default_group()
            logger.info("!!! Multi-node setting :: turn on fp16 compression hook")
            model.register_comm_hook(pg, fp16_compress_hook)
        else:
            logger.info("!!! Single-node setting :: skip fp16 compression hook")

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.fp16)

    if cfg.checkpoint.resume:
        load_checkpoint(cfg, model.module, optimizer, lr_scheduler, scaler)

    if cfg.evaluate.eval_only:
        res = evaluate(cfg, model, val_loaders)
        logger.info(res)
        r = ", ".join([f"{v:.2f}" for v in res.values()])
        logger.info(f" >> {r}")
        return

    logger.info("Start training")
    start_time = time.time()

    do_training(cfg, model, data_loader_train, optimizer, lr_scheduler, scaler, val_loaders)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))
    dist.barrier()


def do_training(config, model, data_loader, optimizer, lr_scheduler, scaler, val_loaders):
    logger = get_logger()
    dist.barrier()
    model.train()
    optimizer.zero_grad()
    if config.wandb and dist.get_rank() == 0:
        import wandb
    else:
        wandb = None

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    log_vars_meters = defaultdict(AverageMeter)

    total_steps = config.train.total_steps
    org_total_steps = total_steps
    # update training steps by evaluation step (discard non-evaluation steps)
    # total_steps = total_steps - (total_steps % config.evaluate.eval_freq) + 1
    if org_total_steps != total_steps:
        logger.info(f"Total step is updated: {org_total_steps} -> {total_steps}")

    ckpt_manager = CheckpointManager(config.checkpoint.save_topk, config.output)

    ust_check = True
    end = time.time()
    for step, samples in enumerate(cyclize(data_loader), config.train.start_step):
        if step >= total_steps:
            break
        # if ust_check and config.train.ust_steps and step >= config.train.ust_steps:
        #     model.module.set_train(decoder_only=False, config=config)
        #     logger.info(f" -- [{step}] UST stage is DONE; Now fine-tuning stage begins ...")
        #     ust_check = False

        batch_size = config.data.batch_size
        caption = samples["caption"]

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=config.train.fp16):
            losses = model(samples['image'], samples["caption"], samples["tag_label"])

        loss, log_vars = parse_losses(losses)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if config.train.clip_grad:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad)
        else:
            grad_norm = get_grad_norm(model.parameters())

        scaler.step(optimizer)
        scaler.update()

        lr_scheduler.step()
        torch.cuda.synchronize()

        loss_meter.update(loss.item(), batch_size)
        for loss_name in log_vars:
            log_vars_meters[loss_name].update(log_vars[loss_name].item(), batch_size)
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if step % config.print_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            epoch = step / num_steps
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            #  etas = batch_time.avg * (num_steps - step)
            log_vars_str = "  ".join(
                f"{n} {m.val:7.4f} ({m.avg:7.4f})" for n, m in log_vars_meters.items()
            )
            logger.info(
                f"Train: [EP {epoch:.1f}][{step:6d}/{total_steps}]  "
                #  f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f"time {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                f"total_loss {loss_meter.val:7.4f} ({loss_meter.avg:7.4f})  "
                f"{log_vars_str}  "
                f"grad_norm {norm_meter.val:7.4f} ({norm_meter.avg:7.4f})  "
                f"lr {lr:.6f}  "
                f"mem {memory_used:.0f}MB"
            )

            if wandb is not None:
                log_stat = {f"iter/train_{n}": m.val for n, m in log_vars_meters.items()}
                log_stat["iter/train_total_loss"] = loss_meter.val
                log_stat["iter/grad_norm"] = norm_meter.val
                log_stat["iter/learning_rate"] = lr
                log_stat["iter/epoch"] = epoch
                log_stat["iter/grad_scale"] = scaler.get_scale()

                # image & mask logging
                if "mask" in losses and step % 500 == 0:
                    N = 3

                    # un-normalize image
                    org_img = us.unnorm(samples["image"][:N])
                    org_img = torch.clamp(org_img, 0.0, 1.0)  # random erasing makes out-of-range value
                    mask = losses["mask"][:N].repeat(1, 3, 1, 1).cpu().float()
                    mask = F.interpolate(mask, org_img.shape[2:]) > 0.5
                    log_images = [org_img, mask, org_img * mask]
                    if "neg_mask" in losses:
                        neg_mask = losses["neg_mask"][:N, :1].repeat(1, 3, 1, 1).cpu().float()
                        neg_mask = F.interpolate(neg_mask, org_img.shape[2:]) > 0.5
                        log_images.append(neg_mask)

                    log_images = torch.cat(log_images)
                    grid = make_grid(log_images, nrow=N, value_range=(0, 1))
                    cap = "\n".join([f"[{i}] {c}" for i, c in enumerate(caption[:N])])
                    log_stat["examples"] = wandb.Image(grid, caption=cap)

                wandb.log(log_stat, step=step)

        if step and step % config.evaluate.eval_freq == 0:
            if us.is_global_zero():
                ckpt_kwargs = {
                    "config": config,
                    "step": step,
                    "model": model,
                    "optimizer": optimizer,
                    "lr_scheduler": lr_scheduler,
                    "scaler": scaler,
                }
                # save latest to "checkpoint.pth"
                # save_checkpoint(**ckpt_kwargs)
                # save all
                if config.checkpoint.save_all:
                    save_checkpoint(**ckpt_kwargs, filename=f"ckpt_{step}.pth")
            dist.barrier()

            batch_time.reset()
            loss_meter.reset()
            norm_meter.reset()
            for m in log_vars_meters.values():
                m.reset()

        # if step and step % config.evaluate.eval_freq == 0:
        #     metrics = evaluate(config, model, val_loaders)

        #     if us.is_global_zero():
        #         ckpt_kwargs = {
        #             "config": config,
        #             "step": step,
        #             "model": model,
        #             "optimizer": optimizer,
        #             "lr_scheduler": lr_scheduler,
        #             "scaler": scaler,
        #             "metrics": metrics,
        #         }
        #         # save latest to "checkpoint.pth"
        #         save_checkpoint(**ckpt_kwargs)
        #         # save all
        #         if config.checkpoint.save_all:
        #             save_checkpoint(**ckpt_kwargs, filename=f"ckpt_{step}.pth")
        #         # save best
        #         if config.checkpoint.save_topk:
        #             miou = metrics["val/avg_miou"]
        #             ckpt_manager.add(miou, ckpt_kwargs, step)
        #     dist.barrier()

        #     if wandb is not None:
        #         wandb.log(metrics, step=step)

        #     batch_time.reset()
        #     loss_meter.reset()
        #     norm_meter.reset()
        #     for m in log_vars_meters.values():
        #         m.reset()


@torch.no_grad()
def evaluate(cfg, model, val_loaders):
    logger = get_logger()
    ret = {}
    model.eval()

    for key, loader in val_loaders.items():
        if key == "cls":
            continue

        dataset_class = loader.dataset.__class__.__name__
        logger.info(f"### Validation dataset: {key} ({dataset_class})")

        miou, metrics = validate_seg(cfg, cfg.evaluate.get(key), loader, model)

        logger.info(f"[{key}] mIoU of {len(loader.dataset)} test images: {miou:.2f}%")
        ret[f"val/{key}_miou"] = miou

    ret["val/avg_miou"] = np.mean([v for k, v in ret.items() if "miou" in k])

    model.train()

    return ret


@torch.no_grad()
def validate_seg(config, seg_config, data_loader, model):
    logger = get_logger()
    dist.barrier()
    model.eval()

    if hasattr(model, "module"):
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    text_transform = build_text_transform()
    seg_model = build_seg_inference(
        model_without_ddp,
        data_loader.dataset,
        text_transform,
        config,
        seg_config,
    )

    mmddp_model = MMDistributedDataParallel(
        seg_model, device_ids=[torch.cuda.current_device()], broadcast_buffers=False, 
    )
    mmddp_model.eval()

    results = multi_gpu_test(
        model=mmddp_model,
        data_loader=data_loader,
        tmpdir=None,
        gpu_collect=True,
        efficient_test=False,
        pre_eval=True,
        format_only=False,
    )

    if dist.get_rank() == 0:
        metric = [data_loader.dataset.evaluate(results, metric="mIoU", logger=logger)]
    else:
        metric = [None]

    dist.broadcast_object_list(metric)
    miou_result = metric[0]["mIoU"] * 100

    torch.cuda.empty_cache()
    dist.barrier()
    return miou_result, metric


def main():
    parser = get_argparser()
    args = parser.parse_args()

    if args.resume and args.eval:
        # update config when resume
        # default config -> org config -> eval config
        default_cfg = load_config("configs/tcl.yml")
        org_cfg_path = Path(args.resume).parent / "config.json"
        if org_cfg_path.exists():
            org_cfg = OmegaConf.load(Path(args.resume).parent / "config.json")
        else:
            org_cfg = OmegaConf.create()  # empty container
        eval_cfg = OmegaConf.load("configs/eval.yml")
        cfg = OmegaConf.merge(default_cfg, org_cfg, eval_cfg)
        if args.opts is not None:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.opts))

        cfg.checkpoint.resume = args.resume
        cfg.wandb = False
        cfg.evaluate.eval_only = args.eval
        cfg.output = "output/eval"
    else:
        cfg = get_config(args)

    # start faster ref: https://github.com/open-mmlab/mmdetection/pull/7036
    mp.set_start_method("fork", force=True)
    init_dist("pytorch")
    rank, world_size = get_dist_info()
    print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")

    dist.barrier()

    set_random_seed(cfg.seed, use_rank_shift=True)
    cudnn.benchmark = True

    os.makedirs(cfg.output, exist_ok=True)
    logger = get_logger(cfg)

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = cfg.train.base_lr * cfg.data.batch_size * world_size / 4096.0
    linear_scaled_min_lr = cfg.train.min_lr * cfg.data.batch_size * world_size / 4096.0

    with read_write(cfg):
        logger.info(f"Scale base_lr from {cfg.train.base_lr} to {linear_scaled_lr}")
        logger.info(f"Scale min_lr from {cfg.train.min_lr} to {linear_scaled_min_lr}")
        cfg.train.base_lr = linear_scaled_lr
        cfg.train.min_lr = linear_scaled_min_lr

    if dist.get_rank() == 0:
        path = os.path.join(cfg.output, "config.json")
        OmegaConf.save(cfg, path)
        logger.info(f"Full config saved to {path}")

    # log env info
    env_info_dict = collect_env()
    env_info = "\n".join([f"{k}: {v}" for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)

    logger.info(f"Git hash: {get_git_hash(digits=7)}")

    # print config
    logger.info(OmegaConf.to_yaml(cfg))

    train(cfg)
    dist.barrier()


if __name__ == "__main__":
    main()
