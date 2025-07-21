"""
A script for training a diffusion model for paired image-to-image translation.
"""

import argparse
import numpy as np
import random
import sys
import torch as th
from scripts.taming.data.brats_t2w import BraTS2021Train
from torch.utils.data.distributed import DistributedSampler

sys.path.append(".")
sys.path.append("..")

from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (model_and_diffusion_defaults, create_model_and_diffusion,
                                          args_to_dict, add_dict_to_argparser)
from guided_diffusion.train_util import TrainLoop

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.distributed as dist
import yaml
from taming.models.vqgan import VQModel
import os





def setup_ddp():
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank), local_rank

def main():
    print("RUNNING RIGHT NOW-: T2W")
    dist.init_process_group(backend="nccl", init_method="env://")    
    args = create_argparser().parse_args()
    seed = args.seed
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    summary_writer = None
    if args.use_tensorboard:
        logdir = None
        if args.tensorboard_path:
            logdir = args.tensorboard_path
        summary_writer = SummaryWriter(log_dir=logdir)
        summary_writer.add_text(
            'config',
            '\n'.join([f'--{k}={repr(v)} <br/>' for k, v in vars(args).items()])
        )
        logger.configure(dir=summary_writer.get_logdir())
    else:
        logger.configure()

    #dist_util.setup_dist(devices=['0','1', '2', '3'])
    logger.log(args.devices)
    logger.log("Creating model and diffusion...")
    arguments = args_to_dict(args, model_and_diffusion_defaults().keys())
    model, diffusion = create_model_and_diffusion(**arguments)
    with open("/home/users/ntu/mohor001/cwdm-modified/scripts/vqgan_config.yaml", "r") as f:
        vq_config = yaml.safe_load(f)


    vq_model_config = vq_config["model"]["params"]
    vq_model_config["lossconfig"] = None  # Or use Identity if needed

    vq_model = VQModel(**vq_model_config, ckpt_path="/home/users/ntu/mohor001/scratch/vqgan_checkpoint.ckpt")

    vq_model.eval()
    
    # logger.log("Number of trainable parameters: {}".format(np.array([np.array(p.shape).prod() for p in model.parameters()]).sum()))
   # model.to(dist_util.dev([0, 1,2,3]) if len(args.devices) > 1 else dist_util.dev())  # allow for 2 devices
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=1000)
    data_path_train= ["/home/users/ntu/mohor001/scratch/Task8DataBrats/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData", "/home/users/ntu/mohor001/scratch/Task8DataBrats/ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData", "/home/users/ntu/mohor001/scratch/Task8DataBrats/ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData_Additional"] # to be filled


    ds_train = BraTS2021Train(data_path_train)
    sampler = DistributedSampler(ds_train, shuffle=True)
    
    datal = th.utils.data.DataLoader(ds_train,sampler=sampler,
                                     batch_size=args.batch_size,
                                     num_workers=args.num_workers,
                                     )

    logger.log("Start training...")
    print(f"[Rank {torch.distributed.get_rank()}] Number of samples: {len(sampler)}")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=datal,
        batch_size=args.batch_size,
        in_channels=args.in_channels,
        image_size=args.image_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        resume_step=args.resume_step,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        dataset=args.dataset,
        summary_writer=summary_writer,
        mode='i2i',
        # contr=args.contr,
        vqmodel=vq_model
    ).run_loop()


def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=5000,
        resume_checkpoint='',
        resume_step=0,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        dataset='brats',
        use_tensorboard=True,
        tensorboard_path='',  # set path to existing logdir for resuming
        devices=[0],
        dims=3,
        learn_sigma=False,
        num_groups=32,
        channel_mult="1,2,2,4,4",
        in_channels=8,
        out_channels=8,
        bottleneck_attention=False,
        num_workers=0,
        mode='default',
        renormalize=True,
        additive_skips=False,
        use_freq=False,
        contr='t1n',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
