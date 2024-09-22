import argparse
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.image_datasets import load_data
from torchvision import utils
from resizer import Resizer
import math


# added
def load_reference(data_dir, data_dir2, mask_dir2, batch_size, range_s, image_size, class_cond=False):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=class_cond,
        deterministic=True,
        random_flip=False,
    )
    if range_s:
        data2 = load_data(
            data_dir=data_dir2,
            batch_size=batch_size,
            image_size=image_size,
            class_cond=class_cond,
            deterministic=True,
            random_flip=False,
        )
    if mask_dir2:
        mask2 = load_data(
            data_dir=mask_dir2,
            batch_size=batch_size,
            image_size=image_size,
            class_cond=class_cond,
            deterministic=True,
            random_flip=False,
        )
    
    if range_s and mask_dir2:    
        for d1, d2, m2 in zip(data,data2,mask2):
            large_batch, model_kwargs = d1
            large_batch2, model_kwargs2 = d2
            mask_batch2, model_msargs2 = m2
            model_kwargs["ref_img"] = large_batch
            model_kwargs["ref_img2"] = large_batch2
            model_kwargs["msk_img2"] = mask_batch2
            model_kwargs["m"] = th.tensor(1)
            yield model_kwargs
    elif range_s:    
        for d1, d2 in zip(data,data2):
            large_batch, model_kwargs = d1
            large_batch2, model_kwargs2 = d2
            model_kwargs["ref_img"] = large_batch
            model_kwargs["ref_img2"] = large_batch2
            model_kwargs["m"] = th.tensor(0) 
            yield model_kwargs
    else:
        for large_batch, model_kwargs in data:
            model_kwargs["ref_img"] = large_batch
            yield model_kwargs


def main():
    args = create_argparser().parse_args()

    # th.manual_seed(0)

    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)

    logger.log("creating model...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("creating resizers...")
    assert math.log(args.down_N, 2).is_integer()

    if args.down_N1 == 0: 

        shape = (args.batch_size, 1, args.image_size, args.image_size)
        shape_d = (args.batch_size, 1, int(args.image_size / args.down_N), int(args.image_size / args.down_N))
        down = Resizer(shape, 1 / args.down_N).to(next(model.parameters()).device)
        up = Resizer(shape_d, args.down_N).to(next(model.parameters()).device)
        resizers = (down, up)

    else:
        assert math.log(args.down_N1, 2).is_integer()
        shape = (args.batch_size, 1, args.image_size, args.image_size)
        shape_d = (args.batch_size, 1, int(args.image_size / args.down_N), int(args.image_size / args.down_N))
        down = Resizer(shape, 1 / args.down_N).to(next(model.parameters()).device)
        up = Resizer(shape_d, args.down_N).to(next(model.parameters()).device)

        shape_d1 = (args.batch_size, 1, int(args.image_size / args.down_N1), int(args.image_size / args.down_N1))
        down1 = Resizer(shape, 1 / args.down_N1).to(next(model.parameters()).device)
        up1 = Resizer(shape_d1, args.down_N1).to(next(model.parameters()).device)

        resizers = (down, up, down1, up1)



    logger.log("loading data...")
    data = load_reference(
        args.base_samples,
        args.base_samples2,
        args.mask_samples2,
        args.batch_size,
        args.range_s,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("creating samples...")
    count = 0
    while count * args.batch_size < args.num_samples:
        model_kwargs = next(data)
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        sample = diffusion.p_sample_loop(
            model,
            (args.batch_size, 1, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            resizers=resizers,
            range_t=args.range_t,
            range_s=args.range_s
        )

        for i in range(args.batch_size):
            out_path = os.path.join(logger.get_dir(),
                                    f"{str(count * args.batch_size + i).zfill(5)}.png")
            utils.save_image(
                sample[i].unsqueeze(0),
                out_path,
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )

        count += 1
        logger.log(f"created {count * args.batch_size} samples")

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=5,
        down_N=32,
        down_N1=0,
        range_t=0,
        range_s=0,
        use_ddim=False,
        base_samples="",
        base_samples2="",
        mask_samples2="",
        model_path="",
        save_dir="",
        save_latents=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
