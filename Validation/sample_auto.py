"""
A script for sampling from a diffusion model for paired image-to-image translation.
"""

import argparse
import nibabel as nib
import numpy as np
import os
import pathlib
import random
import sys
from ssim import ssim
from scipy.stats import wilcoxon
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch as th
import torch.nn.functional as F
import yaml

sys.path.append(".")
from scripts.taming.models.vqgan import VQModel
from guided_diffusion import (dist_util,
                              logger)
from brats import BraTS2021Test
from guided_diffusion.script_util import (model_and_diffusion_defaults, create_model_and_diffusion,
                                          add_dict_to_argparser, args_to_dict)
from DWT_IDWT.DWT_IDWT_layer import IDWT_3D


def strip_module_prefix(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}

def get_psnr(x, y, data_range):
    EPS = 1e-8

    x = x / float(data_range)
    y = y / float(data_range)

    # if (x.size(1) == 3) and convert_to_greyscale:
    #     # Convert RGB image to YCbCr and take luminance: Y = 0.299 R + 0.587 G + 0.114 B
    #     rgb_to_grey = torch.tensor([0.299, 0.587, 0.114]).view(1, -1, 1, 1).to(x)
    #     x = torch.sum(x * rgb_to_grey, dim=1, keepdim=True)
    #     y = torch.sum(y * rgb_to_grey, dim=1, keepdim=True)

    mse = th.mean((x - y) ** 2, dim=[1, 2, 3, 4])
    score: th.Tensor = - 10 * th.log10(mse + EPS)
    return th.mean(score, dim = 0)




def main():
    args = create_argparser().parse_args()
    seed = args.seed
    dist_util.setup_dist(devices=args.devices)
    logger.configure()

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    diffusion.mode = 'i2i'

    data_path_test=["/home/users/ntu/mohor001/scratch/Task8DataBrats/pseudo_val_set"]
    ds = BraTS2021Test(data_path_test)

    datal = th.utils.data.DataLoader(ds,
                                     batch_size=args.batch_size,
                                     num_workers=12,
                                     shuffle=False,)

    model.eval()
    with open("/home/users/ntu/mohor001/cwdm-modified/scripts/vqgan_config.yaml", "r") as f:
        vq_config = yaml.safe_load(f)


    vq_model_config = vq_config["model"]["params"]
    vq_model_config["lossconfig"] = None  # Or use Identity if needed

    vq_model = VQModel(**vq_model_config, ckpt_path="/home/users/ntu/mohor001/scratch/vqgan_checkpoint.ckpt")

    vq_model.eval()

    idwt = IDWT_3D("haar")
    

    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    selected_model_path="/home/users/ntu/mohor001/scratch/cwchkpt/Jul14_07-40-32_x1000c0s0b0n0/checkpoints/brats_150000.pt"
    logger.log("Load model from: {}".format(selected_model_path))
    state_dict = th.load(selected_model_path, map_location="cpu")
    state_dict = strip_module_prefix(state_dict)

    model.load_state_dict(state_dict)

    ssim_list=[]
    psnr_list=[]
    for batch in iter(datal):
        subject_name = batch['subject_id'][0] #start from here and also tweak .sh file
        missing_target=batch['target_modality'][0]




        batch['t1n'] = batch['t1n'].to(dist_util.dev())
        batch['t1c'] = batch['t1c'].to(dist_util.dev())
        batch['t2w'] = batch['t2w'].to(dist_util.dev())
        batch['t2f'] = batch['t2f'].to(dist_util.dev())
        batch['source'] = batch['source'].to(dist_util.dev())
        batch['target'] = batch['target'].to(dist_util.dev())
        batch['target_class'] = batch['target_class'].to(dist_util.dev())


        miss_name = args.data_dir + '/' + subject_name + '/' + subject_name +'-' + "generated"
        print(miss_name)



        vq_model=vq_model.to(dist_util.dev())
        with th.no_grad():
            vq_model.eval()
  

        
            src_idx=vq_model.modalities_to_indices(batch["sources_list"])
            input=batch["source"]
            y = batch["target_class"].long()
            h1=vq_model.encode_noclamp(input[:,0].unsqueeze(1)) #(1,8,112,112,80)
            h2=vq_model.encode_noclamp(input[:,1].unsqueeze(1))
            h3=vq_model.encode_noclamp(input[:,2].unsqueeze(1))
            cond_dwt=vq_model.forward_latent(input, y,src_idx)
            cond= th.cat([cond_dwt,h1,h2,h3], dim=1)


        
        header = nib.load(batch['header_path'][0]).header

            
        vq_model.to('cpu')
        th.cuda.empty_cache()

        model.to(dist_util.dev())
        # Noise
        noise = th.randn(args.batch_size, 8, 112, 112, 80).to(dist_util.dev())

        model_kwargs = {}

        sample_fn = diffusion.p_sample_loop

        sample = sample_fn(model=model,
                           shape=noise.shape,
                           noise=noise,
                           cond=cond,
                           clip_denoised=args.clip_denoised,
                           model_kwargs=model_kwargs)

        B, _, D, H, W = sample.size()
        sample = idwt(sample[:, 0, :, :, :].view(B, 1, D, H, W) * 3.,
                      sample[:, 1, :, :, :].view(B, 1, D, H, W),
                      sample[:, 2, :, :, :].view(B, 1, D, H, W),
                      sample[:, 3, :, :, :].view(B, 1, D, H, W),
                      sample[:, 4, :, :, :].view(B, 1, D, H, W),
                      sample[:, 5, :, :, :].view(B, 1, D, H, W),
                      sample[:, 6, :, :, :].view(B, 1, D, H, W),
                      sample[:, 7, :, :, :].view(B, 1, D, H, W))

        sample[sample <= 0.04] = 0

        if len(sample.shape) == 5:
            sample = sample.squeeze(dim=1)  # don't squeeze batch dimension for bs 1

        # Pad/Crop to original resolution
        pad_sample = F.pad(sample, (0, 0, 8, 8, 8, 8), mode='constant', value=0)
        sample = pad_sample[:, :, :, :155]





        for i in range(sample.shape[0]):
            output_name = miss_name +'.nii.gz'
            img = nib.Nifti1Image(sample.detach().cpu().numpy()[i, :, :, :], None, header)
            nib.save(img=img, filename=output_name)
            print(f'Saved to {output_name}')
        
        for b in range(batch.shape[0]):
            output=sample.detach().cpu().numpy()[b, :, :, :]
            input_image=batch["target"][b]
            ssim_list.append(ssim(input_image, output))
            psnr_list.append(get_psnr(input_image, output , data_range=output.max() - output.min()))
        
    print(" average SSIM: ", np.mean(ssim_list))
    print(" average PSNR: ", np.mean(psnr_list))
    
    print(" std of SSIM: ", np.std(ssim_list))
    print(" std of PSNR: ", np.std(psnr_list))
    
    print(" p-value of SSIM: ", wilcoxon(ssim_list)[-1])
    print(" p-value of PSNR: ", wilcoxon(psnr_list)[-1])
            




def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        data_mode='validation',
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        class_cond=False,
        sampling_steps=0,
        model_path="",
        devices=[0],
        output_dir='./results',
        mode='default',
        renormalize=False,
        image_size=256,
        half_res_crop=False,
        concat_coords=False, # if true, add 3 (for 3d) or 2 (for 2d) to in_channels
        contr="",
    )
    defaults.update({k:v for k, v in model_and_diffusion_defaults().items() if k not in defaults})
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
















