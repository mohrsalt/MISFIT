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


import torch as th
import torch.nn.functional as F
import yaml
import nibabel as nib
sys.path.append(".")

# reqd imports

from tools.taming.models.vqgan import VQModel
from tools.guided_diffusion import (dist_util,
                              logger)
from tools.brats import BraTS2021Test
from tools.guided_diffusion.script_util import (model_and_diffusion_defaults, create_model_and_diffusion,
                                          args_to_dict)
from tools.DWT_IDWT.DWT_IDWT_layer import IDWT_3D


def strip_module_prefix(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}

def load_args_from_yaml(yaml_path):


    with open(yaml_path, "r") as f:
        user_config = yaml.safe_load(f)

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
    defaults.update(user_config)  

    return argparse.Namespace(**defaults)





def main():
    args = load_args_from_yaml("./tools/Stage2.yaml")
    seed = args.seed
    dist_util.setup_dist(devices=args.devices)
    # logger.configure()

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    diffusion.mode = 'i2i'

    data_path_test=["./input"]
    ds = BraTS2021Test(data_path_test)

    datal = th.utils.data.DataLoader(ds,
                                     batch_size=args.batch_size,
                                     num_workers=12,
                                     shuffle=False,)

    model.eval()
    with open("./tools/Stage1.yaml", "r") as f:
        vq_config = yaml.safe_load(f)


    vq_model_config = vq_config["model"]["params"]
    vq_model_config["lossconfig"] = None  # Or use Identity if needed

    vq_model = VQModel(**vq_model_config, ckpt_path="./checkpoints/Stage1.ckpt")

    vq_model.eval()

    idwt = IDWT_3D("haar")
    

    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    for batch in iter(datal):
        subject_name = batch['subject_id'][0] #start from here and also tweak .sh file
        




        batch['t1n'] = batch['t1n'].to(dist_util.dev())
        batch['t1c'] = batch['t1c'].to(dist_util.dev())
        batch['t2w'] = batch['t2w'].to(dist_util.dev())
        batch['t2f'] = batch['t2f'].to(dist_util.dev())
        batch['source'] = batch['source'].to(dist_util.dev())
       
        batch['target_class'] = batch['target_class'].to(dist_util.dev())


        miss_name = args.output_dir + '/' + subject_name +'-' + batch['t_list'][0]+ '-'+"inference"
        print("Generation storage in :",miss_name)



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
            cond_caff=vq_model.forward_caff(input,src_idx)
            cond= th.cat([cond_dwt,cond_caff,h1,h2,h3], dim=1)


        
        header = nib.load(batch['header_path'][0]).header

            
        vq_model.to('cpu')
        th.cuda.empty_cache()
        statetoload=batch['t_list'][0]
        selected_model_path=""
        if statetoload=="t1n":
            selected_model_path="./checkpoints/Stage2_t1n.pt"        
        elif statetoload=="t1c":
            selected_model_path="./checkpoints/Stage2_t1c.pt"         
        elif statetoload=="t2w":
            selected_model_path="./checkpoints/Stage2_t2w.pt"        
        elif statetoload=="t2f":
            selected_model_path="./checkpoints/Stage2_t2f.pt"         

        print("Target is from: {}".format(statetoload)) #remove later
        print("Load model from: {}".format(selected_model_path))	



        state_dict = th.load(selected_model_path, map_location="cpu")
        state_dict = strip_module_prefix(state_dict)
        model.load_state_dict(state_dict)
	



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
        





if __name__ == "__main__":
    main()
















