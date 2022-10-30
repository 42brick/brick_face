import torch
from torchvision import utils
import legacy
import dnnlib
import numpy as np
import os
from PIL import Image
from projector import run_projection
import argparse
import random
import matplotlib.pyplot as plt

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", default='./image')  # test용
    parser.add_argument("-s", "--seed", default=42)  # test용
    parser.add_argument("-n", "--num_steps", default=500)  # test용
    parser.add_argument("-v", "--version", default = 2) #test용
    parser.add_argument('-p',"--path",type=str, default='')
    parser.add_argument('-o',"--outdir",type=str,default='')
    args = parser.parse_args()

    seed = args.seed
    num_steps = int(args.num_steps)
    version = args.version
    outdir = args.outdir
    image = args.image
    path = args.path

    # Cuda
    device = torch.device('cuda')

    # seed 지정
    #if seed == -1 :
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # version change
    if version == 1 :
        resolution = '1024'
    else :
        resolution = '256'

    lego_network = f'./checkpoints/lego{resolution}.pkl'
    with dnnlib.util.open_url(lego_network) as fp:
        legoG = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)  # type: i

    face_network = f'./checkpoints/face{resolution}.pkl'
    with dnnlib.util.open_url(face_network) as fp2:
        faceG = legacy.load_network_pkl(fp2)['G_ema'].requires_grad_(False).to(device)

    if path == '' :

        # Face Latent Vector 생성 (14,512)
        '''
        (14,512) -> 256  : face256, lego.pkl 
        (18,512) -> 1024 : face.pkl
        이건 맞춰줘야해서 face.pkl 쓰고 싶음 1024환경에서 train해야함.
        '''
        projected_w = run_projection(G = faceG,
            target_fname = image, #
            outdir = outdir,
            save_video = False, #if Test Mode, Do not Save Video
            seed = seed,#P
            num_steps = num_steps,
            device = device)

        # 학습시킨 Latent
        # lego_latent = torch.cat((projected_w[:9],projected_w[:9]), 0)
        face2lego = legoG.synthesis(projected_w.unsqueeze(0), noise_mode='const')
        face2lego = (face2lego + 1) * (255 / 2)
        face2lego = face2lego.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        image_name = image.split('/')[-1]
        image_name = image_name.split('.')[0] + '_LEGO'

        Image.fromarray(face2lego, 'RGB').save(f'{outdir}/{image_name}.png')

        

    else :

        image_path = os.listdir(path)
        for img_p in image_path :
            image = f'{path}/{img_p}'
            projected_w = run_projection(G = faceG,
                target_fname = image, #
                outdir = outdir,
                save_video = False, #if Test Mode, Do not Save Video
                seed = seed,#P
                num_steps = num_steps,
                device = device)

            face2lego = legoG.synthesis(projected_w.unsqueeze(0), noise_mode='const')
            face2lego = (face2lego + 1) * (255 / 2)
            face2lego = face2lego.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            image_name = image.split('/')[-1]
            image_name = image_name.split('.')[0] + '_LEGO'

            Image.fromarray(face2lego, 'RGB').save(f'{outdir}/{image_name}.png')
