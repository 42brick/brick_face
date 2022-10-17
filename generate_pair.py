import torch
from torchvision import utils
import legacy
import dnnlib
import numpy as np
import os
from PIL import Image
from projector import run_projection

# Cuda
device = torch.device('cuda')

# seed 지정
seed = 2022
np.random.seed(seed)
torch.manual_seed(seed)

# input Image
target_fname = f'./freezeg_test/input/human5.jpg'

# output dir
outdir = './freezeg_test/output'

lego_network = '../model/lego256.pkl'
with dnnlib.util.open_url(lego_network) as fp:
    legoG = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: i

face_network = '../model/face256.pkl'
with dnnlib.util.open_url(face_network) as fp2:
    faceG = legacy.load_network_pkl(fp2)['G_ema'].requires_grad_(False).to(device) # type: ignore


# Face Latent Vector 생성 (14,512)
'''
(14,512) -> 256  : face256, lego.pkl 
(18,512) -> 1024 : face.pkl
이건 맞춰줘야해서 face.pkl 쓰고 싶음 1024환경에서 train해야함.
'''
projected_w = run_projection(G = faceG,
    target_fname = target_fname, #
    outdir = outdir,
    save_video = True, #if Test Mode, Do not Save Video
    seed = seed,#P
    num_steps = 800,
    device = device)

# 학습시킨 Latent
# lego_latent = torch.cat((projected_w[:9],projected_w[:9]), 0)
face2lego = legoG.synthesis(projected_w.unsqueeze(0), noise_mode='const')
face2lego = (face2lego + 1) * (255 / 2)
face2lego = face2lego.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

Image.fromarray(face2lego, 'RGB').save(f'{outdir}/face2lego.png')

