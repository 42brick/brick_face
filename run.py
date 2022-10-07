--nimport numpy as np
import dnnlib
import torch
import legacy
import os
import argparse
from datetime import datetime

from style_mixing import generate_style_mix
from projector import run_projection

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--start',type=int, required = True)
    parser.add_argument('--end',type=int, required=True)
    parser.add_argument('--range',type=int, default=5, required = False)
    parser.add_argument('--seed', type = int, default=2022, required=False)
    parser.add_argument('--network','-n', type=str, default='../model/network-snapshot-001000.pkl', required = False)
    parser.add_argument('--test_mode', type=bool, default = False, required = False)
    parser.add_argument('--num_steps', type=int, default = 1000, required = False)
    parser.add_argument('--save_video','-s', type=bool, default =True, required = True)

    args = parser.parse_args()
    now = datetime.now()
    output_dir = './output/'+now.strftime('%Y%m%d_%H%M%S')
    test_mode = args.test_mode
    seed = args.seed
    network_pkl = args.network

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')

    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore

    '''
    Image1_url = "./test_images/test1.png"
    Image2_url = "./test_images/test2.png"
    '''

    # Image1, 2에 대한 latent vector를 생성
    w_dict = {}
    testfile_length = len(os.listdir('./test_images'))
    for i in range(1,testfile_length+1) :
        print(f"이미지를 생성합니다. ===> Input : {i}")#P
        projected_w = run_projection(G = G,
            target_fname = f'./test_images/test{i}.jpg', #
            outdir = output_dir,
            save_video = args.save_video, #if Test Mode, Do not Save Video
            seed = seed,#P
            num_steps = args.num_steps,
            device = device
        )
        w_dict[f"Image{i}"] = projected_w
    # print(w_dict["Image1"][0:6])

    # Style Mixing
    print("Style Mixing...")
    # Testing
    col_start = args.start
    col_end = args.end
    r = args.range
    if col_start < r :
        r = col_start-1
        print("RANGE가 자동으로 조정됩니다.")
    test_num = 0

    if test_mode == True :
        for i in range(col_start, col_end) :
            generate_style_mix(
                G = G,
                w_dict = w_dict,
                col_start = i-r,
                col_end = i,
                test_num = test_num,
                noise_mode='const',
                outdir=output_dir
            )
            test_num += 1
    else :
        # 파라미터 4~8이 적당해보임.(7~8섞으면 피부색 합성)
        generate_style_mix(
            G = G,
            w_dict = w_dict,
            col_start = col_start,
            col_end = col_end,
            test_num = test_num,
            noise_mode='const',
            outdir=output_dir
        )
    
