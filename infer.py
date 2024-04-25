from model import *
from ssim import *
import torch
from torchvision import transforms
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import os

def denoise(img, noise_level= 15, already_noisy= False):
    model = Model()
    model.eval()

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    model = model.to(device)
    model.load_state_dict(torch.load('simple_cnn_best.pth'))

    totensor = transforms.PILToTensor()
    img_tensor = totensor(img).float()

    if not already_noisy:
        noise = torch.FloatTensor(img_tensor.size()).normal_(mean=0, std= noise_level)
        img_noisy = img_tensor + noise

        img_noisy = img_noisy.to(device).unsqueeze(0)
    else:
        img_noisy = img_tensor.to(device).unsqueeze(0)

    with torch.no_grad():
        denoised = model(img_noisy)

    denoised = denoised.squeeze().cpu().numpy()

    ssim_map, mssim = SSIM(img, denoised)
    _, psnr = MSE(img, denoised)

    return denoised, ssim_map, mssim, psnr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path')
    parser.add_argument('--save_dir', default= 'run1')
    parser.add_argument('--already_noisy', action= 'store_true')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    img = Image.open(args.img_path).convert('L')

    denoised, ssim_map, mssim, psnr = denoise(img, 15, args.already_noisy)

    log = open(rf'{args.save_dir}\results.txt', 'w+')
    log.write(f'avg_ssim= {mssim}, avg_psnr= {psnr}')
    log.close()

    plt.imsave(f'{args.save_dir}\denoised_img.png', denoised, cmap= 'gray')
    plt.imsave(f'{args.save_dir}\ssim_map.png', ssim_map, cmap= 'gray')


