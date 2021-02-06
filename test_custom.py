import torch
import argparse
from model.NBNet import NBNet
from utils.metric import calculate_psnr,calculate_ssim
import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from utils.training_util import load_checkpoint
from PIL import Image
import glob
import time
# from torchsummary import summary

torch.set_num_threads(4)
torch.manual_seed(0)
torch.manual_seed(0)

def test(args):
    model = NBNet()
    checkpoint_dir = args.checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # try:
    checkpoint = load_checkpoint(checkpoint_dir, device == 'cuda', args.load_type)
    start_epoch = checkpoint['epoch']
    global_step = checkpoint['global_iter']
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    print('=> loaded checkpoint (epoch {}, global_step {})'.format(start_epoch, global_step))
    # except:
    #     print('=> no checkpoint file to be loaded.')    # model.load_state_dict(state_dict)
    #     exit(1)
    model.eval()
    model = model.to(device)
    trans = transforms.ToPILImage()
    torch.manual_seed(0)
    noisy_path = sorted(glob.glob(args.noise_dir+ "/*.png"))
    clean_path = [ i.replace("noisy","clean") for i in noisy_path]
    print(noisy_path)
    for i in range(len(noisy_path)):
        noise = transforms.ToTensor()(Image.open(noisy_path[i]).convert('RGB'))[:, 0:args.image_size, 0:args.image_size].unsqueeze(0)
        noise = noise.to(device)
        begin = time.time()
        print(noise.size())
        pred = model(noise)
        pred = pred.detach().cpu()
        gt = transforms.ToTensor()(Image.open(clean_path[i]).convert('RGB'))[:, 0:args.image_size, 0:args.image_size]
        gt = gt.unsqueeze(0)
        psnr_t = calculate_psnr(pred, gt)
        ssim_t = calculate_ssim(pred, gt)
        print(i,"   UP   :  PSNR : ", str(psnr_t)," :  SSIM : ", str(ssim_t))
        if save_img != '':
            if not os.path.exists(args.save_img):
                os.makedirs(args.save_img)
            plt.figure(figsize=(15, 15))
            plt.imshow(np.array(trans(pred[0])))
            image_name = noisy_path[i].split("/")[-1].split(".")[0]
            plt.axis("off")
            plt.suptitle(image_name+"   UP   :  PSNR : "+ str(psnr_t)+" :  SSIM : "+ str(ssim_t), fontsize=25)
            plt.savefig( os.path.join(args.save_img,image_name + "_" + args.checkpoint + '.png'),pad_inches=0)


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='parameters for training')
    parser.add_argument('--noise_dir','-n', default='/home/dell/Downloads/FullTest/noisy', help='path to noise image file')
    # parser.add_argument('--noise_dir','-n', default='/home/dell/Downloads/noise/0001_NOISY_SRGB', help='path to noise image file')
    parser.add_argument('--cuda', '-c', action='store_true', help='whether to train on the GPU')
    parser.add_argument('--checkpoint', '-ckpt', type=str, default='checkpoints',
                        help='the checkpoint to eval')
    parser.add_argument('--image_size', '-sz', default=64, type=int, help='size of image')
    parser.add_argument('--save_img', "-s" ,default="", type=str, help='save image in eval_img folder ')
    parser.add_argument('--load_type', "-l" ,default="best", type=str, help='Load type best_or_latest ')

    args = parser.parse_args()
    #
    # args.noise_dir = '/home/dell/Downloads/FullTest/noisy'
    test(args)

