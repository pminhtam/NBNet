import torch.utils.data as data
import torch
from PIL import Image
import os
import os.path
import glob
import torchvision.transforms as transforms
import numpy as np
from data.dataset_utils import Augment_RGB_torch
import random
##

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
augment   = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')]

def random_cut(image_noise, image_gt, w, h=None):
    h = w if h is None else h
    nw = image_gt.size(-1) - w
    nh = image_gt.size(-2) - h
    if nw < 0 or nh < 0:
        raise RuntimeError("Image is to small {} for the desired size {}". \
                           format((image_gt.size(-1), image_gt.size(-2)), (w, h))
                           )

    idx_w = torch.randint(0, nw + 1, (1,))[0]
    idx_h = torch.randint(0, nh + 1, (1,))[0]
    image_noise_burst_crop = image_noise[:, idx_h:(idx_h + h), idx_w:(idx_w + w)]
    image_gt_crop = image_gt[:, idx_h:(idx_h + h), idx_w:(idx_w + w)]
    return image_noise_burst_crop, image_gt_crop
# --------------------------------------------
# inverse of pixel_shuffle
# --------------------------------------------
def pixel_unshuffle(input, upscale_factor):
    r"""Rearranges elements in a Tensor of shape :math:`(C, rH, rW)` to a
    tensor of shape :math:`(*, r^2C, H, W)`.

    Authors:
        Zhaoyi Yan, https://github.com/Zhaoyi-Yan
        Kai Zhang, https://github.com/cszn/FFDNet

    Date:
        01/Jan/2019
    """
    # print(input.size())
    channels, in_height, in_width = input.size()

    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    input_view = input.contiguous().view(
        channels, out_height, upscale_factor,
        out_width, upscale_factor)

    # channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(2, 4,0, 1, 3).contiguous()
    return unshuffle_out.view(upscale_factor ** 2, channels, out_height, out_width)

class SingleLoader(data.Dataset):
    """
    Args:

     Attributes:
        noise_path (list):(image path)
    """

    def __init__(self, noise_dir, gt_dir, image_size=512):

        self.noise_dir = noise_dir
        self.gt_dir = gt_dir
        self.image_size = image_size
        self.noise_path = []
        for files_ext in IMG_EXTENSIONS:
            self.noise_path.extend(glob.glob(self.noise_dir + "/**/*" + files_ext, recursive=True))
        self.gt_path = []
        for files_ext in IMG_EXTENSIONS:
            self.gt_path.extend(glob.glob(self.gt_dir + "/**/*" + files_ext, recursive=True))

        if len(self.noise_path) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + self.noise_dir + "\n"
                                                                                       "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.transforms = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, groundtrue) where image is a noisy version of groundtrue
        """
        # rand_hflip = torch.rand(1)[0]
        # rand_vflip = torch.rand(1)[0]
        image_noise = Image.open(self.noise_path[index]).convert('RGB')
        # name_image_gt = self.noise_path[index].split("/")[-1].replace("NOISY_", "GT_")
        # image_folder_name_gt = self.noise_path[index].split("/")[-2].replace("NOISY_", "GT_")
        # image_gt = Image.open(os.path.join(self.gt_dir, image_folder_name_gt, name_image_gt)).convert('RGB')
        image_gt = Image.open(self.noise_path[index].replace("Noisy",'Clean')).convert('RGB')
        # image_noise = random_flip(image_noise,rand_hflip,rand_vflip)
        # image_gt = random_flip(image_gt,rand_hflip,rand_vflip)
        image_noise = self.transforms(image_noise)
        image_gt = self.transforms(image_gt)
        image_noise, image_gt = random_cut(image_noise, image_gt, w=self.image_size)
        apply_trans = transforms_aug[random.getrandbits(3)]

        image_gt = getattr(augment, apply_trans)(image_gt)
        image_noise = getattr(augment, apply_trans)(image_noise)

        return image_noise, image_gt

    def __len__(self):
        return len(self.noise_path)



