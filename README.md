# NBNet: Noise Basis Learning for Image Denoising with Subspace Projection

Source model from https://github.com/leonmakise/NBNet_Pytorch

Paper https://arxiv.org/abs/2012.15028

## Requirement 

## Train
This repo. supports training on multiple GPUs.  

Train

```
CUDA_VISIBLE_DEVICES=0,1 python train.py --noise_dir ../image/noise/ --gt_dir ../image/gt/ --image_size 512 --batch_size 16 --save_every 100 --loss_every 10 -nw 4 -c -ckpt nbnet --restart```
```
If no `--restart`, the train process would be resumed.

## Test on SIDD Validate 

```
python test_custom_mat.py  -n /mnt/vinai/SIDD/ValidationNoisyBlocksSrgb.mat  -g /mnt/vinai/SIDD/ValidationGtBlocksSrgb.mat  -c -ckpt mir_kpn -m KPN
```

- ```--save_img``` to save image result