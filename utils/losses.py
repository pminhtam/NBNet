import torch
import torch.nn as nn
import torch.nn.functional as F




def tv_loss(x, beta = 0.5, reg_coeff = 5):
    '''Calculates TV loss for an image `x`.
        
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta` 
    '''
    dh = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2)
    dw = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2)
    a,b,c,d=x.shape
    return reg_coeff*(torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta))/(a*b*c*d))

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]



class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class AlginLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(AlginLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        y = F.pad(y,[1,1,1,1])
        diff0 = torch.abs(x-y[:,:,1:-1,1:-1])
        diff1 = torch.abs(x-y[:,:,0:-2,0:-2])
        diff2 = torch.abs(x-y[:,:,0:-2,1:-1])
        diff3 = torch.abs(x-y[:,:,0:-2,2:])
        diff4 = torch.abs(x-y[:,:,1:-1,0:-2])
        diff5 = torch.abs(x-y[:,:,1:-1,2:])
        diff6 = torch.abs(x-y[:,:,2:,0:-2])
        diff7 = torch.abs(x-y[:,:,2:,1:-1])
        diff8 = torch.abs(x-y[:,:,2:,2:])
        diff_cat = torch.stack([diff0, diff1, diff2, diff3, diff4, diff5, diff6, diff7, diff8])
        diff = torch.min(diff_cat,dim=0)[0]
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

if __name__ == "__main__":
    x = torch.rand((3,16,16))
    y = torch.rand((3,16,16))
    # y = x
    y = F.pad(y, [1, 1, 1, 1])
    print(y[:,1:-1, 1:-1].size())
    diff0 = torch.abs(x - y[:,1:-1, 1:-1])
    diff1 = torch.abs(x - y[:,0:-2, 0:-2])
    diff2 = torch.abs(x - y[:,0:-2, 1:-1])
    diff3 = torch.abs(x - y[:,0:-2, 2:])
    diff4 = torch.abs(x - y[:,1:-1, 0:-2])
    diff5 = torch.abs(x - y[:,1:-1, 2:])
    diff6 = torch.abs(x - y[:,2:, 0:-2])
    diff7 = torch.abs(x - y[:,2:, 1:-1])
    diff8 = torch.abs(x - y[:,2:, 2:])

    diff_cat = torch.stack([diff0, diff1, diff2, diff3, diff4, diff5, diff6, diff7, diff8])
    # print(diff0)
    # print(diff_cat.size())
    diff = torch.min(diff_cat, dim=0)
    print(diff[0].size())
    print(diff[0])