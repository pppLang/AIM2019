import torch
import warnings


class _Loss(torch.nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        self.reduction = reduction
            

def l1_focal_loss(input, target, weight, reduction='mean'):
    if not (target.size() == input.size()):
        warnings.warn(
            "Input size({}) is not same with target size({}).".format(target.size(), input.size()),
            stacklevel=2
        )

    ret = torch.abs(input - target) * weight
    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)

    return ret


class L1_Focal_Loss(_Loss):
    def __init__(self, reduction='mean'):
        super(L1_Focal_Loss, self).__init__(reduction)

    def forward(self, input, target):
        weight = torch.exp(2.5*torch.abs(input - target))
        # print(torch.abs(input - target))
        # print(torch.abs(input - target).mean())
        # print(weight)
        return l1_focal_loss(input, target, weight)


if __name__ == "__main__":
    import sys
    sys.path.append('../')
    from evalution import new_psnr
    cretion = L1_Focal_Loss()
    cretion2 = torch.nn.L1Loss()
    # a = torch.rand(16, 3, 64, 64)
    # b = torch.rand(16, 3, 64, 64)
    a = torch.rand(6, 6)
    # b = torch.rand(6, 6)
    b = a + 0.038
    loss = cretion(a, b)
    print(loss)
    print(cretion2(a, b))
    print(new_psnr(a, b))