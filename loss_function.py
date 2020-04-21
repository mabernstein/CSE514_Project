import torch
import torch.nn.functional as F


def cross_entropy(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt, bl = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        print("Inconsistent Input and Target Size -- Adjust")
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)\

    print(input.size())
    print(target.size())

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)

    print(input.size())
    print(target.size())

    #Loss Function
    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss


def multi_scale_cross_entropy2d(input, target, weight=None, size_average=True, scale_weight=None):
    if not isinstance(input, tuple):
        return cross_entropy(input=input, target=target, weight=weight, size_average=size_average)

    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight is None:  # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp).float()).to(
            target.device
        )

    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy(
            input=inp, target=target, weight=weight, size_average=size_average
        )

    return loss
