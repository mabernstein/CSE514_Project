import torch
import torch.nn.functional as F


def cross_entropy(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt, bl = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        print("Inconsistent Input and Target Size -- Adjust")
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)

    print(input.size())
    print(target.size())

    #Loss Function
    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss
