import torch

def interpolate(rr, map_size, sigma=10.0):

    N,  sh, sw = map_size

    # rr : N x K x 3
    K = rr.size(1)

    ig = torch.stack([
        torch.linspace(-1, 1, sw).view(1, -1).repeat(sh, 1),
        torch.linspace(-1, 1, sh).view(-1, 1).repeat(1, sw),
    ], 0).to(rr.device)
    ig = ig.view(1, 2, sh, sw).repeat(N, 1, 1, 1)

    #rr = torch.randn(N, K, 3)

    c = rr[:, :, :2] #.tanh()
    w = 1e-2 * rr[:, :, 2] #.tanh()

    if rr.size(2) == 4:
        sigma = rr[:, :, 3].view(N, K, 1, 1)

    # N x K x sh x sw
    ig = ig.unsqueeze(1)
    c = c.view(N, K, 2, 1, 1)

    return (torch.exp(- (ig - c).pow(2).sum(dim=2) * sigma) * w.view(N, K, 1, 1)).sum(dim=1)