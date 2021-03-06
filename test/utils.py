import torch

#dtypes = [torch.float, torch.double]
dtypes = [torch.double]

devices = [torch.device('cpu')]
if torch.cuda.is_available():
    devices += [torch.device('cuda:{}'.format(torch.cuda.current_device()))]


def tensor(x, dtype, device):
    return None if x is None else torch.tensor(x, dtype=dtype, device=device)
