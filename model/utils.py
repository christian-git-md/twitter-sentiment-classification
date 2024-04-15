import torch


def get_available_device():
    """
    Returns the first available gpu as a torch device, if available, otherwise cpu.
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")
