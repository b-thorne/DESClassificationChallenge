import platform
import torch


def set_device():
    if platform.system() == "Darwin":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
