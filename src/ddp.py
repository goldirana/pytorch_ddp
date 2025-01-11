import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.optim as optim
import torch



# Define model
class model(nn.Module):
    def __init__(self);
        super(model, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)
    

# Training Loop
def train(rank:int, world_size: int):
    # Initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # set device for this process
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
