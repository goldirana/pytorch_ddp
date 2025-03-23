import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.optim as optim
import torch
from src.create_data import my_data



# Define model
class model(nn.Module):
    def __init__(self):
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
    
    # wrap the model
    model = DDP(model().to(device), device_ids=[rank])
    
    # get the data
    data = my_data(1000)
    data = data.data
    label = data.label
    
    # criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs.labels)
        loss.backward()
        optimizer.step()
        
        print(f"Rank {rank} Loss: {loss.item()}")
    
    # clean up
    dist.destroy_process_group()
    

# entry point
def main():
    world_size = 2
    mp.spawn(train, args=(world_size,), nprocs=world_size)
    