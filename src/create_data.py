import torch 
from torch.utils.data import Dataset

class my_data(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]
        self.label = torch.rand(size)
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):  
        return self.data[index]
    
if __name__ == "__main__":
    data = my_data(1000)
    