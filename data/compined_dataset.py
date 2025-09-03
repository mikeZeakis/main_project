import torch 
from torch.utils.data import Dataset
from group_dataset import DMU_group

class Group_collection(Dataset):
    
    def __init__(self):
        self.collection = []
        
        
    def add_group(self, group):
        self.collection.append(group)
        
        
    def __len__(self):
        return len(self.collection)
    
    
    def __getitem__(self, index):
        
        curr_group = self.collection[index]
        
        #convert group into tensor
        tensor_group = torch.tensor(list(curr_group.elements), dtype=torch.float32)

        return tensor_group