import torch 
import numpy as np
from torch.utils.data import Dataset

class Group_collection(Dataset):
    
    def __init__(self):
        self.collection = [] # this is the list of groups that takes as input to 
                             #create tha dataset ready for the transformer
        self.X = []
        self.y = []
        
        
    #add a group of DMUs (class -> group_dataset)
    def add_group(self, group):
        self.collection.append(group)
        
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx],self.y[idx]
    
    
    def data_manipulation(self,dtype=torch.float32):
        
        for idx,curr_group in enumerate(self.collection):
            
            #ensure shapes are correct 
            inputs = np.asarray(curr_group.inputs)
            outputs = np.asarray(curr_group.outputs)
            efficiencies = np.asarray(curr_group.efficiencies).reshape(-1,1)
            
            #validate row count 
            assert inputs.shape[0] == outputs.shape[0] == efficiencies.shape[0]
            
            X_set = np.concatenate((inputs,outputs),axis=1)
            #curr_y = np.array(curr_group.efficiencies).reshape(-1,1)
            
            self.X.append(X_set)
            self.y.append(efficiencies)
            
        #convert dataset to tensors 
        self.X = torch.tensor(np.stack(self.X), dtype=dtype)
        self.y = torch.tensor(np.stack(self.y), dtype=dtype)   
    
    