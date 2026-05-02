#import libraries 
from data.group_dataset import DMU_group
from data.compined_dataset import Group_collection
#from data.visualization import plot_efficiencies

from set_transformer_1.model import SetTransformer
from efficiency_scores import calculate_efficiencies

from torch.utils.data import random_split, DataLoader


if __name__ == "__main__":
    #you might need this line for an error!
    #os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
    
    #every group will have    
    n_dmus = 5
    n_inputs = 3
    n_outputs = 2
    
    input_range = [1,10]
    output_range = [1,40]
    
    n_groups = 10
    
    #dataset creation
    #TODO: na ginei mia methodos na fygei apo tin main
    dataset = Group_collection()
    for group_index in range(n_groups):
        current_dataset = DMU_group(n_dmus, n_inputs, n_outputs, input_range, output_range)
        current_dataset.data_generator()
        
        #calculate efficiencies
        for dmu_index in range(n_dmus):
        #     theta, s_minus, s_plus, lambdas = dea_input_oriented_extended(current_dataset.inputs, current_dataset.outputs, dmu_index)
        #     current_dataset.efficiencies.append(round(theta.item(),6))
            
            
        # dataset.add_group(current_dataset)    
    
    #plot_efficiencies(my_dataset.efficiencies)
            theta, s_minus, s_plus, lambdas = calculate_efficiencies(current_dataset.inputs, current_dataset.outputs, dmu_index)
            current_dataset.efficiencies.append(round(theta.item(),6)) 
        
        dataset.add_group(current_dataset)
    
    #manipulate data to the correct form 
    dataset.data_manipulation()
    
    #train-test split
    train_size = int(0.8*len(dataset)) #80% 
    test_size = int(len(dataset)-train_size) #20%
    
    train_dataset, test_dataset = random_split(dataset,[train_size,test_size])
    
    #dataloaders
    train_loader = DataLoader(train_dataset,batch_size=4)
    test_loader = DataLoader(test_dataset,batch_size=2)

    #load model 
    dim_input = n_inputs + n_outputs
    
    
    model_1 = SetTransformer(
        dim_input = dim_input,
        num_outputs =1,
        dim_output = n_dmus
    )
    
    #run one single forward
    X, y = next(iter(train_loader))
    
    output = model_1(X)
    print(output)
    print(output.shape)
    
    
    #feed model (train data)
     
    #evaluate model (test data)