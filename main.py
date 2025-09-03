#import libraries 
from data.group_dataset import DMU_group
from efficiency_scores import dea_input_oriented_extended


if __name__ == "__main__":
    
    #data configuration for 1 group 
    #in training we will need N groups 
    
    n_dmus = 10
    n_inputs = 5
    n_outputs = 3

    input_range = [1,100]
    output_range = [1,400]
    
    
    my_dataset = DMU_group(n_dmus, n_inputs, n_outputs, input_range, output_range)
    
    #generate data 
    my_dataset.data_generator()    
    
    X = my_dataset.inputs
    Y = my_dataset.outputs
    
    #calculate efficiencies
    for dmu_index in range(n_dmus):
        theta, s_minus, s_plus, lambdas = dea_input_oriented_extended(X, Y, dmu_index)
        my_dataset.efficinces.append(round(theta.item(),6))
    print(f"These are the efficincy scores for one groupf of {n_dmus} DMUs: \n {my_dataset.efficinces}")
    
    #manipulate data to the correct form 
    
    #train-test split 
    
    #load model 
    
    #feed model (train data)
     
    #evaluate model (test data)   