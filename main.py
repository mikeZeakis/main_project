#import libraries 
from data import DMU_group


if __name__ == "__main__":
    
    #data configuration for 1 group 
    #in training we will need N groups 
    
    n_dmus = 10
    n_inputs = 5
    n_outputs = 3

    input_range = [1,10]
    output_range = [5,40]
    
    
    my_dataset = DMU_group(n_dmus, n_inputs, n_outputs, input_range, output_range)
    
    #generate data 
    my_dataset.data_generator()    
    
    print("e?")
    #calculate efficiencies 
    
    #manipulate data to the correct form 
    
    #train-test split 
    
    #load model 
    
    #feed model (train data)
     
    #evaluate model (test data)   