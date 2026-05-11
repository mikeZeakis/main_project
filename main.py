#import libraries 
from data.group_dataset import DMU_group
from data.compined_dataset import Group_collection
#from data.visualization import plot_efficiencies

from set_transformer_1.model import SetTransformer
from efficiency_scores import calculate_efficiencies

from torch.utils.data import random_split, DataLoader

import torch
import torch.nn as nn

def train_one_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()

    total_loss = 0.0

    for X, y in train_loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        preds = model(X)
        loss = loss_fn(preds, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def test(model, test_loader, loss_fn, device):
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)

            preds = model(X)
            loss = loss_fn(preds, y)

            total_loss += loss.item()
            
            print(f"\nthe predictions: {preds}")
            print(f"the target {y}\n")
            
    avg_loss = total_loss / len(test_loader)
        
    return avg_loss

def show_test_predictions(model, test_loader, device, samples=5):
    model.eval()

    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            if i == samples: break
            X = X.to(device)
            y = y.to(device)

            preds = model(X).squeeze(1)

            target = y.cpu().numpy().flatten()
            prediction = preds.cpu().numpy().flatten()

            print(f"\nSample {i+1}")
            for dmu_idx, (t, p) in enumerate(zip(target, prediction)):
                print(f"DMU {dmu_idx+1}: target={t:.4f}, prediction={p:.4f}")


if __name__ == "__main__":
    #you might need this line for an error!
    #os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
    
    #every group will have    
    n_dmus = 10
    n_inputs = 4
    n_outputs = 3
    
    input_range = [1,10]
    output_range = [1,40]
    
    n_groups = 100
    
    #dataset creation
    #TODO: na ginei mia methodos na fygei apo tin main
    print("Dataset creation...")
    dataset = Group_collection()
    for group_index in range(n_groups):
        current_dataset = DMU_group(n_dmus, n_inputs, n_outputs, input_range, output_range)
        current_dataset.data_generator()
        
        #calculate efficiencies
        for dmu_index in range(n_dmus):
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
    test_loader = DataLoader(test_dataset,batch_size=1)

    #load model 
    dim_input = n_inputs + n_outputs
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    model_1 = SetTransformer(
        dim_input = dim_input,
        num_outputs =1,
        dim_output = n_dmus
    ).to(device)
    
    # loss and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model_1.parameters(), lr=1e-3)
    
    # training loop
    num_epochs = 20
    
    print("Starting training...")
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model=model_1,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device
        )
    
        # test_loss = test(
        #     model=model_1,
        #     test_loader=test_loader,
        #     loss_fn=loss_fn,
        #     device=device
        # )
    
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            #f"Test Loss: {test_loss:.6f}"
        )
        
    
    #testing 
    show_test_predictions(model_1, test_loader, device)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #feed model (train data)
     
    #evaluate model (test data)