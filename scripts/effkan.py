import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import time
import gc
import numpy as np
import tqdm
import pandas as pd

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from efficient_kan import KAN
from collections import namedtuple

torch.use_deterministic_algorithms(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)

def run_efficient_kan(trainloader, valloader, testloader, 
                      epochs, architecture, grid_size, seed, learning_rate):
    '''
    Trains and evaluates an Efficient KAN model using the given training, validation, and test data.

    Parameters:
    -----------
    trainloader : DataLoader
        DataLoader object for the training data.
    valloader : DataLoader
        DataLoader object for the validation data.
    testloader : DataLoader
        DataLoader object for the test data.
    epochs : int
        The number of training epochs.
    architecture : list
        A list specifying the architecture of the network, formatted as 
                    [input_size, hidden_layer_1_size, ..., output_size].
    grid_size : int
        The grid size (G) for the model.
    seed : int
        The random seed for reproducibility.
    learning_rate : float
        The learning rate for the optimizer.

    Returns:
    --------
    list
        A list containing the following elements:
        - execution_time: float, the total time taken for model training and evaluation.
        - accuracy: float, the accuracy of the model on the test set.
        - f1: float, the F1-score of the model on the test set.
        - precision: float, the precision of the model on the test set.
        - recall: float, the recall of the model on the test set.
    '''
    
    # Define model
    model_effkan = KAN(architecture, grid_size=grid_size, 
                        spline_order=3, random_seed=seed).to(device)

    
    # Define optimizer
    optimizer = optim.Adam(model_effkan.parameters(), lr=learning_rate)

    # Define Loss function
    criterion = nn.CrossEntropyLoss()

    # Start measuring time for training
    start_time_effkan = time.time()

    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        running_accuracy = 0.0

        model_effkan.train() # Set the model to training mode

        # Iterate over the training data loader
        with tqdm(trainloader, desc=f"Training Epoch {epoch+1}", disable=True) as pbar:
            for data, labels in pbar:
                data, labels = data.to(device), labels.to(device)  # Move data to GPU
                optimizer.zero_grad()  # Reset gradients
                outputs = model_effkan(data)  # Forward pass
                loss = criterion(outputs, labels)  # Calculate loss
                loss.backward()  # Backward pass (calculate gradients)
                optimizer.step()  # Update model weights
                
                running_loss += loss.item()  # Accumulate loss
                running_accuracy += (outputs.argmax(dim=1) == labels).float().mean().item()  # Accumulate accuracy
                pbar.set_postfix(loss=running_loss/len(pbar), accuracy=running_accuracy/len(pbar), lr=optimizer.param_groups[0]['lr'])  # Update progress bar

        # Validation phase (set model to evaluation mode)
        model_effkan.eval()
        val_loss = 0
        val_accuracy = 0

        # Iterate over the validation data loader without computing gradients
        with torch.no_grad():
            for data, labels in tqdm(valloader, desc=f"Validation Epoch {epoch+1}", disable=True):
                data, labels = data.to(device), labels.to(device)
                outputs = model_effkan(data)  # Forward pass on validation set
                val_loss += criterion(outputs, labels).item()  # Accumulate validation loss
                val_accuracy += (outputs.argmax(dim=1) == labels).float().mean().item()  # Accumulate validation accuracy
        
        val_loss /= len(valloader)  # Average validation loss
        val_accuracy /= len(valloader)  # Average validation accuracy

        #print(f"Epoch {epoch + 1}, Train Loss: {running_loss/len(trainloader):.4f}, Train Accuracy: {running_accuracy/len(trainloader):.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # End of training: calculate total execution time
    end_time_effkan = time.time()
    execution_time = end_time_effkan - start_time_effkan

    # Test phase: set model to evaluation mode
    model_effkan.eval()

    all_preds = []
    all_labels = []

    # Predict on test data
    with torch.no_grad():
        for data, labels in testloader:
            data, labels = data.to(device), labels.to(device)  # Move data to GPU
            outputs = model_effkan(data)  # Forward pass
            _, preds = torch.max(outputs, 1)  # Get predicted class labels
            all_preds.append(preds.cpu().numpy())  # Store predictions
            all_labels.append(labels.cpu().numpy())  # Store true labels

    # Concatenate predictions and true labels across batches
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Calculate performance metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    del model_effkan
    gc.collect()

    return [execution_time, accuracy, f1, precision, recall]
    
        

def preprocess(df_train, df_test, batch_size):
    '''
    Arguments:
    ----------
    df_train : pandas.DataFrame
        The training dataset, where the first column contains the labels and the remaining columns are features.
    df_test : pandas.DataFrame
        The test dataset, where the first column contains the labels and the remaining columns are features.
    batch_size : int
        The batch size to be used for DataLoaders.

    Returns:
    --------
    tuple or None
        - If successful, returns a tuple with:
            1. trainloader : DataLoader
                PyTorch DataLoader containing batches of the training dataset.
            2. valloader : DataLoader
                PyTorch DataLoader containing batches of the validation dataset (80% training split).
            3. testloader : DataLoader
                PyTorch DataLoader containing batches of the test dataset.
        - Returns `None` if the input data is invalid or empty.
    '''

    # Drop instances with missing values
    df_train = df_train.dropna()
    df_test = df_test.dropna()

    if df_train.empty or df_test.empty:
        return None

    # Preprocess labels
    y_train = df_train.iloc[:, 0]
    y_test = df_test.iloc[:, 0]
    unique = y_test.unique()

    if -1 in unique:
        y_train = y_train.replace({-1: 0})
        y_test = y_test.replace({-1: 0})
    
    elif 0 not in unique:
        min_label = min(unique)
        y_train = y_train - min_label
        y_test = y_test - min_label

    df_train.iloc[:, 0] = y_train
    df_test.iloc[:, 0] = y_test

    # Split training into training and valdation part 80:20 ratio
    try:
        df_train, df_val= train_test_split(df_train, test_size=0.2, random_state=42)
    except ValueError as e:
        return None
    
    # Normalize input values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train.iloc[:, 1:])
    X_val = scaler.fit_transform(df_val.iloc[:, 1:])
    X_test = scaler.fit_transform(df_test.iloc[:, 1:])
    
    # Labels
    y_train = df_train.iloc[:, 0]
    y_val = df_val.iloc[:, 0]
    y_test = df_test.iloc[:, 0]

    # Make tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).clone().detach()
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)

    trainloader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=False)
    valloader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)
    testloader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)
    
    return trainloader, valloader, testloader
       
def multirun(params, lr, epochs, batch_size, seeds):
    '''
    Arguments:
    ----------
    params : list
        List of instances, each containing architecture and grid size configuration.
    lr : float
        Learning rate for the Efficient KAN model.
    epochs : int
        Number of epochs for training.
    batch_size : int
        Batch size for DataLoader.
    seeds : list
        List of random seeds for reproducibility.

    Returns:
    --------
    None
        Results are saved in a pickle file after each dataset run.
    '''
    # File with UCR Metadata
    df_meta = pd.read_csv("DataSummary.csv")

    # Run each architecture configuration
    for i, instance in enumerate(params):
        architecture = instance.architecture
        architecture_str = ','.join(map(str, architecture))

        grid = instance.grid

        file_path = f'results/effkan/LearningRate{str(lr).replace(".", ",")}/results_{architecture_str}_{grid}.pkl'
        
        # Load already computed results if they exist
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                results = pickle.load(file)
        # Create empty ditionary for saving results        
        else:
            results = {}

        # Skip datasets with existing results
        keys_to_remove = results.keys()
        df_meta = df_meta[~df_meta['Name'].isin(keys_to_remove)]

        # Run each Dataset
        for index, dataset in df_meta.iterrows():
            name = dataset['Name']
            input_layer = dataset['Length']
            output_layer = dataset['Class']
            architecture = [int(input_layer)] + instance.architecture + [int(output_layer)]
            g = instance.grid

            # Skip datasets with varying input lengths
            if input_layer == 'Vary':
                continue

            # Load files for training and testing 
            file_train = f'{name}/{name}_TRAIN.tsv'
            file_test = f'{name}/{name}_TEST.tsv'
            
            df_train = pd.read_csv('UCRArchive_2018/' + file_train, sep='\t')
            df_test = pd.read_csv('UCRArchive_2018/' + file_test, sep='\t')

            data = preprocess(df_train, df_test, batch_size)
            if data is None:
                continue

            trainloader, valloader, testloader = data

            # Run Efficient KAN model for each seed
            for seed in seeds:
                res= run_efficient_kan(trainloader, valloader, testloader, 
                                        epochs, architecture, g, seed, lr)

                results.setdefault(name, []).append(res)
            
                # Update results in file each time
                with open(file_path, 'wb') as f:
                    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    Params = namedtuple('Params', ['architecture', 'grid'])

    # Architecture configurations
    effkan_params = [
        Params(architecture=[5, 5], grid=5),
    ]
    
    # Define parameters
    learning_rates=[0.0001, 0.001, 0.01, 0.1, 1]
    seeds=[0,1,2,5,42]
    epochs=500
    batch=16

    for lr in learning_rates:
        multirun(effkan_params, lr, epochs, batch, seeds)


if __name__ == "__main__":
    main()
    