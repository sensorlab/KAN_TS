import torch
import pickle
import os
import time
import gc
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, '../pykan/')
from kan import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from collections import namedtuple

torch.use_deterministic_algorithms(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)

def run_kan(dataset, epochs, batch_size, grid_size, architecture, 
            X_test, y_test, seed, learning_rate):
    '''
    Parameters:
    -----------
    dataset : dict
        A dictionary with 4 elements 
        ('train_input', 'train_label', 'test_input', 'test_label'),
        where the values are PyTorch tensors.
    epochs : int
        The number of epochs to run.
    batch_size : int
        The batch size for training.
    grid_size : int
        The grid size (G) for the model.
    architecture : list
        A list specifying the architecture of the network, formatted as 
                    [input_size, hidden_layer_1_size, ..., output_size].
    X_test : torch.Tensor
        A tensor containing the test data.
    y_test : torch.Tensor
        A tensor containing the test labels.
    seed : int
        The random seed for reproducibility.
    learning_rate : float
        The learning rate for the optimizer.

    Returns:
    --------
    list
        A list containing the following elements:
        - execution_time: float, the total time taken for model execution.
        - accuracy: float, the accuracy of the model on the test set.
        - f1: float, the F1-score of the model on the test set.
        - precision: float, the precision of the model on the test set.
        - recall: float, the recall of the model on the test set.
    '''
    try:
        def train_acc():
            return torch.mean((torch.argmax(model_kan(dataset['train_input']), dim=1) == dataset['train_label']).float())

        def test_acc():
            return torch.mean((torch.argmax(model_kan(dataset['test_input']), dim=1) == dataset['test_label']).float())
   
        # Number of test instances
        length = dataset['test_input'].shape[0]
    
        # Decrease batch size if there are not enough instances
        if length < batch_size:
            batch_size = length

        # Define model
        model_kan = KAN(width=architecture, grid=grid_size, k=3, 
                    symbolic_enabled=False, seed=seed, device=device).to(device)
        
        # Start measuring time for training
        start_time_kan = time.time()

        # Train
        model_kan.fit(dataset, opt="Adam", steps=int(epochs), metrics=(train_acc, test_acc), 
                      batch=batch_size, lr=learning_rate, 
                      loss_fn=torch.nn.CrossEntropyLoss(), lamb_l1=0.1)
        end_time_kan = time.time()
    
        # Calculate time passed to train
        execution_time = end_time_kan - start_time_kan

        # Predict
        with torch.no_grad():
            y_pred = model_kan(X_test)
        y_pred = np.argmax(y_pred.cpu(), axis=1)
    
        # Calculate performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        
        # Delete model
        del model_kan
        gc.collect()

        return [execution_time, accuracy, f1, precision, recall]
    
    except Exception as e:
        print(f"Error in run_kan: {e}")
        return None

def preprocess(df_train, df_test):
    '''
    Arguments:
    ----------
    df_train : pandas.DataFrame
        The training dataset, where the first column contains the labels and the rest are features.
    df_test : pandas.DataFrame
        The test dataset, where the first column contains the labels and the rest are features.

    Returns:
    --------
    tuple or None
        - If successful, returns a tuple with:
            1. dataset : dict
                A dictionary containing:
                - 'train_input': torch.Tensor of normalized training input data (features).
                - 'train_label': torch.Tensor of training labels.
                - 'test_input': torch.Tensor of normalized validation input data (features).
                - 'test_label': torch.Tensor of validation labels.
            2. X_test_tensor : torch.Tensor
                The normalized input data (features) from the test dataset.
            3. y_test : pandas.Series
                The labels from the test dataset (after any necessary preprocessing).
        - Returns `None` if the input data is invalid or the datasets are empty.
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

    # Make dictionary for training
    dataset = {
        'train_input': torch.tensor(X_train, dtype=torch.float32).to(device),
        'train_label': torch.tensor(y_train.values, dtype=torch.long).to(device),
        'test_input': torch.tensor(X_val, dtype=torch.float32).to(device),
        'test_label': torch.tensor(y_val.values, dtype=torch.long).to(device)
    }

    # Tensor for testing
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    return dataset, X_test_tensor, y_test
        
def multirun(params, lr, epochs, batch, seeds):
    '''
    Arguments:
    ----------
    params : list
        List of instances, each containing the architecture and grid size configuration.
    lr : float
        Learning rate for the KAN model.
    epochs : int
        Number of epochs for training.
    batch : int
        Batch size for training.
    seeds : list
        List of random seeds for reproducibility.

    Returns:
    --------
    None
        Saves results after each run to a pickle file at the specified path.
    '''
    # File with UCR Metadata
    df_meta = pd.read_csv("../DataSummary.csv")

    # Run each architecture configuration
    for i, instance in enumerate(params):
        architecture = instance.architecture
        architecture_str = ','.join(map(str, architecture))

        grid = instance.grid

        file_path = f'../results/kan/LearningRate{str(lr).replace(".", ",")}/results_{architecture_str}_{grid}.pkl'
        
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
            
            df_train = pd.read_csv('../UCRArchive_2018/' + file_train, sep='\t')
            df_test = pd.read_csv('../UCRArchive_2018/' + file_test, sep='\t')

            # Preprocess dataframes inputs and labels
            data = preprocess(df_train, df_test)
            if data is None:
                continue

            dataset, X_test, y_test = data
            
            # Run KAN model for each seed
            for seed in seeds:
                res=run_kan(dataset, epochs=epochs, batch_size=batch, grid_size=g, 
                            architecture=architecture, X_test=X_test, y_test=y_test, 
                            seed=seed, learning_rate=lr)
                
                results.setdefault(name, []).append(res)
            
                # Update results in file each time
                with open(file_path, 'wb') as f:
                    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    
def main():
    Params = namedtuple('Params', ['architecture', 'grid'])

    # Architecture configurations
    kan_params = [
        Params(architecture=[5, 5], grid=5),
    ]

    # Define parameters
    learning_rates=[0.0001, 0.001, 0.01, 0.1, 1]
    seeds=[0,1,2,5,42]
    epochs=500
    batch=16

    for lr in learning_rates:
       multirun(kan_params, lr, epochs, batch, seeds)


if __name__ == "__main__":
    main()