
#export CUBLAS_WORKSPACE_CONFIG=:4096:8

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from kan import *
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.under_sampling import RandomUnderSampler
from keras.utils import to_categorical
from keras.metrics import Precision, Recall
from sklearn.preprocessing import StandardScaler
import os
import torch.nn.functional as F
from efficient_kan import KAN
import time
import gc
from collections import namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def run_efficient_kan(name, input_layer, output_layer, trainloader, valloader, testloader, epochs, architecture, grid_size, file_path, results_effkan):
    
    model=KAN(architecture,grid_size=grid_size,spline_order=3)
    model.to(device)
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Define learning rate scheduler
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0)

    # Define loss
    criterion = nn.CrossEntropyLoss()
    
    start_time_effkan=time.time()
    for epoch in range(epochs):
        # Train
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        with tqdm(trainloader, desc=f"Training Epoch {epoch+1}", disable=True) as pbar:
            for data, labels in pbar:
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                running_accuracy += (outputs.argmax(dim=1) == labels).float().mean().item()
                pbar.set_postfix(loss=running_loss/len(pbar), accuracy=running_accuracy/len(pbar), lr=optimizer.param_groups[0]['lr'])

        # Validation
        model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for data, labels in tqdm(valloader, desc=f"Validation Epoch {epoch+1}", disable=True):
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                val_loss += criterion(outputs, labels).item()
                val_accuracy += (outputs.argmax(dim=1) == labels).float().mean().item()
        val_loss /= len(valloader)
        val_accuracy /= len(valloader)

        # Update learning rate
        #scheduler.step()

        #print(f"Epoch {epoch + 1}, Train Loss: {running_loss/len(trainloader):.4f}, Train Accuracy: {running_accuracy/len(trainloader):.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    end_time_effkan=time.time()
    
    execution_time = end_time_effkan - start_time_effkan
    model.eval()

    # Make predictions
    all_preds = [] #predicted class
    all_labels = [] #true class

    with torch.no_grad():
        for data, labels in testloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Concatenate all predictions and labels
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro',zero_division=1)
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    results_effkan[name]=[execution_time, accuracy, f1, precision, recall]
    
    with open(f'{file_path}', 'wb') as f:
        pickle.dump(results_effkan, f,protocol=pickle.HIGHEST_PROTOCOL)
        
    del model

    gc.collect()
    
    

def main():
    
    Params = namedtuple('Params', ['architecture', 'grid'])
    params = [
        #Params(architecture=[40, 40], grid=5),
        Params(architecture=[40, 40, 40], grid=5)
    ]
    
                 
    #iterate through all architectures
    for i, instance in enumerate(params):
        
        #path=f'optimized_effkan_models/depth_4/{i}'
                
        #file_path = os.path.join(path, 'results_effkan.pkl')
        file_path ='results_effkan_depth_4.pkl'
        # Check if the file exists
        if os.path.exists(file_path):
            # Load the file if it exists
            with open(file_path, 'rb') as file:
                results_effkan = pickle.load(file)
        else:
            # Create an empty dictionary if the file does not exist
            results_effkan = {}
        
        #keys_to_remove = results_effkan.keys()
        df_meta=pd.read_csv("DataSummary.csv")
        # Remove rows where 'Name' is in keys_to_remove
        #df_meta = df_meta[~df_meta['Name'].isin(keys_to_remove)] 
        df_meta = df_meta[df_meta['Name'].isin(['GestureMidAirD1','GestureMidAirD2','GestureMidAirD3'])]
        print('\n',instance,'=======================================================================================================================\n')

        #run all datasets for that architecture
        for index, dataset in df_meta.iterrows():
            name=dataset['Name']
            file_train = f'{name}/{name}_TRAIN.tsv'
            file_test = f'{name}/{name}_TEST.tsv'

            input_layer=dataset['Length']
            output_layer=dataset['Class']

            # FORMAT DATA FOR KAN 
            df_train = pd.read_csv('UCRArchive_2018/'+file_train, sep='\t')
            df_test = pd.read_csv('UCRArchive_2018/'+file_test, sep='\t')

            df_train=df_train.dropna()
            df_test=df_test.dropna()

            try:
                df_train, df_val= train_test_split(df_train, test_size=0.2, random_state=42)
            except ValueError as e:
                continue

            y_train = df_train.iloc[:, 0]  # First column for class labels
            X_train = df_train.iloc[:, 1:]  # All other columns for features
            y_test = df_test.iloc[:, 0]  # First column for class labels
            X_test = df_test.iloc[:, 1:]  # All other columns for features
            X_val = df_val.iloc[:, 1:]  # All other columns for features
            y_val = df_val.iloc[:, 0]  # First column for class labels

            if input_layer=='Vary':
                input_layer= X_train.shape[1]

            # Preprocess labels
            unique=y_test.unique()

            if y_test.empty:
                continue

            # there are [-1,1] classes
            if -1 in unique:
                y_train = y_train.replace({-1: 0})
                y_val = y_val.replace({-1: 0})
                y_test = y_test.replace({-1: 0})

            # for labels starting from 1
            elif 0 not in unique:
                #class labels are starting from 1
                if 1 in unique:
                    y_train=y_train-1
                    y_test=y_test-1
                    y_val=y_val-1

                #class labels are starting from some other number
                else:
                    min_label=min(unique)
                    y_train=y_train-min_label
                    y_test=y_test-min_label
                    y_val=y_val-min_label

            #Normalize features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
            X_val = scaler.fit_transform(X_val)

            # Convert data to torch tensors
            X_train = torch.tensor(X_train, dtype=torch.float32)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            X_val = torch.tensor(X_val, dtype=torch.float32)
            y_train = torch.tensor(y_train.values, dtype=torch.long)
            y_test = torch.tensor(y_test.values, dtype=torch.long)
            y_val = torch.tensor(y_val.values, dtype=torch.long)

            # Create DataLoader
            trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=16, shuffle=False)
            valloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_val, y_val), batch_size=16, shuffle=False)
            testloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, y_test), batch_size=16, shuffle=False)

            architecture=[int(input_layer)]+instance.architecture+[int(output_layer)]
            g=instance.grid



            run_efficient_kan(name, input_layer, output_layer, trainloader, valloader, testloader, 500, architecture, g, file_path, results_effkan)

            print(index, name)
    
if __name__ == "__main__":
    main()