import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
#sys.path.append(os.path.abspath('shared/Irina_KAN'))
from kan import KAN, create_dataset
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from keras.utils import to_categorical
from keras.metrics import Precision, Recall
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import pandas as pd
import time
import gc
from collections import namedtuple

torch.use_deterministic_algorithms(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)

def run_kan(name, dataset, epochs, batch_size, grid_size, architecture, X_test, y_test, path, results_kan):
    print(dataset['train_label'].device)
    model_kan = KAN(width=architecture, grid=grid_size, k=3, symbolic_enabled=False, seed=42).to(device)

    # IMPORTANT TO USE ARGMAX
    def train_acc():
        return torch.mean((torch.argmax(model_kan(dataset['train_input']), dim=1) == dataset['train_label']).float())

    def test_acc():
        return torch.mean((torch.argmax(model_kan(dataset['test_input']), dim=1) == dataset['test_label']).float())
   
    length = dataset['test_input'].shape[0]
    
    if length < batch_size:
        batch_size = length
    

    start_time_kan=time.time()
    results = model_kan.fit(dataset, opt="Adam", steps=int(epochs), metrics=(train_acc, test_acc), batch=batch_size, lr=0.001, loss_fn=torch.nn.CrossEntropyLoss(), lamb_l1=0.1);
    end_time_kan = time.time()
    
    torch.save(model_kan.state_dict(), f'{path}/{name}_model.pth')
    
    execution_time = end_time_kan - start_time_kan
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    # Perform prediction
    with torch.no_grad():
        y_pred = model_kan(X_test_tensor)
    y_pred=np.argmax(y_pred.cpu(), axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro',zero_division=1)
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    results_kan[name]=[execution_time, accuracy, f1, precision, recall]
    
    with open(f'{path}/results_kan.pkl', 'wb') as f:
        pickle.dump(results_kan, f,protocol=pickle.HIGHEST_PROTOCOL)
    del model_kan

    gc.collect()
    
    

def main():
    Params = namedtuple('Params', ['architecture', 'grid'])
    params = [
        Params(architecture=[30, 30], grid=5),
        Params(architecture=[30, 30], grid=10),
        Params(architecture=[40, 40], grid=10)
    ]
    #iterate through all architectures
    for i, instance in enumerate(params):
        path='results'
                
        file_path = os.path.join('results_kan_gpu.pkl')

        # Check if the file exists
        if os.path.exists(file_path):
            # Load the file if it exists
            with open(file_path, 'rb') as file:
                results_kan = pickle.load(file)
        else:
            # Create an empty dictionary if the file does not exist
            results_kan = {}
        
        keys_to_remove = results_kan.keys()
        df_meta=pd.read_csv("shared/Irina_KAN/DataSummary.csv")
        # Remove rows where 'Name' is in keys_to_remove
        df_meta = df_meta[~df_meta['Name'].isin(keys_to_remove)] 
        
        print('\n',instance,'=======================================================================================================================\n')

        for index, dataset in df_meta.iterrows():

            name=dataset['Name']
            file_train = f'{name}/{name}_TRAIN.tsv'
            file_test = f'{name}/{name}_TEST.tsv'

            input_layer=dataset['Length']
            output_layer=dataset['Class']

            if input_layer=='Vary':
                continue

            # FORMAT DATA FOR KAN 
            df_train = pd.read_csv('shared/Irina_KAN/UCRArchive_2018/'+file_train, sep='\t')
            df_test = pd.read_csv('shared/Irina_KAN/UCRArchive_2018/'+file_test, sep='\t')

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


            # For KAN
            dataset = create_dataset(torch.tensor(X_train, dtype=torch.float32), 
                                     torch.tensor(y_train.values, dtype=torch.long), 
                                     torch.tensor(X_val, dtype=torch.float32), 
                                     torch.tensor(y_val.values, dtype=torch.long), 
                                     device=device)
            """
            dataset = {
                'train_input': torch.tensor(X_train, dtype=torch.float32).to(device),
                'train_label': torch.tensor(y_train.values, dtype=torch.long).to(device),
                'test_input': torch.tensor(X_val, dtype=torch.float32).to(device),
                'test_label': torch.tensor(y_val.values, dtype=torch.long).to(device)
            }
            """
            #print(dataset['train_label'].size(),dataset['train_input'].size())
            #print(dataset['test_label'])
            epochs=500
            batch_size=16

            architecture=[int(input_layer)]+instance.architecture+[int(output_layer)]
            g=instance.grid

            run_kan(name, dataset, epochs, batch_size, g, architecture, X_test, y_test, path, results_kan)

            print(index, name)

if __name__ == "__main__":
    main()