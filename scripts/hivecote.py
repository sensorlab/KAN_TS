#pip install -U aeon
#pip install --upgrade pandas dask
#export CUBLAS_WORKSPACE_CONFIG=:4096:8
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
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
import os
import torch.nn.functional as F
import pandas as pd
import time
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import backend as K
from sklearn.neural_network import MLPClassifier
import gc
from aeon.classification.hybrid import HIVECOTEV2
from aeon.datasets import load_from_tsfile_to_dataframe as load_ts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def process_labels(y_train, y_test):
    # Preprocess labels
    unique=y_test.unique()
        
    # there are [-1,1] classes
    if -1 in unique:
        y_train = y_train.replace({-1: 0})
        y_test = y_test.replace({-1: 0})
   
    # for labels starting from 1
    elif 0 not in unique:
        #class labels are starting from 1
        if 1 in unique:
            y_train=y_train-1
            y_test=y_test-1
        
        #class labels are starting from some other number
        else:
            min_label=min(unique)
            y_train=y_train-min_label
            y_test=y_test-min_label
            
    return y_train, y_test

def main():
    df_meta=pd.read_csv("shared/Irina_KAN/DataSummary.csv")
    
    with open('shared/Irina_KAN/results/hive_cote2/results_hc2.pkl', 'rb') as f:
        results_hc2 = pickle.load(f)
        
    keys_to_remove = results_hc2.keys()

    # Remove rows where 'Name' is in keys_to_remove
    df_meta = df_meta[~df_meta['Name'].isin(keys_to_remove)]
    df_meta = df_meta[~df_meta['Name'].isin(['ElectricDevices', 'FordA','FordB', 'HandOutlines', 'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2'])]
    
    for index, dataset in df_meta.iterrows():
        
        name=dataset['Name']
        print(f'Processing {name}...')
        file_train = f'{name}/{name}_TRAIN.tsv'
        file_test = f'{name}/{name}_TEST.tsv'

        input_layer=dataset['Length']
        output_layer=dataset['Class']

        df_train = pd.read_csv('shared/Irina_KAN/UCRArchive_2018/'+file_train, sep='\t')
        df_test = pd.read_csv('shared/Irina_KAN/UCRArchive_2018/'+file_test, sep='\t')

        y_train = df_train.iloc[:, 0]  # First column for class labels
        X_train = df_train.iloc[:, 1:]  # All other columns for features
        y_test = df_test.iloc[:, 0]  # First column for class labels
        X_test = df_test.iloc[:, 1:]  # All other columns for features

        if y_test.empty:
            continue

        y_train, y_test=process_labels(y_train, y_test)

        # Fit HC2
        hc2 = HIVECOTEV2(random_state=42, n_jobs=-1)
        start_time_hc2=time.time()
        hc2.fit(X_train, y_train)
        end_time_hc2=time.time()

        execution_time = end_time_hc2 - start_time_hc2


        # Predict and print accuracy
        predictions = hc2.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='macro',zero_division=1)
        recall = recall_score(y_test, predictions, average='macro')
        f1 = f1_score(y_test, predictions, average='macro')

        results_hc2[name]=[execution_time, accuracy, f1, precision, recall]


        with open('shared/Irina_KAN/results/hive_cote2/results_hc2.pkl', 'wb') as f:
            pickle.dump(results_hc2, f,protocol=pickle.HIGHEST_PROTOCOL)

        print(index, name)
        del hc2

        gc.collect()
    
    
if __name__ == "__main__":
    main()