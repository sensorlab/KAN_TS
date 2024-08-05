UCR archive 2018 - 144 datasets- 120 worked, others need to be ran separately due to missing values etc...
---------------------------------------------------------------------------------------------------------------
results_mlp_depth_3: key= data name, value=[execution_time, accuracy, f1, precision, recall]

parameters for skit learn mlp classifier [input, 300, 300, 300, output]: 
sklearn_mlp = MLPClassifier(
    hidden_layer_sizes=(300,300,300,), 
    activation='relu',          # Activation function is Rectified Linear Unit (ReLU)
    solver='adam',              # Optimization solver is Adam
    alpha=1.0,                  # L2 penalty (regularization term) 
    batch_size=int(batch),      # Size of minibatches for stochastic optimizers
    learning_rate='constant',   # Learning rate 
    learning_rate_init=0.001,   # Initial learning rate
    max_iter=int(epochs),       # Maximum number of iterations
    random_state=42             # Random seed for reproducibility
)
epochs:500
batch:16

results_mlp_depth_4: same but for [input,300,300,300,300,output]

---------------------------------------------------------------------------------------------------------------
results_kan_depth_3: key= data name, value=[execution_time, accuracy, f1, precision, recall]

parameters of pykan: 
[input, 40, 40, output]
grid=5
k=3
opt=adam
lr=0.001
epochs=500
batch=16
lamb_l1=0.1

results_kan_depth_4: [input, 40, 40, 40, output]

--------------------------------------------------------------------------------------------------------------

    