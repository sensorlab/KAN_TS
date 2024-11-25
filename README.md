# KAN_TS
Repository for KAN evaluation for TS data

## Project Background

In domains where understanding the decision-making process is as crucial as achieving high accuracy, interpretable models are essential for time series classification tasks. This project explores Kolmogorov-Arnold Networks (KANs), which have shown promise as interpretable alternatives to traditional deep learning models like Multi-Layer Perceptrons (MLPs).

Our research investigates the transferability of KANs from regression tasks to time series classification, with a focus on evaluating their broader generalization capabilities. We evaluate KANs and their efficient variant, Efficient KAN, on the UCR dataset, demonstrating improvements in both accuracy and computational efficiency. 

Results indicate that Efficient KAN performs competitively against MLPs, while also providing faster training times and maintaining a smaller architecture compared to state-of-the-art models like HIVE-COTE2. Additionally, our analysis highlights the sensitivity of KANs to variations in architectural configurations, offering insights into how these factors influence model performance.

For more detailed findings, please refer to our [research paper](https://arxiv.org/abs/2411.14904).

## Installation

First, ensure you have Python 3.x installed. 

Requirements

```
torch>=1.10.0
numpy>=1.21.0
pandas>=1.3.0
tqdm>=4.60.0
scikit-learn>=0.24.0
aeon>=0.2.0
```

To install all required dependencies, use the following command:

```bash
pip install -r requirements.txt
```
### Set Up UCR Dataset

Ensure you have the UCR dataset folder available. This project assumes the UCR folder structure, with each dataset in its own directory. Place the UCR dataset folder in the root directory of this project.

[Download the UCR dataset here](http://www.cs.ucr.edu/~eamonn/time_series_data)

### Running Model Scripts
Each script in the scripts folder is designed to run a specific model on the UCR datasets. Results are automatically saved in the results directory, organized by model type and learning rate (where applicable).

## Project Structure

```plaintext
├── scripts
│   ├── effkan.py         # Script to run Efficient KAN on UCR datasets
│   ├── hivecote.py       # Script to run Hive-Cote 2.0 on UCR datasets
│   ├── mlp.py            # Script to run MLP Classifier on UCR datasets
│   └── pykan.py          # Script to run KAN on UCR datasets
├── results               # Folder where model results are saved
│   ├── effkan
│   │   ├── LearningRate0,0001
│   │   │   ├── results_40,40_5.pkl
│   │   │   └── results_40,40_10.pkl
│   │   ├── LearningRate0,001
│   │   ├── LearningRate0,01
│   │   ├── LearningRate0,1
│   │   └── LearningRate1
│   ├── kan
│   │   └── LearningRate0,01
│   ├── mlp 
│   │   └── LearningRate0,01
│   └── HiveCote2
│       └── results_hc2.pkl
├── DataSummary.csv        # Metadata file for navigating dataset specifications
├── requirements.txt       # List of dependencies
└── README.md              # Documentation
```
## Results Structure

The `results` directory is organized by model type, with subfolders for different learning rates where applicable. For `kan`, `mlp`, and `effkan` models, there are five learning rate folders each:

- `LearningRate0,0001`
- `LearningRate0,001`
- `LearningRate0,01`
- `LearningRate0,1`
- `LearningRate1`

Each learning rate folder (e.g., `LearningRate0,0001`) contains `.pkl` files named according to specific configurations, such as `results_40,40_5.pkl`, where `40,40` refers to architecture specifications, and `5` refers to the grid size.

The `HiveCote2` model does not include these folders and instead has a single results file (`results_hc2.pkl`) stored directly in its folder.

### Results File Contents

Each `.pkl` file contains a dictionary where:

- **Key**: The dataset name.
- **Value**: A nested list containing the results for each seed.

If five seeds are used, the value will be a list containing five lists, with each inner list corresponding to the results from one seed. Each inner list includes the following metrics, in order:

1. **Measured time for training**
2. **Accuracy**
3. **F1 Score**
4. **Precision**
5. **Recall**

## Citation

```
@misc{barasin2024exploringkan,
      title={Exploring Kolmogorov-Arnold Networks for Interpretable Classification}, 
      author={Irina Barašin and Blaž Bertalanič and Miha Mohorčič and Carolina Fortuna},
      year={2024},
      eprint={2411.14904},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
