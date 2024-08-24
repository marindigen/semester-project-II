import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import numpy as np
import tqdm
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns

##################################################### Data Loader ###################################################
class NumpyDataset(Dataset):
    def __init__(self, data, flag_float=True):
        """
        Initialize the dataset with the data.
        
        Parameters:
        data (numpy.ndarray): The data matrix.
        """
        if flag_float:
            self.data = torch.from_numpy(data).float()
        else:
            self.data = torch.from_numpy(data).long()
    
    def __len__(self):
        """
        Returns the total number of samples.
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Generates one sample of data.
        
        Parameters:
        idx (int): Index of the sample to retrieve.
        
        Returns:
        torch.Tensor: The data sample at index idx.
        """
        return self.data[idx]

def create_dataloaders(train_matrix, batch_size, flag_float=False):
    """
    Creates dataloaders for the training and testing matrices.
    
    Parameters:
    train_matrix (numpy.ndarray): Training data matrix of shape (1000, 200).
    test_matrix (numpy.ndarray): Testing data matrix of shape (1000, 200).
    batch_size (int): The batch size for the dataloaders.
    
    Returns:
    tuple: (train_loader, test_loader) The training and testing dataloaders.
    """
    # Create dataset objects
    train_dataset = NumpyDataset(train_matrix, flag_float)
    #test_dataset = NumpyDataset(test_matrix, flag_float)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader

##################################################### Data Generation ###################################################

def sample_data(num_samples, distributions, len_of_seq):
    # Define the parameters for different distributions
    
    # Initialize the tensor to store the samples
    all_samples = torch.zeros(len(distributions), num_samples, len_of_seq)
    
    # Sample data from each distribution
    for i in range(len(distributions)):
        distr = distributions[i]
        if distr["type"] == "normal":
            samples = np.random.normal(distr["mean"], distr["std"], (num_samples, len_of_seq))
            samples = np.where(samples >= 0, 1, -1)
        elif distr["type"] == "uniform":
            samples = np.random.uniform(distr["low"], distr["high"], (num_samples, len_of_seq))
            samples = np.where(samples >= 0, 1, -1)
        elif distr["type"] == "exponential":
            samples = np.random.exponential(distr["scale"], (num_samples, len_of_seq))
            samples = np.where(samples >= distr["scale"], 1, -1)
        elif distr["type"] == "gamma":
            samples = np.random.poisson(distr["scale"], (num_samples, len_of_seq))
            samples = np.where(samples >= np.mean(samples), 1, -1)
        elif distr["type"] == "poisson":
            samples = np.random.poisson(distr["lam"], (num_samples, len_of_seq))
            samples = np.where(samples >= np.mean(samples), 1, -1)
        else:
            raise ValueError("Unsupported distribution type")
        
        # Store the samples in the tensor
        all_samples[i] = torch.tensor(samples, dtype=torch.float32)
    
    return all_samples.reshape(num_samples, len(distributions), len_of_seq)

def one_hot_encoding_batch(seq, vocab):
    seq = torch.tensor(seq, dtype=torch.long)
    batch_size, seq_length = seq.shape
    vocab_size = len(vocab)
    
    one_hot = torch.zeros((batch_size, seq_length, vocab_size), dtype=torch.int)
    
    for i in range(batch_size):
        for j in range(seq_length):
            token = seq[i, j].item()
            if token in vocab:
                one_hot[i, j, vocab[token]] = 1
            else:
                raise KeyError(f"Token {token} not found in vocabulary.")
    
    return one_hot

def one_hot_encoding(seq, vocab):
    if isinstance(seq, int):
        one_hot = torch.zeros(1, len(vocab), dtype=int)
        one_hot[:, vocab[seq]] = 1
    else:
        one_hot = torch.zeros((len(seq), len(vocab)), dtype=int)
        for i, spin in enumerate(seq):
            one_hot[i, vocab[spin]] = 1
    return one_hot

##################################################### Plotting ###################################################

def generate_heatmap(weight_matrix):
    """
    Generates a heatmap from a given weight matrix.
    
    Parameters:
    weight_matrix (numpy.ndarray): A 2D numpy array of shape (200, 200) representing the weights.
    
    Returns:
    None
    """
    if weight_matrix.shape != (200, 200):
        raise ValueError("The weight matrix must be of shape (200, 200)")
    
    plt.figure(figsize=(10, 8))
    plt.imshow(weight_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Weight Value')
    plt.title('Weight Matrix Heatmap')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.show()

def plotting_heatmaps(initial_sequences, final_chains_train, title1, title2):
    # Set up a figure with two subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Create a heatmap on each subplot
    sns.heatmap(initial_sequences, ax=ax1, cmap='viridis')
    ax1.set_title(title1)

    sns.heatmap(final_chains_train, ax=ax2, cmap='viridis')
    ax2.set_title(title2)

    # Adjust layout
    plt.tight_layout()

    # Display the plot
    plt.show()

################################################### Evaluation metrics ############################################

'''
def calculate_true_positives(cov_inv, J):
    """
    Calculate the number of true positives.
    Args:
        cov_inv (np.array): Inverse covariance matrix.
        J (np.array): True interaction matrix.
    Returns:
        int: Number of true positives.
    """
    TP = 0
    num_positives = np.sum(J==1)
    for c, j in zip(cov_inv, J):
        if (c == j) and (c == 1):
            TP += 1
    return TP, TP/num_positives
'''

def calculate_true_positives(cov_inv, J):
    """
    Calculate the number of true positives.
    Args:
        cov_inv (np.array): Inverse covariance matrix.
        J (np.array): True interaction matrix.
    Returns:
        int: Number of true positives.
    """
    TP = 0
    FP = 0
    for c, j in zip(cov_inv, J):
        if (c == j) and (c == 1):
            TP += 1
        elif (c != j) and (c == 1):
            FP += 1
    if TP+FP==0:
        return TP, 0
    return TP, TP/(TP+FP)

def calculate_false_positives(cov_inv, J):
    """
    Calculate the number of false positives.
    Args:
        cov_inv (np.array): Inverse covariance matrix.
        J (np.array): True interaction matrix.
    Returns:
        int: Number of false positives.
    """
    FP = 0
    num_negatives = np.sum(J==0)
    for c, j in zip(cov_inv, J):
        if (c != j) and (c == 1):
            FP += 1
    return FP, FP/num_negatives

