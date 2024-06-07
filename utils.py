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

def create_dataloaders(train_matrix, test_matrix, batch_size, flag_float=False):
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
    test_dataset = NumpyDataset(test_matrix, flag_float)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

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

def mask_random_spins_batch(sequence_batch, vocab, pos=1, mask_token=2, one_hot_flag=False):
    """
    Mask random spins in a batch of sequences of protein spins.

    Parameters:
    - sequence_batch (torch.Tensor): A batch of sequences with shape (32, 200).
    - vocab (dict): A dictionary mapping each element in the sequence to an index.
    - pos (int): The number of positions to mask in each sequence (default is 25).
    - mask_token (int): The token used to mask a spin (default is 2).
    - one_hot_flag (bool): Flag to indicate if the output should be one-hot encoded (default is False).

    Returns:
    - masked_sequences (torch.Tensor): A batch of sequences with masked spins.
    - masked_positions (torch.Tensor): The positions of the spins that were masked for each sequence.
    """
    assert isinstance(sequence_batch, torch.Tensor), "The sequence_batch must be a torch.Tensor"

    masked_sequences = sequence_batch.clone()  # Avoid modifying the original batch
    masked_positions = []

    for i in range(sequence_batch.size(0)):
        mask_positions = random.sample(range(sequence_batch.shape[1]), pos)
        for mask_position in mask_positions:
            masked_sequences[i, mask_position] = mask_token
        masked_positions.append(mask_positions)

    masked_positions = torch.tensor(masked_positions, dtype=torch.long)

    if one_hot_flag:
        one_hot = one_hot_encoding_batch(masked_sequences, vocab)
        return torch.tensor(one_hot, dtype=torch.float32), masked_positions
    else:
        return torch.tensor(masked_sequences, dtype=torch.long), masked_positions
    
def mask_random_spins(sequence, vocab, pos=5, mask_token=2, one_hot_flag=False):
    """
    Mask one random spin in a sequence of protein spins.
    
    Parameters:
    - sequence: a list or sequence of spins (integers)
    - mask_token: the token used to mask a spin (default is 2)
    
    Returns:
    - masked_sequence: a sequence similar to the input but with one spin masked
    - masked_position: the position of the spin that was masked
    """
    # Ensure the sequence can be converted to a list for masking
    sequence_list = sequence.numpy().tolist() if isinstance(sequence, torch.Tensor) else list(sequence)
    # Choose a random position to mask, excluding the first spin
    mask_positions = random.sample(range(len(sequence)), pos)
    
    # Mask the chosen position
    masked_sequence = sequence_list.copy()
    for mask_position in mask_positions:
        masked_sequence[mask_position] = mask_token

    # Create an array of zeros with shape (len(sequence), len(vocab))
    if one_hot_flag:
        one_hot = one_hot_encoding(masked_sequence, vocab)
        return torch.tensor(one_hot, dtype=torch.float), torch.tensor(mask_positions, dtype=torch.long)
    else:
        return torch.tensor(masked_sequence, dtype=torch.long), torch.tensor(mask_positions, dtype=torch.long)

def extract_target_tokens(input_one_hot, positions):
    """
    Extract target tokens from one-hot encoded batch based on masked positions.

    Parameters:
    - input_one_hot (torch.Tensor): One-hot encoded input batch of shape (32, 200, 3).
    - positions (torch.Tensor): Tensor of masked positions for each sequence of shape (32) or (32, num_positions).

    Returns:
    - target_tokens (torch.Tensor): Tensor of target tokens of shape (32) or (32, num_positions).
    """
    batch_size, seq_length, num_features = input_one_hot.shape

    def get_token(input_one_hot, i, pos):
        token_tensor = torch.where(input_one_hot[i, pos] == 1)[0]
        print("token tensor:", token_tensor)
        if len(token_tensor) > 0:
            return token_tensor.item()
        else:
            return -1
    
    # Check if positions is a single dimension or two dimensions
    if positions.dim() == 1:  # Single masked position per sequence
        target_tokens = [get_token(input_one_hot, i, positions[i]) for i in range(batch_size)]
        target_tokens = torch.tensor(target_tokens, dtype=torch.long)
    elif positions.dim() == 2:  # Multiple masked positions per sequence
        target_tokens = []
        for i in range(batch_size):
            tokens = [get_token(input_one_hot, i, pos) for pos in positions[i]]
            target_tokens.append(tokens)
        target_tokens = torch.tensor(target_tokens, dtype=torch.long)
    else:
        raise ValueError("positions tensor must be either 1D or 2D")

    return target_tokens


