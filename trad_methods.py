import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
import tqdm
import pickle
import os
import h5py
from sklearn.metrics import mutual_info_score
from numba.typed import List
from numba import njit,prange,jit
from utils import *

# Calculate the covariance matrix and its inverse
def calc_cov_inv(final_chains_train):
    # Calculate the covariance matrix
    cov_matrix = np.cov(final_chains_train.T)
    # Calculate the inverse of the covariance matrix
    cov_inv = np.linalg.inv(cov_matrix+1e-5*np.eye(cov_matrix.shape[0]))
    return cov_matrix, cov_inv

# Calculate the mutual information matrix
def calc_mutual_info(final_chains_train):
    mi_matrix = np.zeros((final_chains_train.shape[1], final_chains_train.shape[1]))
    for i in range(final_chains_train.shape[1]):
        for j in range(final_chains_train.shape[1]):
            if i != j:
                mi_matrix[i, j] = mutual_info_score(final_chains_train[:, i], final_chains_train[:, j])
            else:
                mi_matrix[i, j] = 0  # Mutual information of a variable with itself can be ignored or set to 0
    return mi_matrix

# Altenrative method for calculating mean-field DCA
def compute_frequencies(msa, q, lambda_):
    """
    Compute single-site and pair-site frequencies with pseudocount regularization.

    Args:
    msa (np.ndarray): An L x N array of MSA, where L is the length of the sequence and N is the number of sequences.
    q (int): Number of possible states (e.g., 2 for simulated proteins).
    lambda_ (float): Pseudocount parameter.

    Returns:
    tuple: Tuple containing single-site frequencies and pair-site frequencies.
    """
    msa = (msa + 1)/2
    L, N = msa.shape
    # Single-site frequencies
    f_i = np.zeros((L, q))
    for i in range(L):
        f_i[i, :], __ = np.histogram(msa[i], bins=np.arange(q + 1) - 0.5, density=False)
    f_i /= N
    #print(f_i)
    # f_i = calc_frequency(msa.T, lambda_).T
    #frequencies, fri, frj = calc_frequency(msa.T, lambda_)

    # Compute <sigma_i> and <sigma_i sigma_j>
    sigma_i = np.sum(f_i * np.arange(q), axis=1)
    sigma_ij = np.outer(sigma_i, sigma_i)

    # Pair-site frequencies
    f_ij = np.zeros((L, L, q, q))
    for i in range(L):
        for j in range(L):
            if i != j:
                for alpha in range(N):
                    f_ij[i, j,int(msa[i, alpha]), int(msa[j, alpha])] += 1
            f_ij[i, j] += np.outer(f_i[i], f_i[j])
            f_ij[i, j] /= (N + lambda_)

    # Covariance matrix adjusted with pseudocounts
    C_ij = f_ij # - sigma_ij[:, :, None, None]
    for i in range(L):
        C_ij[i, i] = (1 - lambda_)**2 * sigma_i[i] + lambda_ * (2 - lambda_)

    return f_i, C_ij

def infer_couplings(f_i, C_ij, q):
    """
    Infer coupling matrix J from the adjusted covariance matrix C_ij.

    Args:
    f_i (np.ndarray): Single-site frequencies.
    C_ij (np.ndarray): Adjusted covariance matrix.

    Returns:
    np.ndarray: Coupling matrix J.
    """
    L = f_i.shape[0]
    J = np.zeros((L, L, q, q))

    for i in range(L):
        for j in range(i + 1, L):
            # Regularization to prevent singular matrix
            reg = np.eye(q) * 0.0001
            J[i, j] = -np.linalg.inv(C_ij[i, j]+reg)
            J[j, i] = J[i, j].T
    return J

def compute_frobenius_norm(J):
    """
    Compute the Frobenius norm of the coupling matrix for each pair of residues.

    Args:
    J (np.ndarray): Coupling matrix.

    Returns:
    np.ndarray: Frobenius norms.
    """
    L = J.shape[0]
    norms = np.zeros((L, L))
    for i in range(L):
        for j in range(i + 1, L):
            norms[i, j] = np.linalg.norm(J[i, j])
            norms[j, i] = norms[i, j]

    return norms

