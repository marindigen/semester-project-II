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
from utils import *

############################################### Transformer Model ################################################
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        
        self.W_q = nn.Linear(embed_dim, embed_dim) # Linear layer for query 
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim) # Output layer to ensure that dimensionality matches the model's expected dimensionality
        
        self.combine_heads = nn.Linear(num_heads, 1)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, embed_dim = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
     
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        #print("x somewhere:", x.shape)
        #print("seqeunce length:",seq_length)
        #print("d model dim:",self.d_model)
        return x.transpose(1, 2).contiguous().view(batch_size,seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        #print("shape of Q before splitting heads:", Q.shape)
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        #print("shape of Q after splitting heads:", Q.shape)
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        #print(" before W_o:", self.combine_heads(attn_output.permute(1,2,0)).squeeze(-1).shape)
        output = self.W_o(self.combine_heads(attn_output))
        return output
    
class VanillaAttention(nn.Module):
    def __init__(self, embed_dim, a, max_seq_length, num_spins=3, dropout_rate=0.0):
        super(VanillaAttention, self).__init__()
        self.word_embeddings = nn.Linear(num_spins, embed_dim)
        self.position_embeddings = nn.Embedding(max_seq_length, embed_dim)
        self.a = a  # parameter controlling how important positions are
        self.value_weight = nn.Linear(embed_dim, embed_dim)
        self.query_weight = nn.Linear(embed_dim, embed_dim)
        self.key_weight = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)  # output layer
        self.dropout = nn.Dropout(dropout_rate)
        self.scale = math.sqrt(embed_dim)

    def forward(self, s, enc_output):

        #position_ids = torch.arange(s.size(0), dtype=torch.long)
        #x = self.word_embeddings(s) + self.a*self.position_embeddings(position_ids)
        #x = s + self.a*self.position_embeddings(position_ids)
        batch_size_decoder = s.shape[0]
        batch_size_encoder = enc_output.shape[0]
        if batch_size_decoder != batch_size_encoder:
            # Example approach: expand enc_output to match s's batch size
            # This simple repeat might not be the best approach depending on your specific needs
            s = s.repeat(batch_size_encoder // batch_size_decoder, 1, 1)

        query = self.query_weight(s)
        key = self.key_weight(enc_output)
        values = self.value_weight(enc_output)
        # Simple attention score calculation (Dot product): this is equivalent to the interaction matrix
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale # Transpose last two dimensions for matrix multiplication
        attn_weights = torch.softmax(scores, dim=-1)  # Apply softmax to scores to get probabilities

        # Apply attention scores to values
        output= torch.matmul(attn_weights, values)
        if batch_size_decoder != batch_size_encoder:
            output = output.view(batch_size_decoder, batch_size_encoder // batch_size_decoder, output.size(-2), output.size(-1)).mean(dim=1)

        # Sum over the sequence length dimensions
        #attn_output = attn_output.sum(dim=1)
        output = self.fc(self.dropout(output)) # should have size (20,3)

        return output, attn_weights
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        #return x + self.pe[:, :x.size(1)]
        return x + self.pe[:, :, :x.size(2)]
    
class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, proj_layer_dim, dropout, num_distr=5):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_distr)
        self.feed_forward = PositionWiseFeedForward(d_model=embed_dim, d_ff=proj_layer_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_distr,1)
        
    def forward(self, x):
        #x = self.fc(x.permute(1,2,0)).permute(2,0,1)
        attn_output = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, a, max_seq_length, num_spins, proj_layer_dim, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = VanillaAttention(embed_dim=embed_dim, a=a, max_seq_length=max_seq_length, num_spins=num_spins, dropout_rate=dropout)# masking one of the word in the sequence
        self.cross_attn = VanillaAttention(embed_dim=embed_dim, a=a, max_seq_length=max_seq_length, num_spins=num_spins, dropout_rate=dropout)
        self.feed_forward = PositionWiseFeedForward(d_model=embed_dim, d_ff=proj_layer_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.cross_attn_weights = None
        self.decoder_attn_weights = None
        
    def forward(self, x, enc_output):
        enc_output = enc_output.squeeze(0)
        attn_output, decoder_attn_weights = self.self_attn(x, x)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output, cross_attn_weights = self.cross_attn(x, enc_output)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        #x = x.sum(dim=0)
        self.cross_attn_weights = cross_attn_weights 
        self.decoder_attn_weights = decoder_attn_weights
        return x
    