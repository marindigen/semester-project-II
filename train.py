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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils import *

# for tensorboard
from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter()

########################################################################################## Normal Model #################################################################################################

def evaluate(model, data_test, data_test_dec, vocab, criterion, one_hot_flag, device=0):
    model.eval()
    epoch_loss = 0
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    for i, data in tqdm.tqdm(enumerate(data_test), total=len(data_test)):
        # Get the inputs
        input_seq_enc = data
        input_seq_dec = data_test_dec[i]
        if one_hot_flag:
            input_encoder_one_hot = torch.stack([one_hot_encoding(input_seq_enc[i].tolist(), vocab) for i in range(len(input_seq_enc))], dim=0)
            input_encoder_one_hot = torch.tensor(input_encoder_one_hot, dtype=torch.float)

            input_decoder_one_hot = one_hot_encoding(input_seq_dec.tolist(), vocab)
            # mask a token
            masked_sequence_dec, positions = mask_random_spins(input_seq_dec, vocab, mask_token=2, one_hot_flag=True)
        
            # Forward pass
            outputs = model.forward(input_encoder_one_hot, masked_sequence_dec)
            target_tokens = torch.where(input_decoder_one_hot[positions]==1)[1]
        else:
            # mask a token
            masked_sequence_dec, positions = mask_random_spins(input_seq_dec, vocab, mask_token=2)
        
            # Forward pass
            outputs = model.forward(input_seq_enc, masked_sequence_dec).squeeze(0)
            target_tokens = input_seq_dec[positions] #target_token = input_seq[position]
            target_tokens = torch.tensor([vocab[int(token)] for token in target_tokens], dtype=torch.long)

        # Compute loss
        loss = criterion(outputs[positions], target_tokens)
        epoch_loss += loss.item()
    return epoch_loss / len(data_test)

def train(model, data_train, data_train_dec, data_test, data_test_dec, vocab, optimizer, criterion, one_hot_flag, writer, num_epochs=15, device=0):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # Training loop
    model.train()
    best_eval_loss = 1e-3 # used to do early stopping

    for epoch in tqdm.tqdm(range(num_epochs), leave=False, position=0):
        running_loss = 0
        epoch_loss = 0
        
        for i, data in tqdm.tqdm(enumerate(data_train), total=len(data_train)):
            # Get the inputs
            input_seq_enc = data
            input_seq_dec = data_train_dec[i]

            if one_hot_flag:
                input_encoder_one_hot = torch.stack([one_hot_encoding(input_seq_enc[i].tolist(), vocab) for i in range(len(input_seq_enc))], dim=0)
                input_encoder_one_hot = torch.tensor(input_encoder_one_hot, dtype=torch.float)

                input_decoder_one_hot = one_hot_encoding(input_seq_dec.tolist(), vocab)

                # mask tokens in decoder
                masked_sequence_dec, positions = mask_random_spins(input_seq_dec, vocab, mask_token=2, one_hot_flag=True)

                # Forward pass
                predictions = model.forward(input_encoder_one_hot, masked_sequence_dec)
                #predictions = prediction[positions]
                target_tokens = torch.where(input_decoder_one_hot[positions]==1)[1]
            else:
                input_seq_enc = torch.tensor(input_seq_enc, dtype=torch.long)
                # mask tokens in decoder
                masked_sequence_dec, positions = mask_random_spins(input_seq_dec, vocab, mask_token=2, one_hot_flag=False)

                # Forward pass
                predictions = model.forward(input_seq_enc, masked_sequence_dec).squeeze(0) #masked_sequence[masked_position])
                target_tokens = input_seq_dec[positions] #input_seq[masked_position]
                target_tokens = torch.tensor([vocab[int(token)] for token in target_tokens], dtype=torch.long)
            
            # Compute loss
            loss = criterion(predictions[positions], target_tokens)
            epoch_loss += loss.item()
            
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 10 == 9. :    # print every 10 mini-batches
                writer.add_scalar("Running Loss", running_loss / 100, epoch)
                #print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

        print(f'Epoch {epoch + 1} | Train Loss: {(epoch_loss / len(data)):.4f}')
        writer.add_scalar("Train Loss", epoch_loss / len(data), epoch)
        eval_loss = evaluate(model, data_test, data_test_dec, vocab, criterion, one_hot_flag, device=device)
        writer.add_scalar("Eval Loss", eval_loss, epoch)
        print(f'Epoch {epoch + 1} | Eval Loss: {(eval_loss):.4f}')
        
        # Perform early stopping based on eval loss
        if eval_loss < best_eval_loss:
            return epoch_loss / len(data_train)
    return epoch_loss / len(data_train)

def training_script(path, model, data_train, data_train_dec, data_test, data_test_dec, vocab, optimizer, criterion, one_hot_flag, num_epochs=15, device=0):
    writer = SummaryWriter(path)
    train(model, data_train, data_train_dec, data_test, data_test_dec, vocab, optimizer, criterion, one_hot_flag, writer, num_epochs=15, device=0)
    writer.flush()
    writer.close()

######################################################################################## Ablated Model ###############################################################################################

def evaluate_ablated(new_model, data_test, data_test_dec, vocab, criterion, one_hot_flag, device=0):
    new_model.eval()
    epoch_loss = 0
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    for i, data in tqdm.tqdm(enumerate(data_test), total=len(data_test)):
        # Get the inputs
        input_seq_dec = data_test_dec[i]
        if one_hot_flag:
            input_decoder_one_hot = one_hot_encoding(input_seq_dec.tolist(), vocab)
            # mask a token
            masked_sequence_dec, positions = mask_random_spins(input_seq_dec, vocab, mask_token=2, one_hot_flag=True)
        
            # Forward pass
            outputs = new_model.forward(masked_sequence_dec)
            target_tokens = torch.where(input_decoder_one_hot[positions]==1)[1]
        else:
            # mask a token
            masked_sequence_dec, positions = mask_random_spins(input_seq_dec, vocab, mask_token=2)
        
            # Forward pass
            outputs = new_model.forward(masked_sequence_dec).squeeze(0)
            target_tokens = input_seq_dec[positions] #target_token = input_seq[position]
            target_tokens = torch.tensor([vocab[int(token)] for token in target_tokens], dtype=torch.long)

        # Compute loss
        loss = criterion(outputs[positions], target_tokens)
        epoch_loss += loss.item()
    return epoch_loss / len(data_test)

def train_ablated(model, new_model, data_train, data_train_dec, data_test, data_test_dec, vocab, optimizer, criterion, one_hot_flag, writer, num_epochs=15, device=0):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # Training loop
    model.train()
    best_eval_loss = 1e-3 # used to do early stopping

    for epoch in tqdm.tqdm(range(num_epochs), leave=False, position=0):
        running_loss = 0
        epoch_loss = 0
        
        for i, data in tqdm.tqdm(enumerate(data_train), total=len(data_train)):
            # Get the inputs
            input_seq_enc = data
            input_seq_dec = data_train_dec[i]

            if one_hot_flag:
                input_encoder_one_hot = torch.stack([one_hot_encoding(input_seq_enc[i].tolist(), vocab) for i in range(len(input_seq_enc))], dim=0)
                input_encoder_one_hot = torch.tensor(input_encoder_one_hot, dtype=torch.float)

                input_decoder_one_hot = one_hot_encoding(input_seq_dec.tolist(), vocab)

                # mask tokens in decoder
                masked_sequence_dec, positions = mask_random_spins(input_seq_dec, vocab, mask_token=2, one_hot_flag=True)

                # Forward pass
                predictions = model.forward(input_encoder_one_hot, masked_sequence_dec)
                target_tokens = torch.where(input_decoder_one_hot[positions]==1)[1]
            else:
                input_seq_enc = torch.tensor(input_seq_enc, dtype=torch.long)
                # mask tokens in decoder
                masked_sequence_dec, positions = mask_random_spins(input_seq_dec, vocab, mask_token=2, one_hot_flag=False)

                # Forward pass
                predictions = model.forward(input_seq_enc, masked_sequence_dec).squeeze(0) #masked_sequence[masked_position])
                target_tokens = input_seq_dec[positions] #input_seq[masked_position]
                target_tokens = torch.tensor([vocab[int(token)] for token in target_tokens], dtype=torch.long)
            
            # Compute loss
            loss = criterion(predictions[positions], target_tokens)
            epoch_loss += loss.item()
            
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 10 == 9. :    # print every 10 mini-batches
                writer.add_scalar("Running Loss", running_loss / 100, epoch)
                #print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

        model.eval()
        new_model.decoder_layer = model.decoder_layer
        new_model.fc = model.fc
        print(f'Epoch {epoch + 1} | Train Loss: {(epoch_loss / len(data)):.4f}')
        writer.add_scalar("Train Loss", epoch_loss / len(data), epoch)
        eval_loss = evaluate_ablated(new_model, data_test, data_test_dec, vocab, criterion, one_hot_flag, device=device)
        writer.add_scalar("Eval Loss", eval_loss, epoch)
        print(f'Epoch {epoch + 1} | Eval Loss: {(eval_loss):.4f}')
        
        # Perform early stopping based on eval loss
        if eval_loss < best_eval_loss:
            return epoch_loss / len(data_train)
    return epoch_loss / len(data_train)

def training_script_ablated(path, model, new_model, data_train, data_train_dec, data_test, data_test_dec, vocab, optimizer, criterion, one_hot_flag, num_epochs=15, device=0):
    writer = SummaryWriter(path)
    train_ablated(model, new_model, data_train, data_train_dec, data_test, data_test_dec, vocab, optimizer, criterion, one_hot_flag, writer, num_epochs=15, device=0)
    writer.flush()
    writer.close()