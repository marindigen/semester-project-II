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
import wandb

from utils import *
from model import *

# for tensorboard
from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter()

########################################################################################## Normal Model #################################################################################################

def evaluate(model, data_test, data_test_dec, vocab, criterion, one_hot_flag, device=0):
    model.eval()
    epoch_loss = 0
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    for data_enc, data_dec in tqdm.tqdm(zip(data_test, data_test_dec), total=len(data_test)):
        # Get the inputs
        input_seq_enc = data_enc
        input_seq_dec = data_dec
        if one_hot_flag:
            input_encoder_one_hot = torch.stack([one_hot_encoding(input_seq_enc[i].tolist(), vocab) for i in range(len(input_seq_enc))], dim=0)
            input_encoder_one_hot = torch.tensor(input_encoder_one_hot, dtype=torch.float)

            input_decoder_one_hot = one_hot_encoding(input_seq_dec.tolist(), vocab)
            # mask tokens
            masked_sequence_dec, positions = mask_random_spins(input_seq_dec, vocab, mask_token=2, one_hot_flag=True)
        
            # Forward pass
            outputs = model.forward(input_encoder_one_hot, masked_sequence_dec)
            target_tokens = torch.where(input_decoder_one_hot[positions]==1)[1]
        else:
            # mask tokens
            masked_sequence_dec, positions = mask_random_spins_batch(input_seq_dec, vocab, mask_token=2)
        
            # Forward pass
            predictions = model.forward(input_seq_enc, masked_sequence_dec).squeeze(0) #masked_sequence[masked_position])
            target_tokens = [input_seq_dec[l][positions[l]] for l in range(positions.shape[0])]
            target_tokens = torch.stack(target_tokens).flatten()
            target_tokens = torch.tensor([vocab[int(token)] for token in target_tokens], dtype=torch.long).reshape(data_dec.shape[0], -1)
            #outputs = model.forward(input_seq_enc, masked_sequence_dec).squeeze(0)
            #target_tokens = input_seq_dec[positions] #target_token = input_seq[position]
            #target_tokens = torch.tensor([vocab[int(token)] for token in target_tokens], dtype=torch.long)

        # Compute loss
        batch_indices = torch.arange(predictions.size(0)).unsqueeze(-1).expand(-1, positions.shape[1])
        output = predictions[batch_indices, positions, :].reshape(-1, predictions.shape[-1])
        loss = criterion(output, target_tokens.flatten())
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
        i = 0 
        store_cross_attn_weights = []
        store_decoder_attn_weights = []
        for data_enc, data_dec in tqdm.tqdm(zip(data_train, data_train_dec), total=len(data_train)):
            # Get the inputs
            input_seq_enc = data_enc
            input_seq_dec = data_dec

            if one_hot_flag:
                input_encoder_one_hot = torch.stack([one_hot_encoding(input_seq_enc[i].tolist(), vocab) for i in range(len(input_seq_enc))], dim=0)
                input_encoder_one_hot = torch.tensor(input_encoder_one_hot, dtype=torch.float)

                input_decoder_one_hot = one_hot_encoding(input_seq_dec.tolist(), vocab)

                # mask tokens in decoder
                masked_sequence_dec, positions = mask_random_spins_batch(input_seq_dec, vocab, mask_token=2, one_hot_flag=True)

                # Forward pass
                predictions = model.forward(input_encoder_one_hot, masked_sequence_dec)
                #predictions = prediction[positions]
                target_tokens = torch.where(input_decoder_one_hot[positions]==1)[1]
            else:
                input_seq_enc = torch.tensor(input_seq_enc, dtype=torch.long)
                # mask tokens in decoder
                masked_sequence_dec, positions = mask_random_spins_batch(input_seq_dec, vocab, mask_token=2, one_hot_flag=False)
                # Forward pass
                predictions = model.forward(input_seq_enc, masked_sequence_dec).squeeze(0) #masked_sequence[masked_position])
                target_tokens = [input_seq_dec[l][positions[l]] for l in range(positions.shape[0])] #input_seq[masked_position]
                target_tokens = torch.stack(target_tokens).flatten()
                target_tokens = torch.tensor([vocab[int(token)] for token in target_tokens], dtype=torch.long).reshape(data_dec.shape[0], -1)
            
            # Compute loss
            batch_indices = torch.arange(predictions.size(0)).unsqueeze(-1).expand(-1, positions.shape[1])
            output = predictions[batch_indices, positions, :].reshape(-1, predictions.shape[-1])
            loss = criterion(output, target_tokens.flatten())
            epoch_loss += loss.item()

            store_cross_attn_weights.append(model.decoder_layer.cross_attn_weights)
            store_decoder_attn_weights.append(model.decoder_layer.decoder_attn_weights)
            
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 10 == 9. :    # print every 10 mini-batches
                writer.add_scalar("Running Loss", running_loss / 100, epoch)
                #print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
            i += 1

        print(f'Epoch {epoch + 1} | Train Loss: {(epoch_loss / len(data_enc)):.4f}')
        writer.add_scalar("Train Loss", epoch_loss / len(data_enc), epoch)
        eval_loss = evaluate(model, data_test, data_test_dec, vocab, criterion, one_hot_flag, device=device)
        writer.add_scalar("Eval Loss", eval_loss, epoch)
        print(f'Epoch {epoch + 1} | Eval Loss: {(eval_loss):.4f}')
        wandb.log({"eval_loss": eval_loss, "train_loss": loss})
        
        # Perform early stopping based on eval loss
        if eval_loss < best_eval_loss:
            return epoch_loss / len(data_train)
    return epoch_loss / len(data_train), store_cross_attn_weights, store_decoder_attn_weights

def training_script(path, model, data_train, data_train_dec, data_test, data_test_dec, vocab, optimizer, criterion, one_hot_flag, num_epochs=15, device=0):
    writer = SummaryWriter(path)
    loss, store_cross_attn_weights, store_decoder_attn_weights = train(model, data_train, data_train_dec, data_test, data_test_dec, vocab, optimizer, criterion, one_hot_flag, writer, num_epochs=num_epochs, device=0)
    writer.flush()
    writer.close()
    return store_cross_attn_weights, store_decoder_attn_weights

    

######################################################################################## Ablated Model ###############################################################################################

def evaluate_ablated(new_model, data_test_dec, vocab, criterion, one_hot_flag, device=0):
    new_model.eval()
    epoch_loss = 0
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    for i, data in tqdm.tqdm(enumerate(data_test_dec), total=len(data_test_dec)):
        # Get the inputs
        input_seq_dec = data
        if one_hot_flag:
            input_decoder_one_hot = one_hot_encoding(input_seq_dec.tolist(), vocab)
            # mask a token
            masked_sequence_dec, positions = mask_random_spins(input_seq_dec, vocab, mask_token=2, one_hot_flag=True)
        
            # Forward pass
            outputs = new_model.forward(masked_sequence_dec)
            target_tokens = torch.where(input_decoder_one_hot[positions]==1)[1]
        else:
            # mask a token
            masked_sequence_dec, positions = mask_random_spins_batch(input_seq_dec, vocab, mask_token=2)
            # Forward pass
            predictions = new_model.forward(masked_sequence_dec) #masked_sequence[masked_position])
            target_tokens = [input_seq_dec[l][positions[l]] for l in range(positions.shape[0])]
            target_tokens = torch.stack(target_tokens).flatten()
            target_tokens = torch.tensor([vocab[int(token)] for token in target_tokens], dtype=torch.long).reshape(data.shape[0], -1)
            #outputs = new_model.forward(masked_sequence_dec)[0].squeeze(0)
            #target_tokens = input_seq_dec[positions] #target_token = input_seq[position]
            #target_tokens = torch.tensor([vocab[int(token)] for token in target_tokens], dtype=torch.long)

        # Compute loss
        batch_indices = torch.arange(predictions.size(0)).unsqueeze(-1).expand(-1, positions.shape[1])
        output = predictions[batch_indices, positions, :].reshape(-1, predictions.shape[-1])
        loss = criterion(output, target_tokens.flatten())
        epoch_loss += loss.item()
    return epoch_loss / len(data_test_dec)

def train_ablated(model, new_model, data_train, data_train_dec, data_test, data_test_dec, vocab, optimizer, criterion, one_hot_flag, writer, num_epochs=15, device=0):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # Training loop
    model.train()
    best_eval_loss = 1e-3 # used to do early stopping

    for epoch in tqdm.tqdm(range(num_epochs), leave=False, position=0):
        running_loss = 0
        epoch_loss = 0
        i = 0 
        store_cross_attn_weights = []
        store_decoder_attn_weights = []
        for data_enc, data_dec in tqdm.tqdm(zip(data_train, data_train_dec), total=len(data_train)):
            # Get the inputs
            input_seq_enc = data_enc
            input_seq_dec = data_dec

            if one_hot_flag:
                input_encoder_one_hot = torch.stack([one_hot_encoding(input_seq_enc[i].tolist(), vocab) for i in range(len(input_seq_enc))], dim=0)
                input_encoder_one_hot = torch.tensor(input_encoder_one_hot, dtype=torch.float)

                input_decoder_one_hot = one_hot_encoding(input_seq_dec.tolist(), vocab)

                # mask tokens in decoder
                masked_sequence_dec, positions = mask_random_spins_batch(input_seq_dec, vocab, mask_token=2, one_hot_flag=True)

                # Forward pass
                predictions = model.forward(input_encoder_one_hot, masked_sequence_dec)
                target_tokens = torch.where(input_decoder_one_hot[positions]==1)[1]
            else:
                #input_seq_enc = torch.tensor(input_seq_enc, dtype=torch.long)
                # mask tokens in decoder
                masked_sequence_dec, positions = mask_random_spins_batch(input_seq_dec, vocab, mask_token=2, one_hot_flag=False)

                # Forward pass
                #predictions = model.forward(input_seq_enc, masked_sequence_dec).squeeze(0) #masked_sequence[masked_position])
                #target_tokens = input_seq_dec[positions] #input_seq[masked_position]
                #target_tokens = torch.tensor([vocab[int(token)] for token in target_tokens], dtype=torch.long)
                predictions = model.forward(input_seq_enc, masked_sequence_dec) 
                target_tokens = [input_seq_dec[l][positions[l]] for l in range(positions.shape[0])] #input_seq[masked_position]
                target_tokens = torch.stack(target_tokens).flatten()
                target_tokens = torch.tensor([vocab[int(token)] for token in target_tokens], dtype=torch.long).reshape(data_dec.shape[0], -1)
            
            # Compute loss
            batch_indices = torch.arange(predictions.size(0)).unsqueeze(-1).expand(-1, positions.shape[1])
            output = predictions[batch_indices, positions, :].reshape(-1, predictions.shape[-1])
            loss = criterion(output, target_tokens.flatten())
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
            i += 1
            store_cross_attn_weights.append(model.decoder_layer.cross_attn_weights)
            store_decoder_attn_weights.append(model.decoder_layer.decoder_attn_weights)

        model.eval()
        new_model.decoder_layer = model.decoder_layer
        new_model.fc = model.fc
        print(f'Epoch {epoch + 1} | Train Loss: {(epoch_loss / len(data_enc)):.4f}')
        writer.add_scalar("Train Loss", epoch_loss / len(data_enc), epoch)
        eval_loss = evaluate_ablated(new_model, data_test_dec, vocab, criterion, one_hot_flag, device=device)
        writer.add_scalar("Eval Loss", eval_loss, epoch)
        print(f'Epoch {epoch + 1} | Eval Loss: {(eval_loss):.4f}')
        wandb.log({"eval_loss": eval_loss, "train_loss": loss})
        
        # Perform early stopping based on eval loss
        if eval_loss < best_eval_loss:
            return epoch_loss / len(data_train)
    return epoch_loss / len(data_train), store_cross_attn_weights, store_decoder_attn_weights

def training_script_ablated(path, model, new_model, data_train, data_train_dec, data_test, data_test_dec, vocab, optimizer, criterion, one_hot_flag, num_epochs=15, device=0):
    writer = SummaryWriter(path)
    loss, store_cross_attn_weights, store_decoder_attn_weights = train_ablated(model, new_model, data_train, data_train_dec, data_test, data_test_dec, vocab, optimizer, criterion, one_hot_flag, writer, num_epochs=num_epochs, device=0)
    writer.flush()
    writer.close()
    return store_cross_attn_weights, store_decoder_attn_weights