import DP_OOPPM.predictive_models as predictive_models
import DP_OOPPM.custom_loss_functions as custom_loss_functions


import math
import torch
from torch.utils.data import DataLoader, TensorDataset

import numpy as np



#! check if standard values make sense
def train_and_return_LSTM(X_train, seq_len_train, y_train, s_train, 
                          loss_function, 
                          vocab_sizes, embed_sizes=None, num_numerical_features=None, 
                          dropout=0.2, lstm_size=64, num_lstm=1, bidirectional = False,
                          max_length=8, 
                          learning_rate=0.001, max_epochs=200, batch_size=128,
                          patience=10, 
                          get_history=False, 
                          X_val=None, seq_len_val=None, y_val=None, s_val=None,
                          balance_fair_BCE = 0.1):


    if embed_sizes == None:
        #Default equal to square root
        embed_sizes = [math.ceil(math.sqrt(vocab_sizes[i])) for i in range(len(vocab_sizes))]
    if num_numerical_features == None:
        num_numerical_features = 0
    
    
    # Setting up GPU 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Define the model
    model = predictive_models.LSTM_Model(vocab_sizes=vocab_sizes, embed_sizes=embed_sizes, 
                                         num_numerical_features=num_numerical_features, max_length=max_length, 
                                         dropout=dropout, lstm_size=lstm_size, num_lstm=num_lstm, bidirectional=bidirectional)

    # Assign to device
    model.to(device)
    
    #set model to train
    model.train()

    #Pick a loss function
    if loss_function == 'dp':
        criterion = custom_loss_functions.dp_reg()
    elif loss_function == 'diff_abs':
        criterion = custom_loss_functions.diff_abs_reg()
    elif loss_function == 'wasserstein':
        criterion = custom_loss_functions.wasserstein_reg()
    elif loss_function == 'KL_divergence':
        criterion = custom_loss_functions.KL_divergence_reg()
    elif loss_function == 'diff_quadr':
        criterion = custom_loss_functions.diff_quadr_reg()
    elif loss_function == 'histogram':
        criterion = custom_loss_functions.histogram_reg()
    elif loss_function == 'histogram_sq':
        criterion = custom_loss_functions.histogram_sq_reg()
    elif loss_function == 'BCE':
        criterion = torch.nn.BCELoss()
    else:
        print('No correct loss function given, defaulted to binary cross entropy')
        loss_function = 'BCE'
        criterion = torch.nn.BCELoss()
    
    #if fair loss we also use bce as balance
    criterion_bce = torch.nn.BCELoss()

    # Optimizer and scheduler used by benchmark
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.75, patience=10, threshold=0.001,
        threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08
    )

    # Create DataLoader for training and validation
    train_loader, val_loader = prepare_data_loaders(X_train, seq_len_train, y_train, s_train, X_val, seq_len_val, y_val, s_val, batch_size)

    # Early stopping variables
    best_val_loss = np.inf
    epochs_without_improvement = 0
    early_stopping = False

    losses = []
    val_losses = []

    # Store the best model state
    best_model_state = None

    #load data to device only once
    #X_train, seq_len_train, y_train, s_train = X_train.to(device), seq_len_train.to(device), y_train.to(device), s_train.to(device)
    #if X_val is not None and y_val is not None and s_val is not None:
    #    X_val, seq_len_val, y_val, s_val = X_val.to(device), seq_len_val.to(device), y_val.to(device), s_val.to(device)

    for epoch in range(max_epochs):
        if early_stopping:
            break
        # Set model to training mode each epoch
        model.train()

        epoch_loss = 0.0

        # Training loop over each batch
        for X_batch, seq_len_batch, y_batch, s_batch in train_loader:
            X_batch, seq_len_batch, y_batch, s_batch = X_batch.to(device), seq_len_batch.to(device), y_batch.to(device), s_batch.to(device)

            # Forward pass
            outputs = model(X_batch, seq_len_batch)
            #restore greadients to zero
            optimizer.zero_grad()
            
            # Calculate loss
            loss = calculate_loss(outputs, y_batch, s_batch, criterion, criterion_bce, loss_function, balance_fair_BCE)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Track average epoch loss
        avg_train_loss = epoch_loss / len(train_loader)
        losses.append(avg_train_loss)


        # Validation step if validation data is provided
        if X_val is not None and seq_len_val is not None and y_val is not None and s_val is not None:
            model.eval()
            val_epoch_loss = 0.0
            with torch.no_grad():
                for X_val_batch, seq_len_val_batch, y_val_batch, s_val_batch in val_loader:
                    # Move tensors to device
                    X_val_batch, seq_len_val_batch, y_val_batch, s_val_batch = X_val_batch.to(device), seq_len_val_batch.to(device), y_val_batch.to(device), s_val_batch.to(device)
                    val_outputs = model(X_val_batch, seq_len_val_batch)
                    
                    # Calculate validation loss
                    val_loss = calculate_loss(val_outputs, y_val_batch, s_val_batch, criterion, criterion_bce, loss_function, balance_fair_BCE)
                    
                    val_epoch_loss += val_loss.item()
            
            avg_val_loss = val_epoch_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            # Check for improvement in validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                best_model_state = model.state_dict()
            else:
                epochs_without_improvement += 1

            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}, no improvement in validation loss.")
                early_stopping = True

        lr_scheduler.step(avg_train_loss)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{max_epochs}], Loss: {avg_train_loss:.4f}")
            if X_val is not None and y_val is not None:
                print(f"Validation Loss: {avg_val_loss:.4f}")
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    model.eval()

    if get_history:
        return model, losses, val_losses
    else:
        return model
    
def calculate_loss(outputs, y_batch, s_batch, criterion, criterion_bce, loss_function, balance_fair_BCE):
    if loss_function == 'BCE':
        return criterion(outputs, y_batch)
    #! change this when loss functions cleaned up
    fair_loss, _, _, _ = criterion(outputs, s_batch, y_batch, 0.0, 1.0)
    bce_loss = criterion_bce(outputs, y_batch)
    return (1.0 - balance_fair_BCE) * bce_loss + balance_fair_BCE * fair_loss


def prepare_data_loaders(X_train, seq_len_train, y_train, s_train, X_val, seq_len_val, y_val, s_val, batch_size):
    train_dataset = TensorDataset(X_train, seq_len_train, y_train, s_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if X_val is not None and seq_len_val is not None and y_val is not None and s_val is not None:
        val_dataset = TensorDataset(X_val, seq_len_val, y_val, s_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None
    return train_loader, val_loader