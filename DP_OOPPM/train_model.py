import DP_OOPPM.predictive_models as predictive_models
import DP_OOPPM.loss_functions as loss_functions


import math
import torch

import numpy as np



#! check if standard values make sense
def train_and_return_LSTM(X_train, y_train, loss_function, 
                          vocab_sizes, embed_sizes=None, num_numerical_features=None, 
                          dropout=0.2, lstm_size=64, 
                          num_lstm=1, max_length=8, learning_rate=0.001, max_epochs=200,
                          patience=10, get_history=False, X_val=None, y_val=None):


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
                                         dropout=dropout, lstm_size=lstm_size, num_lstm=num_lstm)

    # Assign to GPU 
    model.to(device)
    
    #set model to train
    model.train()

    #Pick a loss function
    if loss_function == 'dp':
        criterion = loss_functions.dp_reg()
    elif loss_function == 'diff_abs':
        criterion = loss_functions.diff_abs_reg()
    elif loss_function == 'wasserstein':
        criterion = loss_functions.wasserstein_reg()
    elif loss_function == 'KL_divergence':
        criterion = loss_functions.KL_divergence_reg()
    elif loss_function == 'diff_quadr':
        criterion = loss_functions.diff_quadr_reg()
    elif loss_function == 'histogram':
        criterion = loss_functions.histogram_reg()
    elif loss_function == 'histogram_sq':
        criterion = loss_functions.histogram_sq_reg()
    elif loss_function == 'BCE':
        criterion = torch.nn.BCELoss()
    else:
        print('No correct loss function given, defaulted to binary cross entropy')
        criterion = torch.nn.BCELoss()

    #! decide if we want to use NAdam or not, check what state of art is now

    # Optimizer and scheduler used by benchmark
    optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=16, threshold=0.0001,
        threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08
    )

    # Move data to device
    X_train, y_train = X_train.to(device), y_train.to(device)

    # Early stopping variables
    best_val_loss = np.inf
    epochs_without_improvement = 0

    losses = []
    val_losses = []

    # Store the best model state
    best_model_state = None


    for epoch in range(max_epochs):
        # Set model to training mode each epoch
        model.train()

        # Forward pass
        outputs = model(X_train)
        
        # Adjust y_train shape if needed for loss function compatibility
        #y_train = y_train.unsqueeze(-1)

        # Zero gradients, compute loss, backward pass, and optimize
        optimizer.zero_grad()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Track loss and adjust learning rate as needed
        losses.append(loss.item())

        # Validation step (only if validation data is provided)
        if X_val is not None and y_val is not None:
            model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                val_losses.append(val_loss.item())

                # Check if validation loss improved
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0  # Reset patience counter
                    # Save the current model's state as the best model
                    best_model_state = model.state_dict()
                else:
                    epochs_without_improvement += 1

                # Check for early stopping
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch+1}, no improvement in validation loss.")
                    break

        lr_scheduler.step(loss)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{max_epochs}], Loss: {loss.item():.4f}")
            if X_val is not None and y_val is not None:
                print(f"Validation Loss: {val_loss.item():.4f}")
            # Print learning rate
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
    # After training, load the model with the best validation loss
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Set model to evaluation mode before returning
    model.eval()

    if get_history == True:
        return model, losses, val_losses
    else:
        return model
    