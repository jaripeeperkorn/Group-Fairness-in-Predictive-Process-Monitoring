import DP_OOPPM.predictive_models as predictive_models
import DP_OOPPM.loss_functions as loss_functions


import math
import torch


#! check if standard values make sense
def train_and_return_LSTM(X_train, y_train, loss_function, 
                          vocab_sizes, embed_sizes=None, num_numerical_features=None, 
                          dropout=0.2, lstm_size=64, 
                          num_lstm=1, max_length=8, learning_rate=0.001, max_epochs=200,
                          get_history=False):


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
        threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True
    )

    # Move data to device
    X_train, y_train = X_train.to(device), y_train.to(device)

    losses = []

    for epoch in range(max_epochs):
            # Set model to training mode each epoch
            model.train()

            # Forward pass
            outputs = model(X_train)
            
            # Adjust y_train shape if needed for loss function compatibility
            y_train = y_train.unsqueeze(-1)

            # Zero gradients, compute loss, backward pass, and optimize
            optimizer.zero_grad()
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            # Track loss and adjust learning rate as needed
            losses.append(loss.item())
            lr_scheduler.step(loss)

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{max_epochs}], Loss: {loss.item():.4f}")

    # Set model to evaluation mode before returning
    model.eval()

    if get_history == True:
        return model, losses
    else:
        return model
    