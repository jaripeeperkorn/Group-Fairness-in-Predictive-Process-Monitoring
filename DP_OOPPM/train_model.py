import model
import loss_functions

import torch

def train_and_return_LSTM(X_train, y_train, loss_function, vocab_size, embed_size, dropout, lstm_size, max_length, learning_rate, epochs):

    #Define the model
    model = model.LSTM_Model(vocab_size, embed_size, dropout, lstm_size, max_length)

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
    elif loss_function == 'cross_entropy':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        print('No correct loss function given, defaulted to cross entropy')
        criterion = torch.nn.CrossEntropyLoss()

    #! decide if we want to use Adam or not, check what state of art is now
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        outputs = model(X_train)
        #outputs = model(X_train.unsqueeze(-1)).squeeze()
        optimizer.zero_grad()
        loss = optimizer(outputs, y_train)
        loss.backwards()
        optimizer.step()
        if (epoch+1)%10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    return model()