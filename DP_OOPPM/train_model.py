import DP_OOPPM.predictive_models as predictive_models
import DP_OOPPM.loss_functions as loss_functions

import torch


#! check if standard values make sense
def train_and_return_LSTM(X_train, y_train, loss_function, vocab_size, embed_size=64, dropout=0.2, lstm_size=64, num_lstm=1, max_length=8, learning_rate=0.001, max_epochs=200):

    #Define the model
    model = predictive_models.LSTM_Model(vocab_size, embed_size, dropout, lstm_size, num_lstm, max_length)
    
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

    #! decide if we want to use Adam or not, check what state of art is now
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(max_epochs):
        losses = []
        outputs = model(X_train)
        #outputs = model(X_train.unsqueeze(-1)).squeeze()
        y_train =   _train.unsqueeze(-1) #you need this if your LSTM model only predicts 1 probability
        optimizer.zero_grad()
        loss = optimizer(outputs, y_train)
        loss.backwards()
        optimizer.step()
        losses.append([loss.item()])
        if (epoch+1)%10 == 0:
            print(f"Epoch [{epoch+1}/{max_epochs}], Loss: {loss.item():.4f}")

    #return model in test state
    model.test()
    return model()