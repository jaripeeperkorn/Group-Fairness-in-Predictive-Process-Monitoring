import Preprocessing.import_data as imp
import Preprocessing.log_preparation_specific as prepare
import Preprocessing.list_to_tensor as convert
import DP_OOPPM.train_model as train_model
import DP_OOPPM.evaluate_model as eval

import numpy as np





#todo:
#! ADD REG DELAY: A FEW EPOCHS WHERE WE NEVER USE FAIR LOSS ONLY BCE? → ask Simon if makes sense


#! Add masking (or fix padding)
#! Change default values training/LSTM
#! Change default embedding sizes + add possibility for manual choosing

#!fix gpu

#! fix image plot




log = imp.import_xes("Datasets/hiring_log_high.xes")


tr_X, tr_y, tr_s, val_X, val_y, val_s, te_X, te_y, te_s, vocsizes, num_numerical_features = prepare.prepare_log(df = log, log_name= "hiring", max_prefix_len = 8, test_fraction=0.2, 
                                                                                                                return_valdiation_set = True, validation_fraction=0.2,
                                                                                                                act_label = 'concept:name', case_id='case:concept:name', 
                                                                                                                sensitive_column = 'case:gender', drop_sensitive=False)

X_train, seq_len_train = convert.nested_list_to_tensor(tr_X)
y_train = convert.list_to_tensor(tr_y)
s_train = convert.list_to_tensor(tr_s)

X_val, seq_len_val = convert.nested_list_to_tensor(val_X)
y_val = convert.list_to_tensor(val_y)
s_val = convert.list_to_tensor(val_s)

y_train = y_train.view(-1, 1)
y_val = y_val.view(-1, 1)


te_X, seq_len_te = convert.nested_list_to_tensor(te_X)
te_y = np.array(te_y)
te_s = np.array(te_s)



lambdas = [0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 0.99]

for lam in lambdas:
    model, loss_history, val_loss_history = train_model.train_and_return_LSTM(X_train = X_train, seq_len_train=seq_len_train, y_train= y_train, s_train=s_train,
                                                        loss_function = 'diff_quadr', 
                                                        vocab_sizes = vocsizes, 
                                                        num_numerical_features=num_numerical_features, 
                                                        dropout = 0.2, lstm_size = 32, num_lstm=1, bidirectional=False,
                                                        max_length=8, learning_rate=0.0005, 
                                                        max_epochs = 300, batch_size=128,
                                                        patience = 30,
                                                        get_history=True, X_val=X_val, seq_len_val=seq_len_val, y_val=y_val, s_val=s_val,
                                                        balance_fair_BCE = lam)
    te_output = model(te_X, seq_len_te)
    te_np = te_output.detach().numpy()

    print("############################ RESULTS FOR: ", lam, "############################")

    eval.get_evaluation(y_gt= te_y, y_pred=te_np, s=te_s, binary_threshold = 0.5)
    