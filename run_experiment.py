import Preprocessing.import_data as imp
import Preprocessing.log_preparation_specific as prepare
import Preprocessing.list_to_tensor as convert
import DP_OOPPM.train_model as train_model
import DP_OOPPM.evaluate_model as eval

import numpy as np



#todo:
#! Add masking (or fix padding)
#! Change default values training/LSTM
#! Change default embedding sizes + add possibility for manual choosing




log = imp.import_xes("Datasets/hiring_log_high.xes")


tr_X, tr_y, tr_s, val_X, val_y, val_s, te_X, te_y, te_s, vocsizes, num_numerical_features = prepare.prepare_log(df = log, log_name= "hiring", max_prefix_len = 10, test_fraction=0.3, 
                                                                                                                return_valdiation_set = True, validation_fraction=0.1,
                                                                                                                act_label = 'concept:name', case_id='case:concept:name', 
                                                                                                                sensitive_column = 'case:gender', drop_sensitive=False)

X_train = convert.nested_list_to_tensor(tr_X)
y_train = convert.list_to_tensor(tr_y)

X_val = convert.nested_list_to_tensor(val_X)
y_val = convert.list_to_tensor(val_y)

y_train = y_train.view(-1, 1)
y_val = y_val.view(-1, 1)


te_X = convert.nested_list_to_tensor(te_X)
te_y = np.array(te_y)
te_s = np.array(te_s)


model, loss_history, val_loss_history = train_model.train_and_return_LSTM(X_train = X_train, y_train= y_train,
                                                        loss_function = 'BCE', 
                                                        vocab_sizes = vocsizes, 
                                                        num_numerical_features=num_numerical_features, 
                                                        dropout=0.2, lstm_size=64, num_lstm=1, max_length=8, learning_rate=0.001, max_epochs=100, patience =10,
                                                        get_history=True, X_val=X_val, y_val=y_val)



te_output = model(te_X)

te_np = te_output.detach().numpy()

eval.get_evaluation(y_gt= te_y, y_pred=te_np, s=te_s, binary_threshold = 0.5)