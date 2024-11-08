import Preprocessing.import_data as imp
import Preprocessing.log_preparation_specific as prepare
import Preprocessing.list_to_tensor as convert
import DP_OOPPM.train_model as train_model


from sklearn.metrics import roc_auc_score


def run_hyper(dataset_name):
    log = imp.import_xes(dataset_name)
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

    #!change this

    num_layers_lst = [1]
    bidirectional_lst = [False, True]
    LSTM_size_lst = [16, 32]
    batch_size_lst = [64, 128]
    learning_rate_lst = [0.0001, 0.001, 0.01]
    dropout_lst = [0.0, 0.2]

    for num_layers in num_layers_lst:
            for bidirectional in bidirectional_lst:
                for lstm_size in LSTM_size_lst:
                    for batch_size in batch_size_lst:
                        for learnig_rate in learning_rate_lst:
                            for dropout in dropout_lst:
                                    model = train_model.train_and_return_LSTM(X_train = X_train, seq_len_train=seq_len_train, y_train= y_train, s_train=s_train, 
                                                                            loss_function = 'BCE', vocab_sizes = vocsizes, 
                                                                            num_numerical_features=num_numerical_features, 
                                                                            dropout = dropout, lstm_size = lstm_size, num_lstm=num_layers,
                                                                            bidirectional = bidirectional, 
                                                                            max_length=8, 
                                                                            learning_rate=learnig_rate, max_epochs = 300, batch_size=batch_size,
                                                                            patience = 30, get_history=False, 
                                                                           X_val=X_val, seq_len_val=seq_len_val, y_val=y_val, s_val=s_val)
                                    val_output = model(X_val)
                                    val_np = val_output.detach().numpy()

                                    y_gt = y_val.ravel()
                                    y_pred = val_np.ravel()

                                    auc = roc_auc_score(y_gt, y_pred)


