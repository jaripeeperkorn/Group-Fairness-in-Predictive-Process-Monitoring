import Preprocessing.import_data as imp
import Preprocessing.log_preparation_specific as prepare
import DP_OOPPM.train_model as train_model
import DP_OOPPM.evaluate_model as eval

log = imp.import_xes("Datasets/hiring_log_high")
tr_X, tr_y, te_X, te_y, voc_sizes, num_numerical = prepare.prepare_log(df = log, 
                                                        log_name= "hiring", 
                                                        max_prefix_len = 10, 
                                                        test_fraction=0.3)

model, loss_history = train_model.train_and_return_LSTM(X_train = tr_X, y_train= tr_y,
                                                        loss_function = 'BCE', 
                                                        vocab_sizes = voc_sizes, 
                                                        num_numerical_features=num_numerical, 
                                                        dropout=0.2, lstm_size=64, num_lstm=1, max_length=8, learning_rate=0.001, max_epochs=200,
                                                        get_history=True)

