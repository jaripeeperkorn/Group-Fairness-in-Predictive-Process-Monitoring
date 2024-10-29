import DP_OOPPM.preprocessing as pre
import DP_OOPPM.train_model


train_X, train_y, train_s, test_X, test_y, test_s = pre.get_train_test("Datasets/hiring_log_medium.xes", "hiring", 0.7, "gender")