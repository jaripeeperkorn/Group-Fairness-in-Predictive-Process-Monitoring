import DP_OOPPM.custom_metrics as custom_metrics
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

def get_evaluation(y_gt, y_pred, s, binary_threshold = 0.5):
    """
    Evaluate the performance and fairness of a predictive model.

    This function computes various evaluation metrics for a given set of ground truth
    labels and predicted values. It includes standard performance metrics such as
    accuracy, AUC, precision, recall, and F1 score, as well as fairness metrics like
    demographic parity, Area Between Probability Curves (ABPC), and Area Between
    Cumulative Curves (ABCC).

    Parameters:
        y_gt (array-like): Ground truth binary labels.
        y_pred (array-like): Predicted values, typically probabilities.
        s (array-like): Binary group indicators for each sample.
        binary_threshold (float, optional): Threshold for binarizing predictions. Default is 0.5.

    Returns:
        dict: A dictionary containing the computed metrics:
            - "accuracy": Accuracy score.
            - "auc": Area Under the ROC Curve.
            - "precision": Precision score.
            - "recall": Recall score.
            - "f1": F1 score.
            - "dp": Demographic parity with threshold.
            - "dpe": Demographic parity without threshold.
            - "abpc": Area Between Probability Curves.
            - "abcc": Area Between Cumulative Curves.
    """
    y_gt = y_gt.ravel()
    y_pred = y_pred.ravel()
    s = s.ravel()

    auc = roc_auc_score(y_gt, y_pred) #* 100

    accuracy = accuracy_score(y_gt, y_pred > binary_threshold)
    precision = precision_score(y_gt, y_pred > binary_threshold) #* 100
    recall = recall_score(y_gt, y_pred > binary_threshold)
    f1 = f1_score(y_gt, y_pred > binary_threshold) #* 100

    dp = custom_metrics.demographic_parity(y_pred, s, threshold=binary_threshold)
    dpe = custom_metrics.demographic_parity(y_pred, s, threshold=None)
    abpc = custom_metrics.ABPC(y_pred, s)
    abcc = custom_metrics.ABCC(y_pred, s)

    return {
        "accuracy": accuracy,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "dp": dp,
        "dpe": dpe,
        "abpc": abpc,
        "abcc": abcc
    }

import numpy as np

def find_best_threshold(y_val_gt, y_val_pred, thresholds=np.linspace(0, 1, 101)):
    """
    Find the best threshold that maximizes the F1 score on the validation set.

    Args:
        y_val_gt: Ground truth labels for the validation set.
        y_val_pred: Predicted probabilities for the validation set.
        s_val: Sensitive attribute labels for the validation set.
        thresholds: Array of threshold values to search over (default: 0.0 to 1.0 in 0.01 increments).

    Returns:
        A dictionary containing the best threshold, metrics at that threshold, and demographic parity.
    """
    best_threshold = 0.0
    best_f1 = 0.0

    for threshold in thresholds:
        binary_predictions = y_val_pred > threshold
        # Compute metrics
        f1 = f1_score(y_val_gt, binary_predictions)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold

def get_evaluation_extented(y_gt, y_pred, s, val_gt, val_pred):
    """
    Evaluate model predictions and compute various metrics.

    This function calculates several evaluation metrics for model predictions,
    including accuracy, AUC, precision, recall, F1 score, and fairness metrics
    such as demographic parity, ABPC, and ABCC. It also determines the optimal
    threshold for maximizing the F1 score on a validation set and recalculates
    metrics using this threshold.

    Parameters:
        y_gt (array-like): Ground truth labels for the test set.
        y_pred (array-like): Predicted probabilities for the test set.
        s (array-like): Sensitive attribute labels for the test set.
        val_gt (array-like): Ground truth labels for the validation set.
        val_pred (array-like): Predicted probabilities for the validation set.

    Returns:
        dict: A dictionary containing computed metrics, including accuracy,
            AUC, precision, recall, F1 score, demographic parity, optimal
            threshold, and fairness metrics.
    """
    
    y_gt = y_gt.ravel()
    y_pred = y_pred.ravel()
    s = s.ravel()

    val_gt =val_gt.ravel() 
    val_pred = val_pred.ravel()

    auc = roc_auc_score(y_gt, y_pred) #* 100

    accuracy = accuracy_score(y_gt, y_pred > 0.5)
    precision = precision_score(y_gt, y_pred > 0.5) #* 100
    recall = recall_score(y_gt, y_pred > 0.5)
    f1 = f1_score(y_gt, y_pred > 0.5) #* 100

    dp = custom_metrics.demographic_parity(y_pred, s, threshold=0.5)
    dpe = custom_metrics.demographic_parity(y_pred, s, threshold=None)
    abpc = custom_metrics.ABPC(y_pred, s)
    abcc = custom_metrics.ABCC(y_pred, s)

    print('Fetching optimal threshold')
    opt_threshold = find_best_threshold(val_gt, val_pred)

    accuracy_opt = accuracy_score(y_gt, y_pred > opt_threshold)
    precision_opt = precision_score(y_gt, y_pred > opt_threshold) #* 100
    recall_opt = recall_score(y_gt, y_pred > opt_threshold)
    f1_opt = f1_score(y_gt, y_pred > opt_threshold) #* 100

    dp_opt = custom_metrics.demographic_parity(y_pred, s, threshold=opt_threshold)


    return {"accuracy": accuracy, 
            "auc": auc, 
            "precision": precision, 
            "recall": recall, 
            "f1": f1, 
            "dp": dp, 
            "optimal_threshold":opt_threshold, 
            "accuracy_optimal": accuracy_opt, 
            "precision_optimal": precision_opt, 
            "recall_optimal": recall_opt, 
            "f1_optimal": f1_opt, 
            "dp_optimal": dp_opt,
            "dpe": dpe, 
            "abpc": abpc, 
            "abcc": abcc}
