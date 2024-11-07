import DP_OOPPM.custom_metrics as custom_metrics

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

def get_evaluation(y_gt, y_pred, s, binary_threshold = 0.5):
   
    y_gt = y_gt.ravel()
    y_pred = y_pred.ravel()
    s = s.ravel()

    accuracy = accuracy_score(y_gt, y_pred > binary_threshold)
    auc = roc_auc_score(y_gt, y_pred) #* 100
    precision = precision_score(y_gt, y_pred > binary_threshold) #* 100
    recall = recall_score(y_gt, y_pred > binary_threshold) #* 100

    dp = custom_metrics.demographic_parity(y_pred, s)
    dpe = custom_metrics.demographic_parity(y_pred, s, threshold=None)
    abpc = custom_metrics.ABPC(y_pred, y_gt, s)
    abcc = custom_metrics.ABCC(y_pred, y_gt, s)

    print("Accuracy at threshold: ", accuracy)
    print("Precision at theshold: ", precision)
    print("Recall at theshold: ", recall)
    print("AUC: ", auc)

    print("DP with threshold: ", dp)
    print("DP no threshold: ", dpe)
    print("ABPC: ", abpc)
    print("ABCC", abcc)
