from scipy.stats import gaussian_kde
import numpy as np
import torch

from statsmodels.distributions.empirical_distribution import ECDF

def demographic_parity(y_pred, z_values, threshold=None):
    """
    Computes the demographic parity between two groups in predicted outcomes.

    Parameters:
    - y_pred (array-like): Predicted values, typically probabilities or binary predictions.
    - z_values (array-like): Binary group indicators (e.g., 0 or 1) for each corresponding prediction in y_pred.
    - threshold (float, optional): Threshold value for binarizing predictions. If provided, predictions are
                                   classified as 1 if above the threshold, otherwise 0. If None, raw predictions are used.

    Returns:
    - float: The absolute difference in mean predictions between the two groups, representing demographic parity.
    """
    
    # Step 1: Separate predictions by group, applying the threshold if specified
    # - For the group with z_values == 1, apply the threshold (if given) or keep raw predictions
    y_z_1 = y_pred[z_values == 1] > threshold if threshold else y_pred[z_values == 1]
    # - For the group with z_values == 0, apply the threshold (if given) or keep raw predictions
    y_z_0 = y_pred[z_values == 0] > threshold if threshold else y_pred[z_values == 0]

    # Step 2: Compute the mean prediction for each group
    # - Average prediction for group z == 1
    y_z_1_mean = y_z_1.mean()
    # - Average prediction for group z == 0
    y_z_0_mean = y_z_0.mean()

    # Step 3: Calculate demographic parity as the absolute difference between group means
    parity = abs(y_z_1_mean - y_z_0_mean)

    # Step 4: Return the demographic parity value
    return parity


# Define a function named ABPC that takes in four arguments:
# y_pred: predicted values
# z_values: binary values indicating the group membership of each sample
# bw_method: bandwidth method for the kernel density estimation (default is "scott")
# sample_n: number of samples to generate for the integration (default is 10000)
def ABPC(y_pred, z_values, bw_method = "scott", sample_n = 10000):

    # Flatten the input arrays
    y_pred = y_pred.ravel()
    z_values = z_values.ravel()

    # Extract the predicted values for each group
    y_pre_1 = y_pred[z_values == 1]
    y_pre_0 = y_pred[z_values == 0]

    # Compute the kernel density estimation (KDE) for each group
    kde0 = gaussian_kde(y_pre_0, bw_method = bw_method)
    kde1 = gaussian_kde(y_pre_1, bw_method = bw_method)

    # Generate a set of x values for the integration
    x = np.linspace(0, 1, sample_n)

    # Evaluate the KDEs at the x values
    kde1_x = kde1(x)
    kde0_x = kde0(x)

    # Compute the area between the two KDEs using the trapezoidal rule
    abpc = np.trapz(np.abs(kde0_x - kde1_x), x)

    # Return the computed ABPC value
    return abpc


# Define a function named ABCC that takes in three arguments:
# y_pred: predicted values
# z_values: binary values indicating the group membership of each sample
# sample_n: number of samples to generate for the integration (default is 10000)
def ABCC(y_pred, z_values, sample_n = 10000):

    # Flatten the input arrays
    y_pred = y_pred.ravel()
    z_values = z_values.ravel()

    # Extract the predicted values for each group
    y_pre_1 = y_pred[z_values == 1]
    y_pre_0 = y_pred[z_values == 0]

    # Compute the empirical cumulative distribution function (ECDF) for each group
    ecdf0 = ECDF(y_pre_0)
    ecdf1 = ECDF(y_pre_1)

    # Generate a set of x values for the integration
    x = np.linspace(0, 1, sample_n)

    # Evaluate the ECDFs at the x values
    ecdf0_x = ecdf0(x)
    ecdf1_x = ecdf1(x)

    # Compute the area between the two ECDFs using the trapezoidal rule
    abcc = np.trapz(np.abs(ecdf0_x - ecdf1_x), x)

    # Return the computed ABCC value
    return abcc