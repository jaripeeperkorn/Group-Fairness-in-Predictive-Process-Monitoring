from scipy.stats import gaussian_kde
import numpy as np
import torch

from statsmodels.distributions.empirical_distribution import ECDF


#! CHECK IF THIS IS ACTUALLY WHAT WE WANT TO DO → SIMON'S DP MORE COMPLICATED
def calculate_demographic_parity(y_pred: torch.Tensor, s: torch.Tensor) -> float:
    """
    Calculate the demographic parity (DP) of a model's predictions.

    Args:
    y_pred (torch.Tensor): The model's predicted values.
    s (torch.Tensor): The sensitive attribute values.

    Returns:
    float: The demographic parity value.

    Raises:
    ValueError: If y_pred and s are not tensors of the same length.
    """
    if not isinstance(y_pred, torch.Tensor) or not isinstance(s, torch.Tensor) or len(y_pred) != len(s):
        raise ValueError("Input validation failed: y_pred and s must be tensors of the same length.")
    
    # Select the predicted values corresponding to s == 0
    y0 = y_pred[torch.where(s == 0)]
    # Select the predicted values corresponding to s == 1
    y1 = y_pred[torch.where(s == 1)]
    #demographic partity is just the difference in average 
    DP_value = torch.abs(torch.mean(y0) - torch.mean(y1)).item()
    return DP_value

#! TO DO CHECK THESE FUNCTIONS
def ABPC(y_pred, y_gt, z_values, bw_method="scott", sample_n=5000):
    """
    Calculate the Area Between Probability Curves (ABPC) for two groups of predicted values using KDE.
    
    Args:
    y_pred (numpy array): Predicted values
    y_gt (numpy array): Ground truth values
    z_values (numpy array): Binary values indicating group membership
    bw_method (str): Bandwidth method for KDE (default is "scott")
    sample_n (int): Number of samples for integration (default is 5000)
    
    Returns:
    float: Computed ABPC value
    """
    y_pred = np.ravel(y_pred)
    y_gt = np.ravel(y_gt)
    z_values = np.ravel(z_values)

    y_pre_1 = y_pred[z_values == 1]
    y_pre_0 = y_pred[z_values == 0]

    kde0 = gaussian_kde(y_pre_0, bw_method=bw_method)
    kde1 = gaussian_kde(y_pre_1, bw_method=bw_method)

    x = np.linspace(0, 1, sample_n)

    kde1_x = kde1(x)
    kde0_x = kde0(x)

    abpc = np.trapz(np.abs(kde0_x - kde1_x), x)

    return abpc



def ABCC( y_pred, y_gt, z_values, sample_n = 10000 ):
    """
    Calculate the Area Between two Cumulative Curves (ABCC) metric.

    Parameters:
    y_pred (array): Predicted values.
    y_gt (array): Ground truth values.
    z_values (array): Binary values indicating the group membership of each sample.
    sample_n (int, optional): Number of samples to generate for the integration (default is 10000).

    Returns:
    float: The computed ABCC value.
    """
    # Flatten the input arrays
    y_pred = y_pred.ravel()
    y_gt = y_gt.ravel()
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