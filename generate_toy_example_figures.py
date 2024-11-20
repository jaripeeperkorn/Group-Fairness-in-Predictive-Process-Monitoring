import numpy as np
import DP_OOPPM.evaluate_model as ev
import DP_OOPPM.plot_curves as plot_curves
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import gaussian_kde

from statsmodels.distributions.empirical_distribution import ECDF


plt.rcParams.update({'figure.autolayout': True})
# Define a function to calculate demographic parity (deltaDP)
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

def plot_curves(output_y, s, sensitive_feature, name_plot):

    plt.clf()

    y_pre_1 = output_y[s.flatten() == 1]
    y_pre_0 = output_y[s.flatten() == 0]

    sns.set_style("white")
    sns.set_palette(None)

    # Plot the distributions with specified colors
    sns.kdeplot(y_pre_1.squeeze(), label='s==1', color='blue')
    sns.kdeplot(y_pre_0.squeeze(), label='s==0', color='orange')

    
    # Add labels and title
    plt.xlabel('Probability')
    plt.ylabel('Density')
    plt.title(f'Probability Distribution of {sensitive_feature} = 0 and = 1.')

    # Set x-axis limits
    plt.xlim(0, 1)

    # Add a legend
    plt.legend()

    # Show the plot
    print("saving ", name_plot)
    plt.savefig(name_plot, format="pdf", bbox_inches="tight")

# Set the number of samples for each group
n_samples = 1000000

# Generate binary group indicators (s array), equally split between 0 and 1
s = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

# Generate y probabilities based on group s
# - For group s == 0, y follows a normal distribution around 0.5
# - For group s == 1, y follows a bimodal distribution with peaks at 0.2 and 0.8
y = np.zeros(n_samples)
y[s == 0] = np.random.normal(0.5, 0.05, n_samples // 2)
y[s == 1] = np.concatenate([
    np.random.normal(0.2, 0.025, n_samples // 4),
    np.random.normal(0.8, 0.025, n_samples // 4)
])

# Clip y to ensure all values are within the probability range [0, 1]
y = np.clip(y, 0, 1)

plot_filename = "Extrafigs/toy_plot_disrt.pdf"

plot_curves(y, s, "s", plot_filename)

thresholds = np.linspace(0, 1, 1000)
deltaDP_values = [demographic_parity(y, s, t) for t in thresholds]

# Plot deltaDP vs. threshold
plt.figure()
plt.plot(thresholds, deltaDP_values, color='blue', label="Demographic Parity (Binary)")
plt.axvline(0.5, color='red', linestyle='--', label="Threshold at 0.5")
plt.xlabel("Threshold")
# Set x-axis limits
plt.xlim(0, 1)
plt.ylabel("Demographic Parity")
plt.title("Demographic Parity (binary) for different threshold values")
plt.legend()
plt.savefig("Extrafigs/toy_plot_DP.pdf", format="pdf", bbox_inches="tight")


print("DP_c:", demographic_parity(y_pred=y, z_values=s, threshold=None))
print("ABPC:", ABPC(y, s))
print("ABCC:", ABCC(y, s))