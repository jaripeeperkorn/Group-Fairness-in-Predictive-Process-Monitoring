import pandas as pd
import matplotlib.pyplot as plt

def plot_pareto_curve(csv_path, performance_column='auc', fairness_column='abcc', lambda_column='lambda'):
    """
    Plot a Pareto curve with AUC on one axis and ABCC on the other, for different values of lambda.

    Parameters:
        csv_path (str): Path to the CSV file containing results.
        auc_column (str): Column name for AUC values.
        abcc_column (str): Column name for ABCC values.
        lambda_column (str): Column name for lambda values.
    """
    try:
        # Load the results from the CSV
        results_df = pd.read_csv(csv_path)

        # Extract relevant columns
        auc_values = results_df[performance_column]
        abcc_values = results_df[fairness_column]
        lambda_values = results_df[lambda_column]

        # Plot the Pareto curve
        plt.figure(figsize=(10, 6))
        plt.scatter(abcc_values, auc_values, c='blue', alpha=0.7, label='Pareto Points')

        # Annotate points with lambda values
        for i, lam in enumerate(lambda_values):
            plt.text(abcc_values[i], auc_values[i], f'λ={lam:.2f}', fontsize=9, ha='right', va='bottom')

        # Add axis labels and title
        plt.xlabel('ABCC')
        plt.ylabel('AUC')
        plt.title('Pareto Curve: AUC vs ABCC for Different λ Values')
        plt.grid(alpha=0.5)
        plt.legend()
        plt.tight_layout()

        # Show the plot
        plt.show()

    except FileNotFoundError:
        print(f"CSV file not found at: {csv_path}")
    except KeyError as e:
        print(f"Missing expected column in CSV: {e}")

plot_pareto_curve("Custom_loss_results/KL_divergence/renting_high/caseprotected/full_results.csv")