import pandas as pd
import matplotlib.pyplot as plt

def plot_pareto_curve(csv_path, plot_path, performance_column='auc', fairness_column='abcc', lambda_column='lambda'):
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

        # Configure font sizes globally
        plt.rcParams.update({
            'font.size': 15,
            'axes.labelsize': 20,
            'axes.titlesize': 20,
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
            'legend.fontsize': 15
        })

        X_labels = {'abcc':'ABCC', 'abpc': 'ABPC', 'dpe':'∆DP (continuous)'}

        plt.clf()

        # Pareto front calculation
        pareto_points = []
        sorted_indices = sorted(range(len(abcc_values)), key=lambda i: (abcc_values[i], -auc_values[i]))
        max_auc_so_far = float('-inf')

        for idx in sorted_indices:
            if auc_values[idx] > max_auc_so_far:
                pareto_points.append(idx)
                max_auc_so_far = auc_values[idx]

        # Plot the points
        plt.figure(figsize=(10, 6))
        plt.scatter(abcc_values, auc_values, c='blue', alpha=0.7, label='All Points')

        # Highlight Pareto front
        pareto_abcc = [abcc_values[i] for i in pareto_points]
        pareto_auc = [auc_values[i] for i in pareto_points]
        plt.plot(pareto_abcc, pareto_auc, 'r--', linewidth=2, label='Pareto Front')
        plt.scatter(pareto_abcc, pareto_auc, c='red', alpha=0.9, label='Pareto Points')

        # Annotate points with lambda values
        for i, lam in enumerate(lambda_values):
            plt.text(abcc_values[i], auc_values[i], f'λ={lam:.2f}', fontsize=15, ha='left', va='top')

        # Add axis labels and title
        plt.xlabel(X_labels[fairness_column])
        plt.ylabel('AUC')
        plt.title(f'AUC vs {X_labels[fairness_column]} for Different λ Values')
        plt.grid(alpha=0.5)
        plt.legend()
        plt.tight_layout()

        # Show the plot
        print("saving ", plot_path)
        plt.savefig(plot_path, format="pdf", bbox_inches="tight")

    except FileNotFoundError:
        print(f"CSV file not found at: {csv_path}")
    except KeyError as e:
        print(f"Missing expected column in CSV: {e}")

def save_all_curves(logname, addendum):

    if logname == 'hiring':
        binarys = ['case:german speaking', 'case:gender', 'case:citizen', 'case:protected', 'case:religious']
    elif logname == 'hospital':
        binarys = ['case:german speaking', 'case:private_insurance', 'case:underlying_condition', 'case:gender', 'case:citizen', 'protected']
    elif logname == 'lending':
        binarys = ['case:german speaking', 'case:gender', 'case:citizen', 'case:protected']
    elif logname == 'renting':
        binarys = ['case:german speaking', 'case:gender', 'case:citizen', 'case:protected', 'case:married']


    fairness_metrics = ['abcc', 'abpc', 'dpe']

    loss_fcts = ['wasserstein']

    for loss_fct in loss_fcts:
        for sens in binarys:
            for fairness_metric in fairness_metrics:
                csv_loc = f"Experiment2_full_results/{loss_fct}/{logname}_{addendum}/{sens}/full_results"
                csv_loc = csv_loc.replace(" ", "").replace(":", "").replace(".","")
                csv_loc = csv_loc + ".csv"
                plot_loc =  f"Experiment2_full_results/{loss_fct}/{logname}_{addendum}/{sens}/pareto_plot_{fairness_metric}"
                plot_loc = plot_loc.replace(" ", "").replace(":", "").replace(".","")
                plot_loc = plot_loc + ".pdf"

                print(csv_loc)

                plot_pareto_curve(csv_path=csv_loc, plot_path=plot_loc, performance_column='auc', fairness_column=fairness_metric, lambda_column='lambda')



#save_all_curves('hiring', 'high')

#save_all_curves('lending', 'high')

save_all_curves('renting', 'high')

