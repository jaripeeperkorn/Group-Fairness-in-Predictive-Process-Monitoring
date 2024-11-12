import pandas as pd

def get_best_hyperparameter_combination(logname):
    """
    Load the hyperparameter tuning log and return the best hyperparameter combination based on AUC score.
    
    Parameters:
        logname (str): The base name of the log file (without the '_hyperparameter_tuning_results.csv' suffix).
        
    Returns:
        dict: The best hyperparameter combination and its corresponding AUC score.
    """
    # Define the log file path
    log_path = f"Hyperparameters/BCE/{logname}_hyperparameter_tuning_results.csv"
    
    try:
        # Load the log data
        results_df = pd.read_csv(log_path)
        
        # Check if results are empty
        if results_df.empty:
            print("The results log is empty. No hyperparameter combinations have been recorded.")
            return None
        
        # Find the best hyperparameter combination by AUC score
        best_combination = results_df.loc[results_df['auc_score'].idxmax()]
        
        # Convert the best combination to a dictionary
        best_combination_dict = best_combination.to_dict()
        
        # Print and return the best hyperparameter combination
        print("Best Hyperparameter Combination:")
        print(best_combination_dict)
        
        return best_combination_dict
    
    except FileNotFoundError:
        print(f"Log file '{log_path}' not found. Please ensure the hyperparameter tuning has been run.")
        return None
    except KeyError as e:
        print(f"Expected column not found in log file: {e}")
        return None


def find_worst_hyperparameters(logname):
    """
    Load the hyperparameter tuning log and analyze which hyperparameter choices lead to the worst average AUC scores.
    
    Parameters:
        logname (str): The base name of the log file (without the '_hyperparameter_tuning_results.csv' suffix).
        
    Returns:
        dict: A dictionary with each hyperparameter and its value that led to the worst average AUC score.
    """
    # Define the log file path
    log_path = f"Hyperparameters/BCE/{logname}_hyperparameter_tuning_results.csv"
    
    try:
        # Load the log data
        results_df = pd.read_csv(log_path)
        
        # Check if results are empty
        if results_df.empty:
            print("The results log is empty. No hyperparameter combinations have been recorded.")
            return None
        
        # Dictionary to store worst average scores for each hyperparameter
        worst_hyperparameters = {}
        
        # Iterate over each hyperparameter (excluding 'auc_score')
        for column in results_df.columns:
            if column == 'auc_score':
                continue
            
            # Calculate the average AUC score for each unique value of the hyperparameter
            avg_scores = results_df.groupby(column)['auc_score'].mean()
            
            # Find the value with the lowest average AUC score
            worst_value = avg_scores.idxmin()
            worst_score = avg_scores.min()
            
            # Store the worst-performing value and its average score
            worst_hyperparameters[column] = {
                'worst_value': worst_value,
                'average_auc_score': worst_score
            }
        
        # Print the results
        print("Hyperparameter choices leading to the worst average scores:")
        for param, info in worst_hyperparameters.items():
            print(f"{param}: {info['worst_value']} (Average AUC Score: {info['average_auc_score']:.4f})")
        
        return worst_hyperparameters
    
    except FileNotFoundError:
        print(f"Log file '{log_path}' not found. Please ensure the hyperparameter tuning has been run.")
        return None
    except KeyError as e:
        print(f"Expected column not found in log file: {e}")
        return None


    

best_combination = get_best_hyperparameter_combination('hiring')

print("BEST: ", best_combination)


worst_hyperparameters = find_worst_hyperparameters('hiring')