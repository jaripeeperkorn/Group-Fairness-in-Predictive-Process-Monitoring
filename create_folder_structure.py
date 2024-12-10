import os

def recreate_folders_only(folder_structure):
    """
    Recreates folder structure from a textual representation in the current directory.
    
    :param folder_structure: The folder structure as a string
    """
    base_path = os.getcwd()
    print(f"Recreating folder structure in: {base_path}\n")
    
    current_path = base_path
    folder_stack = []

    for line in folder_structure.splitlines():
        if not line.strip():  # Skip empty lines
            continue

        # Determine the level based on the indentation
        level = (len(line) - len(line.lstrip(' '))) // 4

        # Remove any deeper levels that are no longer applicable
        while len(folder_stack) > level:
            folder_stack.pop()

        # Get the current folder name
        entry = line.strip()

        if entry.endswith('/'):  # It's a folder
            folder_name = entry.rstrip('/')
            current_path = os.path.join(base_path, *folder_stack, folder_name)
            os.makedirs(current_path, exist_ok=True)  # Create the folder
            folder_stack.append(folder_name)

# Example usage
sample_folder_structure = """
Datasets/
DP_OOPPM/
Extrafigs/
Preprocessing/
Results/
    Experiment1_full_results_no_removal/
        figs/
    Experiment1_full_results_with_removal/
        figs/
    Experiment2_full_results/
        wasserstein/
            hiring_high/
                casecitizen/
                casegender/
                casegermanspeaking/
                caseprotected/
                casereligious/
            lending_high/
                casecitizen/
                casegender/
                casegermanspeaking/
                caseprotected/
            renting_high/
                casecitizen/
                casegender/
                casegermanspeaking/
                casemarried/
                caseprotected/
    Hyperparameters/
        BCE/
"""

recreate_folders_only(sample_folder_structure)