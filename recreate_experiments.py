import subprocess
import os

def create_conda_environment(env_file):
    """
    Creates a Conda environment based on the provided YAML file.

    :param env_file: Path to the environment YAML file
    """
    if not os.path.isfile(env_file):
        print(f"Environment file not found: {env_file}")
        return False

    try:
        print(f"Creating Conda environment from {env_file}...")
        subprocess.run(["conda", "env", "create", "-f", env_file], check=True)
        print("Conda environment created successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating Conda environment: {e}")
        return False

def run_python_scripts(scripts, conda_env_name):
    """
    Activates a Conda environment and runs a series of Python scripts.

    :param scripts: List of script paths to execute
    :param conda_env_name: Name of the Conda environment to activate
    """
    for script in scripts:
        if not os.path.isfile(script):
            print(f"Script not found: {script}")
            continue

        print(f"Running script: {script}")
        try:
            # Run the script within the Conda environment
            result = subprocess.run(
                f"conda run -n {conda_env_name} python {script}",
                shell=True,
                capture_output=True,
                text=True,
                check=True,
            )
            print(f"Output of {script}:\n{result.stdout}")
            if result.stderr:
                print(f"Error output of {script}:\n{result.stderr}")
        except subprocess.CalledProcessError as e:
            print(f"Error while running {script}: {e}")
        except Exception as e:
            print(f"Unexpected error while running {script}: {e}")

# Main execution
if __name__ == "__main__":
    # Replace with your environment.yaml path and desired environment name
    env_yaml = "environment.yaml"
    conda_env = "my_conda_env"  # Name of the environment specified in the YAML file

    # List of scripts to execute in order
    scripts_to_run = [
        "create_folder_structure.py",
        "download_data.py",
        "hyperparameter_search_BCE.py",
        "Experiment1_no_removal.py",
        "Experiment1_with_removal.py",
        "Experiment2.py",
        "Experiment2_plot_curves.py"
    ]

    # Create the Conda environment and run scripts
    if create_conda_environment(env_yaml):
        run_python_scripts(scripts_to_run, conda_env)