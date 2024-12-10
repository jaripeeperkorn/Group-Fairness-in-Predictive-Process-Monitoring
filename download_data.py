import os
import requests

def download_files_from_zenodo(record_id, output_folder):
    """
    Downloads all files from a Zenodo record using its API.

    :param record_id: The record ID of the Zenodo dataset
    :param output_folder: The folder where the downloaded files will be saved
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Construct the API URL for the given record
    api_url = f"https://zenodo.org/api/records/{record_id}"

    try:
        # Fetch record metadata from the API
        response = requests.get(api_url)
        response.raise_for_status()  # Raise an error for HTTP requests that failed
        data = response.json()

        # Extract file information
        files = data.get("files", [])
        if not files:
            print("No files found in the Zenodo record.")
            return

        print(f"Found {len(files)} files to download.")

        # Download each file
        for file_info in files:
            file_name = file_info["key"]
            file_url = file_info["links"]["self"]
            file_path = os.path.join(output_folder, file_name)

            print(f"Downloading {file_name}...")
            file_response = requests.get(file_url)
            file_response.raise_for_status()  # Raise error for failed requests

            # Save the file to the output folder
            with open(file_path, 'wb') as f:
                f.write(file_response.content)
            print(f"Saved {file_name} to {output_folder}.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
zenodo_record_id = "8059489"  # Replace with the record ID from the Zenodo URL
output_directory = "Datasets"

download_files_from_zenodo(zenodo_record_id, output_directory)