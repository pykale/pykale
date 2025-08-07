import os
import subprocess

# Define the base folder paths
base_checkpoint_folder = "saved_models/cartnet"
base_test_data_folder = "data/data_by_type"
config_folder = "configs/configs_by_category/filtered_data_cartnet"

# Iterate through the configuration files in the config folder
for config_file in os.listdir(config_folder):
    # Ensure we only process YAML files
    if not config_file.endswith(".yaml"):
        continue

    # Extract the category from the config file name
    category = config_file.split("config_")[1].split(".yaml")[0]

    # Build paths for the checkpoint and test data
    checkpoint_folder = os.path.join(base_checkpoint_folder, category)
    test_data_folder = os.path.join(base_test_data_folder, f"data_{category}")
    test_data_file = os.path.join(test_data_folder, "test_data.json")

    # Ensure the checkpoint folder exists
    if not os.path.exists(checkpoint_folder):
        raise FileNotFoundError(f"Checkpoint folder not found: {checkpoint_folder}")

    # Locate all .ckpt files with 'best-mae' in the checkpoint folder
    checkpoint_files = [f for f in os.listdir(checkpoint_folder) if f.endswith(".ckpt") and "best-mre" in f]
    if len(checkpoint_files) == 0:
        raise ValueError(f"No .ckpt files with 'best-mre' found in {checkpoint_folder}")

    # Ensure the test data file exists
    if not os.path.exists(test_data_file):
        raise FileNotFoundError(f"Test data file not found: {test_data_file}")

    # Build the config path
    config_path = os.path.join(config_folder, config_file)

    # Print the experiment details
    print(f"Running experiment for category: {category}")
    print(f"Using config file: {config_file}")
    print(f"Using test data: {test_data_file}")

    # Iterate over each checkpoint file and run the experiment
    for idx, checkpoint_file in enumerate(checkpoint_files, start=1):
        checkpoint_path = os.path.join(checkpoint_folder, checkpoint_file)
        output_file = f"predictions_{idx}.csv"

        print(f"Using checkpoint: {checkpoint_path}")
        print(f"Generating output file: {output_file}")

        # Run the subprocess with the appropriate paths
        subprocess.run([
            "python", "test_model.py",
            "--cfg", config_path,
            "--checkpoint", checkpoint_path,
            "--test_data", test_data_file,
            "--cif_folder", "cif_file",
            "--output_dir", f"predictions/cartnet/{category}",
            "--output_file", output_file,
        ])

