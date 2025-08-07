import os
import subprocess

config_folder = "configs/configs_by_category/filtered_data_cartnet"
for config_file in os.listdir(config_folder):
    config_path = os.path.join(config_folder, config_file)
    print(f"Running experiment with {config_file}")
    subprocess.run(["python", "main.py", "--cfg", config_path])
