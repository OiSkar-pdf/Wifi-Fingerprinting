import numpy as np
import os
import sigmf
import pandas as pd
import shutil  # Add this import

folder_dir = "/common/users/oi66/Wifi-Fingerprinting/KRI-16Devices-RawData"
new_folder_dir = "/common/users/oi66/Wifi-Fingerprinting/KRI_16Devices_RawData"

def create_new_folder(folder_dir):
    label_map = []
    subfolder = os.path.join(folder_dir, "2ft")
    for file in os.listdir(subfolder):
        name = file.split('_')
        folder_name = name[2] + "_" + name[3]
        label_map.append(folder_name)
        if not os.path.exists(os.path.join(new_folder_dir, folder_name)):
            os.makedirs(os.path.join(new_folder_dir, folder_name), exist_ok=True)

def reformat_data(folder_dir, new_folder_dir):
    for folder in os.listdir(folder_dir):
        src_folder = os.path.join(folder_dir, folder)
        for file in os.listdir(src_folder):
            name = file.split('_')
            folder_name = name[2] + "_" + name[3]
            dst_folder = os.path.join(new_folder_dir, folder_name)
            os.makedirs(dst_folder, exist_ok=True)
            src_file = os.path.join(src_folder, file)
            dst_file = os.path.join(dst_folder, file)
            shutil.move(src_file, dst_file)  # Move file to new location

if __name__ == "__main__":
    create_new_folder(folder_dir)
    reformat_data(folder_dir, new_folder_dir)
    print("Data reformatted successfully.")