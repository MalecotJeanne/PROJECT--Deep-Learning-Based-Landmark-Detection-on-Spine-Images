"""
Script to convert the .mat landmarks files from a folder to .csv files. 
Each .csv file will have the following format:
x1, y1
x2, y2
...
xn, yn
Author: Jeanne Malécot  
"""
import os
import argparse
import pandas as pd
import sys
from tqdm import tqdm

root_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_folder_path)

from utils import load_labels_mat

parser = argparse.ArgumentParser(
    description="Script to convert the .mat landmarks files from a folder to .csv files. "
)

parser.add_argument(
    "--mat_dir", default="landmarks/", type=str, help="path to the .mat files folder"
)
parser.add_argument(
    "--csv_dir", default="landmarks_csv/", type=str, help="path to save the .csv files"
)

args = parser.parse_args()

def main():
    # Load the .mat files
    mat_dir = args.mat_dir
    csv_dir = args.csv_dir

    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    # select only the .mat files
    labels_mat = [
        os.path.join(mat_dir, f) for f in os.listdir(mat_dir) if f.endswith(".mat")
    ]
    labels = load_labels_mat(labels_mat)

    print(f"Found {len(labels)} .mat files")

    for i, label in enumerate(tqdm(labels)):
        df = pd.DataFrame(label)
        name_mat = labels_mat[i].split("\\")[-1]
        name_csv = name_mat.split(".")[0] + ".csv"
        df.to_csv(os.path.join(csv_dir, name_csv), index=False, header=False)

if __name__ == "__main__":
    main()