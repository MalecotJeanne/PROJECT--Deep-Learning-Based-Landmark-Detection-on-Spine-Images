"""
Script to create bounding box from landmarks
The script takes as input the path to the landmarks .csv files and the path to save the bounding boxes
The boudning boxes will be saved in a .csv file, named as the corresponding image, with the following format:
Each line corresponds to a box, with : center_x, center_y, angle, width, height
Author: Jeanne Malécot
"""

import argparse
import os
import csv
import math

from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Script to create bounding box from landmarks."
)

parser.add_argument(
    "--landmarks_dir", default="landmarks/", type=str, help="landmarks path"
)
parser.add_argument(
    "--save_dir", default="bounding_boxes/", type=str, help="path to save the bounding boxes"
)

args = parser.parse_args()

def main():
    landmarks_dir = args.landmarks_dir
    save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for filename in tqdm(os.listdir(landmarks_dir)):
        if filename.endswith(".csv"):
            input_file = os.path.join(landmarks_dir, filename)
            output_file = os.path.join(save_dir, filename)

            with open(input_file, mode="r") as file:
                reader = csv.reader(file)
                landmarks = [(float(row[0]), float(row[1])) for row in reader]
            
            if len(landmarks) % 4 != 0:
                raise ValueError(f"File {input_file} has an invalid number of landmarks (not a multiple of 4).")
            
            with open(output_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                for i in range(0, len(landmarks), 4):
                    box = landmarks[i:i+2] + [landmarks[i+3], landmarks[i+2]] # clockwise
                    center, angle, width, height = make_rectangle(box)
                    # create the bounding box
                    # new_box = [
                    #     (center[0] + width / 2 * math.cos(angle) - height / 2 * math.sin(angle), center[1] + width / 2 * math.sin(angle) + height / 2 * math.cos(angle)),
                    #     (center[0] - width / 2 * math.cos(angle) - height / 2 * math.sin(angle), center[1] - width / 2 * math.sin(angle) + height / 2 * math.cos(angle)),
                    #     (center[0] - width / 2 * math.cos(angle) + height / 2 * math.sin(angle), center[1] - width / 2 * math.sin(angle) - height / 2 * math.cos(angle)),
                    #     (center[0] + width / 2 * math.cos(angle) + height / 2 * math.sin(angle), center[1] + width / 2 * math.sin(angle) - height / 2 * math.cos(angle))
                    # ]
                    # row = [coord for pair in new_box for coord in pair] 
                    row = [center[0], center[1], angle, width, height]
                    writer.writerow(row)

def make_rectangle (box):
    """
    Function to convert a list of 4 points into a rectangle (eventually rotated)
    Input: 
    box: list of 4 points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] (clockwise)
    Output:
    center: tuple (x, y) of the center of the rectangle
    angle: float of the angle of the rectangle
    width: float of the width of the rectangle
    height: float of the height of the rectangle
    """
    # get the center of the box
    x = [coord[0] for coord in box]
    y = [coord[1] for coord in box]
    center = (sum(x) / 4, sum(y) / 4)

    # get the angle of the box
    x14 = (box[3][0] + box[0][0]) / 2
    y14 = (box[3][1] + box[0][1]) / 2
    x23 = (box[2][0] + box[1][0]) / 2
    y23 = (box[2][1] + box[1][1]) / 2
    angle = math.atan2(y23 - y14, x23 - x14)

    # get the width and height of the box
    width = max(x) - min(x)
    height = max(y) - min(y)
    #apply a coefficient to the width and height to make the bounding box larger
    width *= 1.5
    height *= 1.5

    return center, angle, width, height

if __name__ == "__main__":
    main()
