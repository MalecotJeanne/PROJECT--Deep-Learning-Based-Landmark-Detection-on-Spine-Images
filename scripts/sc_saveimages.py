"""
Script to save images with the corresponding landmarks, for visualization purposes.
The script takes as input the path to the images, the path to the landmarks (if needed) and the path to save the images.
The landmarks should be saved in a .csv file, named as the corresponding image, with the following format:
x1, y1
x2, y2
...
xn, yn
Author: Jeanne Malécot
"""

import argparse
import os
import cv2
import numpy as np

from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Script to save images with the corresponding landmarks (optional), for visualization purposes."
)

parser.add_argument(
    "-image_dir", default="images/", type=str, help="images path"
)
parser.add_argument(
    "--landmarks_dir", default="landmarks/", type=str, help="landmarks path"
)
parser.add_argument(
    "--save_dir", default="images/", type=str, help="path to save the images"
)
args = parser.parse_args()


def main():
    image_dir = args.image_dir
    landmarks_dir = args.landmarks_dir
    save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for image_name in tqdm(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)

        if landmarks_dir:   
            landmarks_name = image_name.split(".")[0] + ".csv"
            landmarks_path = os.path.join(landmarks_dir, landmarks_name)
            landmarks = np.loadtxt(landmarks_path, delimiter=",")
            for landmark in landmarks:
                cv2.circle(image, (int(landmark[0]), int(landmark[1])), 4, (0, 255, 0), -4)

        save_path = os.path.join(save_dir, image_name)
        cv2.imwrite(save_path, image)


if __name__ == "__main__":
    main()