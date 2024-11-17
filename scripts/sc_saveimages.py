"""
Script to save images with the corresponding landmarks and/or bounding boxes, for visualization purposes.
The script takes as input the path to the images, the path to the landmarks and bounding boxes (if needed) and the path to save the images.
The landmarks and bounding boxes should be saved in .csv files, named as the corresponding image.
Author: Jeanne Malécot
"""

import argparse
import os
import cv2
import numpy as np
import math

from tqdm import tqdm


def save_images(
    image_dir,
    save_dir,
    landmarks_dir=None,
    bounding_boxes_dir=None,
    load_image=True,
):
    """
    Function to save images with the corresponding landmarks and/or bounding boxes, for visualization purposes.
    Args:
        image_dir (str): path to the images, or list of images
        save_dir (str): path to save the images
        landmarks_dir (str): path to the landmarks (default: None)
        bounding_boxes_dir (str): path to the bounding boxes (default: None)
        load_image (bool): whether to load the images from the image_dir (in the case it is a path) or not (default: True)
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n_images = len(os.listdir(image_dir)) if load_image else len(image_dir)

    for i in tqdm(n_images):
        if load_image:
            image_name = os.listdir(image_dir)[i]
            image_path = os.path.join(image_dir, image_name)
            image = cv2.imread(image_path)
        else:
            image = image_dir[i]

        if landmarks_dir:
            landmarks_name = image_name.split(".")[0] + ".csv"
            landmarks_path = os.path.join(landmarks_dir, landmarks_name)
            landmarks = np.loadtxt(landmarks_path, delimiter=",")
            for landmark in landmarks:
                cv2.circle(
                    image, (int(landmark[0]), int(landmark[1])), 4, (0, 255, 0), -4
                )

        save_path = os.path.join(save_dir, image_name)
        cv2.imwrite(save_path, image)

        if bounding_boxes_dir:
            bounding_boxes_name = image_name.split(".")[0] + ".csv"
            bounding_boxes_path = os.path.join(bounding_boxes_dir, bounding_boxes_name)
            bounding_boxes = np.loadtxt(bounding_boxes_path, delimiter=",")
            for i in range(0, len(bounding_boxes)):
                center_x, center_y, angle, width, height = bounding_boxes[i]
                box = [
                    (
                        center_x
                        + width / 2 * math.cos(angle)
                        - height / 2 * math.sin(angle),
                        center_y
                        + width / 2 * math.sin(angle)
                        + height / 2 * math.cos(angle),
                    ),
                    (
                        center_x
                        - width / 2 * math.cos(angle)
                        - height / 2 * math.sin(angle),
                        center_y
                        - width / 2 * math.sin(angle)
                        + height / 2 * math.cos(angle),
                    ),
                    (
                        center_x
                        - width / 2 * math.cos(angle)
                        + height / 2 * math.sin(angle),
                        center_y
                        - width / 2 * math.sin(angle)
                        - height / 2 * math.cos(angle),
                    ),
                    (
                        center_x
                        + width / 2 * math.cos(angle)
                        + height / 2 * math.sin(angle),
                        center_y
                        + width / 2 * math.sin(angle)
                        - height / 2 * math.cos(angle),
                    ),
                ]
                for j in range(4):
                    cv2.line(
                        image,
                        (int(box[j][0]), int(box[j][1])),
                        (int(box[(j + 1) % 4][0]), int(box[(j + 1) % 4][1])),
                        (0, 0, 255),
                        2,
                    )

            save_path = os.path.join(save_dir, image_name)
            cv2.imwrite(save_path, image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to save images with the corresponding landmarks (optional), for visualization purposes."
    )

    parser.add_argument("-image_dir", default="images/", type=str, help="images path")
    parser.add_argument(
        "--landmarks_dir", default=None, type=str, help="landmarks path"
    )
    parser.add_argument(
        "--bb_dir",
        default=None,
        type=str,
        help="bounding boxes path",
    )
    parser.add_argument(
        "--save_dir", default="images/", type=str, help="path to save the images"
    )

    args = parser.parse_args()
    save_images(args.image_dir, args.save_dir, args.landmarks_dir, args.bb_dir)
