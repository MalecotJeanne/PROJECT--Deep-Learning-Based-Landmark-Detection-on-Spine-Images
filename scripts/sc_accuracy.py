"""
Script to compute the accuracy between 2 sets of labels (list of coordinates)
"""
import os
import argparse
import pandas as pd
import sys
import numpy as np
from tqdm import tqdm
import cv2

root_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_folder_path)

parser = argparse.ArgumentParser(
    description="Script to convert the .mat landmarks files from a folder to .csv files. "
)

parser.add_argument(
    "-i", "--input", default="Ressources/boostnet_labeldata/data/test/", type=str, help="path to the images"
)
parser.add_argument(
    "-p", "--pred", default="landmarks/", type=str, help="path to the predicted landmarks folder"
)
parser.add_argument(
    "-gt", "--ground_truths", default="Ressources/boostnet_labeldata/labels/test_csv/", type=str, help="path to the ground truth landmarks folder"
)

args = parser.parse_args()

def split_corners(coords):
    corners = [[], [], [], []]
    current_corner = -1

    for coord in coords:
        if len(coord) == 1:
            current_corner += 1
        else:
            corners[current_corner].append(coord)

    #make sure all the corners have the same number of points, ie 17
    if not ((len(corners[0]) == len(corners[1]) and len(corners[2]) == len(corners[3])) and len(corners[0]) == len(corners[2])):
        return None 
    
    return corners

def order_corners(corners):
    ordered_corners_split = []
    for corner in corners:
        ordered_corner = sorted(corner, key=lambda x: x[0])
        ordered_corners_split.append(ordered_corner)

    ordered_corners = []
    for i in range(len(ordered_corners_split[0])):
        ordered_corners.append(ordered_corners_split[0][i])
        ordered_corners.append(ordered_corners_split[1][i])
        ordered_corners.append(ordered_corners_split[2][i])
        ordered_corners.append(ordered_corners_split[3][i])

    return ordered_corners

def resize_landmarks(landmarks, image_shape, original_shape = (752, 256)):
    """
    Resize the landmarks from the original image shape to the new image shape
    """
    ratio = image_shape[1] / original_shape[1] 
    shift = (image_shape[0] - original_shape[0] * ratio) / 2 
    new_landmarks = []
    for landmark in landmarks:
        new_landmarks.append([landmark[0] * ratio, landmark[1] * ratio + shift])
    return new_landmarks

def main():
    
    preds_dir = args.pred
    gt_dir = args.ground_truths

    preds = []
    gtruths = []

    for f in os.listdir(preds_dir):
        content = []
        if f.endswith(".txt"):
            name = f.split(".")[0]
            with open(os.path.join(preds_dir, f), 'r') as file:
                lines = file.readlines()
                #convert the lines to a list of coordinates
                for line in lines:
                    content.append([int(x) for x in line.split()])
            preds.append({'name': name, 'landmarks': content})

    correct_preds = []
    for pred in preds:
        corners = split_corners(pred['landmarks'])
        if corners is None:
            continue
        ordered_corners = np.array(order_corners(corners))
        correct_preds.append({'name': pred['name'], 'landmarks': ordered_corners})

    print("Number of dropped predictions: ", len(preds) - len(correct_preds))
    
    for f in os.listdir(gt_dir):
        name = f.split(".")[0]
        if f.endswith(".csv"):
            gtruths.append({'name': name, 'landmarks': np.array(pd.read_csv(os.path.join(gt_dir, f), header=None).values)})

    #fuse the 2 lists
    samples = []
    for gtruth in gtruths:    
        for pred in correct_preds:
            if pred['name'] == gtruth['name']:
                samples.append({'name': pred['name'], 'pred': pred['landmarks'], 'gt': gtruth['landmarks']})

    
    #read images
    images_dir = args.input
    images = []
    for f in os.listdir(images_dir):
        name = f.split(".")[0]
        if f.endswith(".jpg"):
            #load image
            image = cv2.imread(os.path.join(images_dir, f))
            images.append({'name': name, 'image': image})

    #add image in the sample
    for sample in samples:
        for image in images:
            if sample['name'] == image['name']:
                sample['image'] = image['image']

    save_dir = "temp"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #draw the landmarks on the images
    for sample in samples:
        image = sample['image']
        name = sample['name']
        landmarks = resize_landmarks(sample['pred'], image.shape)
        for landmark in landmarks:
            cv2.circle(
                image, (int(landmark[0]), int(landmark[1])), 4, (0, 255, 0), -4
            )
        for landmark in sample['gt']:
            cv2.circle(
                image, (int(landmark[0]), int(landmark[1])), 4, (0, 0, 255), -4
            )
        save_path = os.path.join(save_dir, f'{name}.jpg')
        cv2.imwrite(save_path, image)


    #compute the accuracy
if __name__ == "__main__":
    main()