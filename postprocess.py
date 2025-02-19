import numpy as np
import cv2

DMAX = 500
DMIN = 300

def centeroid(heat, gaussian_thresh = None):
    # Parse center point of connected components
    # Return [p][xy]
    
    heat = np.array(heat * 255., np.uint8)
    otsu_thresh, heat_bin = cv2.threshold(heat, 0, 1., cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # num: point number + 1 background
    num, labels = cv2.connectedComponents(heat_bin)
    thresh = otsu_thresh
    while num < 18: #background included
        thresh -= 0.05
        _, heat_bin = cv2.threshold(heat, thresh * 255, 1., cv2.THRESH_BINARY)
        num, labels = cv2.connectedComponents(heat_bin)
        if thresh < 0.1:
            break

    thresh = otsu_thresh
    while num < 18:
        thresh += 0.05
        _, heat_bin = cv2.threshold(heat, thresh * 255, 1., cv2.THRESH_BINARY)
        num, labels = cv2.connectedComponents(heat_bin)
        if thresh > 0.9:
            break

    return heat_bin

def find_corner_coords(heat_bin, center_coords, flag):

    num, labels = cv2.connectedComponents(heat_bin)

    coords = []
    for label in range(1, num):
        mask = np.zeros_like(labels, dtype=np.uint8)
        mask[labels == label] = 255
        M = cv2.moments(mask)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        coords.append([cX, cY])

    if len(center_coords) != 17:
        return coords

    final_coords = np.zeros((17, 2))

    if flag == "tr":
        for coord in coords:
            valid_center_coords = [center_coord for center_coord in center_coords if center_coord[0] < coord[0] and center_coord[1] > coord[1]]
            if len(valid_center_coords) == 0:
                continue
            valid_center = max(valid_center_coords, key=lambda x: x[1])

            id_valid = center_coords.index(valid_center)
            if final_coords[id_valid][0] == 0:
                final_coords[id_valid] = coord
            else: 
                if np.linalg.norm(np.array(final_coords[id_valid]) - np.array(center_coords[id_valid])) > np.linalg.norm(np.array(coord) - np.array(center_coords[id_valid])):
                    final_coords[id_valid] = coord

    elif flag == "tl":
        for coord in coords:
            valid_center_coords = [center_coord for center_coord in center_coords if center_coord[0] > coord[0] and center_coord[1] > coord[1]]
            if len(valid_center_coords) == 0:
                continue
            valid_center = max(valid_center_coords, key=lambda x: x[1])

            id_valid = center_coords.index(valid_center)
            if final_coords[id_valid][0] == 0:
                final_coords[id_valid] = coord
            else: 
                if np.linalg.norm(np.array(final_coords[id_valid]) - np.array(center_coords[id_valid])) > np.linalg.norm(np.array(coord) - np.array(center_coords[id_valid])):
                    final_coords[id_valid] = coord
    
    elif flag == "br":
        for coord in coords:
            valid_center_coords = [center_coord for center_coord in center_coords if center_coord[0] < coord[0] and center_coord[1] < coord[1]]
            if len(valid_center_coords) == 0:
                continue
            valid_center = max(valid_center_coords, key=lambda x: x[1])

            id_valid = center_coords.index(valid_center)
            if final_coords[id_valid][0] == 0:
                final_coords[id_valid] = coord
            else: 
                if np.linalg.norm(np.array(final_coords[id_valid]) - np.array(center_coords[id_valid])) > np.linalg.norm(np.array(coord) - np.array(center_coords[id_valid])):
                    final_coords[id_valid] = coord

    elif flag == "bl":
        for coord in coords:
            valid_center_coords = [center_coord for center_coord in center_coords if center_coord[0] > coord[0] and center_coord[1] < coord[1]]
            if len(valid_center_coords) == 0:
                continue
            valid_center = max(valid_center_coords, key=lambda x: x[1])

            id_valid = center_coords.index(valid_center)
            if final_coords[id_valid][0] == 0:
                final_coords[id_valid] = coord
            else: 
                if np.linalg.norm(np.array(final_coords[id_valid]) - np.array(center_coords[id_valid])) > np.linalg.norm(np.array(coord) - np.array(center_coords[id_valid])):
                    final_coords[id_valid] = coord
    
    else:
        raise ValueError("flag must be one of 'tr', 'tl', 'br', 'bl'")

    return final_coords

def find_center_coords(heat_bin, heat_top_bottom):

    num, labels = cv2.connectedComponents(heat_bin)

    coords = []
    for label in range(1, num):
        mask = np.zeros_like(labels, dtype=np.uint8)
        mask[labels == label] = 255
        M = cv2.moments(mask)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        coords.append([cX, cY])

    id_max = find_top_bottom(heat_bin, heat_top_bottom)
    id_max = sorted(id_max)
    coords_top_bottom = [coords[id_max[0] - 1], coords[id_max[1] - 1]]

    distance = np.linalg.norm(np.array(coords_top_bottom[0]) - np.array(coords_top_bottom[1]))
    if distance < DMAX and distance > DMIN:
        coords = [coord for coord in coords[id_max[0] - 1: id_max[1] - 1]]

    if len(coords) > 17:
        #check horizontal reparttion of the landmarks
        #if one is too far from the others, it is probably a false positive
        mean_x = np.mean([coord[0] for coord in coords])
        std_x = np.std([coord[0] for coord in coords])
        outliars = [(coord, abs(coord[0] - mean_x)) for coord in coords if abs(coord[0] - mean_x) > 2 * std_x]
        if len (outliars) > len(coords) - 17:
            outliars = sorted(outliars, key=lambda x: x[1], reverse=True)[:len(coords) - 17]
        coords = [coord for coord in coords if coord not in [outliar[0] for outliar in outliars]]

    return coords


def find_top_bottom(heatbin, heat_top_bottom):
    # Find the top and bottom points of the heatbin
    num, labels = cv2.connectedComponents(heatbin)
    maxs = [0, 0]
    id_max = [0, 0]
    for label in range(1, num):
        mask = np.zeros_like(labels, dtype=np.uint8)
        mask[labels == label] = 255
        top_bottom = heat_top_bottom * mask
        top_bottom = np.sum(top_bottom)
        if top_bottom > maxs[1] or top_bottom > maxs[0]:
            if maxs[0] > maxs[1]:
                maxs[1] = top_bottom
                id_max[1] = label
            else:
                maxs[0] = top_bottom
                id_max[0] = label
    
    return id_max
        

    
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
        ordered_corner = sorted(corner, key=lambda x: x[1])
        ordered_corners_split.append(ordered_corner)

    ordered_corners = []
    for i in range(len(ordered_corners_split[0])):
        ordered_corners.append(ordered_corners_split[0][i])
        ordered_corners.append(ordered_corners_split[1][i])
        ordered_corners.append(ordered_corners_split[2][i])
        ordered_corners.append(ordered_corners_split[3][i])

    return ordered_corners

#shape of images in scoliosis: (752, 256)
def resize_landmarks(landmarks, image_shape, original_shape = (512, 160)):
    """
    Resize the landmarks from the original image shape to the new image shape
    """
    ratio = image_shape[1] / original_shape[1] 
    shift = (image_shape[0] - original_shape[0] * ratio) / 2 
    new_landmarks = []
    for landmark in landmarks:
        new_landmarks.append([landmark[0] * ratio, landmark[1] * ratio + shift])
    return new_landmarks