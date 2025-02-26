import numpy as np
import cv2
import torch

class Landmarks:
    def __init__(self, n_landmarks = 17):
        self.landmarks = np.zeros((n_landmarks * 4, 2))
        self.landmarks_loaded = False
        self.landmarks_binheat = None
        self.center_coords = np.zeros((n_landmarks, 2))
        self.center_coords_loaded = False
        self.center_binheat = None
        self.top_center = np.zeros((2, 2))
        self.top_center_loaded = False
        self.top_center_binheat = None
        self.heatmaps = None
        self.n_landmarks = n_landmarks

    def __call__(self, center = False, top_center = False):
        if not self.landmarks_loaded:
            raise ValueError("Landmarks not loaded")
        if center and not self.center_coords_loaded:
            raise ValueError("Center coordinates not loaded")
        if top_center and not self.top_center_loaded:
            raise ValueError("Top and bottom coordinates not loaded")

        if center and top_center:
            return self.landmarks, self.center_coords, self.top_center
        elif center:
            return self.landmarks, self.center_coords

        return self.landmarks

    def get_bin_heatmaps(self, center = True, top_center = True):
        if not self.landmarks_loaded:
            raise ValueError("Landmarks not loaded")
        if center and not self.center_coords_loaded:
            raise ValueError("Center coordinates not loaded")
        if top_center and not self.top_center_loaded:
            raise ValueError("Top and bottom coordinates not loaded")
        
        if center and top_center:
            return self.landmarks_binheat, self.center_binheat, self.top_center_binheat
        
        if center:
            return self.landmarks_binheat, self.center_binheat
        
        return self.landmarks_binheat        

    def load_from_array(self, landmarks_array, centers = True, top_center = True):
        """
        Load the landmarks from an array
        Args:
            landmarks_array (np.array): Array containing the landmarks (68x2)
            centers (Bool): If True, compute the center coordinates
            top_center (Bool): If True, find the top center point
        """
        if landmarks_array.shape != (self.n_landmarks, 2):
            raise ValueError(f"Landmarks array must be of shape ({self.n_landmarks * 4}, 2)")

        if isinstance(landmarks_array, list):
            landmarks_array = np.array(landmarks_array)
        if isinstance(landmarks_array, torch.Tensor):
            landmarks_array = landmarks_array.cpu().detach().numpy()
            landmarks_array = np.array(landmarks_array)

        self.landmarks = landmarks_array
        self.landmarks_loaded = True

        if centers:
            self.center_coords = self._center_coords()
            self.center_coords_loaded = True

        if top_center:
            self.top_center = self._top_coords()
            self.top_center_loaded = True

        self.landmarks = self._split_corners(landmarks_array)

    def load_from_heatmap(self, heatmaps):
        """
        Load the landmarks from the heatmaps
        Args:
            heatmaps (np.array): Array containing the heatmaps
        """

        if isinstance(heatmaps, torch.Tensor):
            heatmaps = heatmaps.cpu().detach().numpy()
            heatmaps = np.array(heatmaps)  
        
        self.heatmaps = heatmaps

        n_channels = heatmaps.shape[0]
        centers = False
        top_center = False
        if n_channels == 5:
            centers = True
        elif n_channels == 6:
            centers = True
            top_center = True
        elif n_channels != 4:
            raise ValueError("Invalid number of channels in heatmaps")
        
        if top_center:
            self.top_center, self.top_center_binheat = self._top_centroids()
            self.top_center_loaded = True

        if centers:
            self.center_coords, self.center_binheat = self._center_centroids()
            self.center_coords_loaded = True

        self.landmarks, self.landmarks_binheat = self._corners_centroids()
        self.landmarks_loaded = True

    # ----- Private methods -----

    # Coordinates
    def _split_corners(self, landmarks_array):
        corners = [[], [], [], []]
        for i, coords in enumerate(landmarks_array):
            if i % 4 == 0:
                corners[0].append(list(coords))
            elif i % 4 == 1:
                corners[1].append(list(coords))
            elif i % 4 == 2:
                corners[2].append(list(coords))
            else:
                corners[3].append(list(coords))

        return corners

    def _center_coords(self):
        center_coords = np.zeros((self.n_landmarks, 2))
        for i in range(0, self.n_landmarks*4, 4):
            y_center = int((self.landmarks[i][0] + self.landmarks[i+1][0] + self.landmarks[i+2][0] + self.landmarks[i+3][0]) / 4)
            x_center = int((self.landmarks[i][1] + self.landmarks[i+1][1] + self.landmarks[i+2][1] + self.landmarks[i+3][1]) / 4)
            center_coords[i//4] = [y_center, x_center]
        return center_coords
    
    def _top_coords(self):
        top_center = np.zeros((1, 2))
        if self.center_coords_loaded:
            top = min(self.center_coords, key=lambda x: x[1])
            top_center = [top]
        else:
            center_coords = self._center_coords()
            top = min(center_coords, key=lambda x: x[1])
            top_center = [top]
        return np.array(top_center)
    
    # Heatmaps

    def _best_thresh(self, heat, n_zones):
        
        otsu_thresh, heat_bin = cv2.threshold(heat, 0, 1., cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        num, _ = cv2.connectedComponents(heat_bin)

        thresh = otsu_thresh
        while num < n_zones:
            thresh -= 1
            _, heat_bin = cv2.threshold(heat, thresh, 1., cv2.THRESH_BINARY)
            num, _ = cv2.connectedComponents(heat_bin)
            if thresh < 50:
                break
        
        thresh = otsu_thresh
        while num < n_zones:
            thresh += 1
            _, heat_bin = cv2.threshold(heat, thresh, 1., cv2.THRESH_BINARY)
            num, _ = cv2.connectedComponents(heat_bin)
            if thresh > 200:
                break
        
        if num < n_zones:
            thresh = otsu_thresh

        return thresh
    
    def _corners_centroids(self):   
        heats_list = []
        coords_list = []
        for i in range(4):
            heat = self.heatmaps[i]
            heat = np.array(heat * 255., np.uint8)

            thresh = self._best_thresh(heat, self.n_landmarks +1)
            _, heat_bin = cv2.threshold(heat, thresh, 1., cv2.THRESH_BINARY)

            num, labels = cv2.connectedComponents(heat_bin)

            coords = []
            for label in range(1, num):
                mask = np.zeros_like(labels, dtype=np.uint8)
                mask[labels == label] = 255
                M = cv2.moments(mask)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                coords.append([cX, cY])
            
            if not (self.center_coords_loaded and len(self.center_coords) == self.n_landmarks):
                coords = self._post_process_coords(coords)

            else:
                coords = self._find_from_center(coords, i)

            coords_list.append(coords)
            heats_list.append(heat_bin)

        return coords_list, heats_list
    
    def _center_centroids(self):

        heat = self.heatmaps[4]
        heat = np.array(heat * 255., np.uint8)

        thresh = self._best_thresh(heat, self.n_landmarks + 1)

        _, heat_bin = cv2.threshold(heat, thresh, 1., cv2.THRESH_BINARY)

        num, labels = cv2.connectedComponents(heat_bin)
        coords = []
        for label in range(1, num):
            mask = np.zeros_like(labels, dtype=np.uint8)
            mask[labels == label] = 255
            M = cv2.moments(mask)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            coords.append([cX, cY])

        coords = self._post_process_coords(coords)
        return np.array(coords), heat_bin
            
    def _top_centroids(self):

        heat = self.heatmaps[5]
        heat = np.array(heat * 255., np.uint8)
        thresh = self._best_thresh(heat, 2)
        _, heat_bin = cv2.threshold(heat, thresh, 1., cv2.THRESH_BINARY)

        num, labels = cv2.connectedComponents(heat_bin)

        if num > 2:
            intensities = np.zeros(num)
            for i in range(1, num):  
                component_mask = (labels == i)
                intensities[i] = np.max(heat[component_mask])

            max_intensity_label = np.argmax(intensities)

            modified_mask = np.zeros_like(labels, dtype=np.uint8)
            modified_mask[labels == max_intensity_label] = 255

            M = cv2.moments(modified_mask)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            coords = [[cX, cY]]
            
        elif num == 2:
            mask = np.zeros_like(labels, dtype=np.uint8)
            mask[labels == 1] = 255
            M = cv2.moments(mask)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            coords = [[cX, cY]]

        else:
            coords = np.unravel_index(np.argmax(heat), heat.shape)
            coords = [[coords[1], coords[0]]]

        return np.array(coords), heat_bin

    def _find_from_center(self, coords, i):
        # i values: 0 = tl, 1 = tr, 2 = bl, 3 = br

        good_coords = -np.ones((self.n_landmarks, 2)) 
        for center in self.center_coords:
            if i == 0:
                valid_coords = [coord for coord in coords if coord[0] < center[0] and coord[1] < center[1]]
            elif i == 1:
                valid_coords = [coord for coord in coords if coord[0] > center[0] and coord[1] < center[1]]
            elif i == 2:
                valid_coords = [coord for coord in coords if coord[0] < center[0] and coord[1] > center[1]]
            elif i == 3:
                valid_coords = [coord for coord in coords if coord[0] > center[0] and coord[1] > center[1]]

            if len(valid_coords) == 0:
                continue
            
            if i == 0 or i == 1:
                valid_coord = max(valid_coords, key=lambda x: x[1])
            else:
                valid_coord = min(valid_coords, key=lambda x: x[1])
            good_coords[np.where((self.center_coords == center).all(axis=1))[0][0]] = valid_coord
            
        #filter duplicates (removing if the same point is assigned to two different centers)
        #find duplicates in good_coords
        duplicates = []
        for i in range(self.n_landmarks - 1):
            if any(np.array_equal(good_coords[i], coord) for coord in good_coords[i+1:]):
                duplicates.append(i)
        #remove duplicates
        for i in duplicates:
            coords_dup = good_coords[i]
            #find the point with coords_dup beiing the closest to its center
            indx_dup = [j for j, x in enumerate(good_coords) if np.array_equal(x, coords_dup)]
            #find the center closest to the point
            distances = [np.linalg.norm(np.array(self.center_coords[j]) - np.array(coords_dup)) for j in indx_dup]
            good_idx = indx_dup[np.argmin(distances)]
            #remove the other points
            for j in indx_dup:
                if j != good_idx:
                    good_coords[j] = -np.ones(2)
        
        return np.array(good_coords)

    def _post_process_coords(self, coords):
        #remove horizontal outliars
        mean_x = np.mean([coord[0] for coord in coords])
        std_x = np.std([coord[0] for coord in coords])
        outliars = [(coord, abs(coord[0] - mean_x)) for coord in coords if abs(coord[0] - mean_x) > 2 * std_x]
        if len (outliars) > len(coords) - self.n_landmarks:
            outliars = sorted(outliars, key=lambda x: x[1], reverse=True)[:len(coords) - self.n_landmarks]
        coords = [coord for coord in coords if coord not in [outliar[0] for outliar in outliars]]

        if len(coords) < self.n_landmarks:
            #check space between landmarks and eventually extrapolate the missing ones
            distances = []
            for i in range(len(coords) - 1):
                distances.append(np.linalg.norm(np.array(coords[i]) - np.array(coords[i+1])))
            mean_distance = np.mean(distances)
            std_distance = np.std(distances)
            new_coords = []
            for i in range(len(distances)):
                new_coords.append(coords[i])
                if distances[i] > mean_distance + 2 * std_distance:
                    new_coord = np.array(coords[i]) + (np.array(coords[i+1]) - np.array(coords[i])) * mean_distance / distances[i]
                    new_coords.append(new_coord) 
            new_coords.append(coords[-1])

            coords = new_coords

        if len(coords) > self.n_landmarks:
            #remove the highest or lowest coordinates
            if self.top_center_loaded:

                if self.n_landmarks == 17:
                    top_center = self.top_center

                    #check if the top point is not low high
                    if top_center[0][1] > 150:
                        top_value = 0
                    else:
                        top_value = top_center[0][1] - 10 #value 10 chosen according to the average width of a vertebrae (~30 +- 10)
                else:
                    top_center = self.top_center
                    top_value = top_center[0][1] - 10
            else:
                top_value = 0

            #select the 17 firsts coords below the top point
            coords = [coord for coord in coords if coord[1] > top_value][:self.n_landmarks]

        return coords
    
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