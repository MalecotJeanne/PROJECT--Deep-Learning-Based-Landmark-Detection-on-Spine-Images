import os
import torch
from loguru import logger
import numpy as np
from os import path
from math import floor

from monai.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import save_dataset, show_results, show_final
from metrics import pick_accuracy
from models.utils import ld2hm
from transforms import testing_transforms
from postprocess import Landmarks


def test_model(dataset, model, chkpt_dir, results_dir, config, device, log_path):
    """
    Test the model using a provided test dataset.
    Args:
        dataset (Dataset): The dataset to be used for testing.
        model (torch.nn.Module): The trained model to be evaluated.
        chkpt_dir (str): Directory to load the best model checkpoint.
        results_dir (str): Directory to save testing results.
        config (dict): Configuration dictionary containing testing parameters.
        device (torch.device): The device to run the testing on (CPU or GPU).
        log_path (str): Path to the log file for logging testing progress.
    """
    # logger.add(log_path, mode="a")

    # Load the model from the best checkpoint
    chkpt_path = os.path.join(chkpt_dir, "best_val_loss.pt")
    if not os.path.exists(chkpt_path):
        logger.error(f"Checkpoint {chkpt_path} not found.")
        return 

    logger.info(f"Loading model from checkpoint: {chkpt_path}")
    checkpoint = torch.load(chkpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval() 

    accuracy_name = config["train"]["accuracy"]
    accuracy, supported_accuracy = pick_accuracy(accuracy_name)
    if not supported_accuracy:
        logger.warning(
            f"Accuracy metric {accuracy_name} is not supported. Using PCK instead."
        )

    distance, _ = pick_accuracy("nme")

    # Create the data loader
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    test_accuracy = 0.0
    test_accuracy_lumbar = 0.0
    test_accuracy_thoracic = 0.0

    logger.info("Starting testing...")
    with open(log_path, "a") as log_file:
        log_file.write(
            f"====================\nStarting testing...\n====================\n"
        )

    output_dataset = []
    final_coords = []
    accuracies = []
    lumbar_accuracies = []
    thoracic_accuracies = []
    distances = []
    lumbar_distances = []
    thoracic_distances = []

    with torch.no_grad():

        for batch in tqdm(test_loader, total=len(test_loader), desc="Testing"):

            inputs, landmarks = batch["image"], batch["landmarks"]
            inputs, landmarks = inputs.to(device), landmarks.to(device)

            image_plot = inputs[0].detach().cpu().numpy()

            gt_heatmaps = ld2hm(landmarks, inputs.shape[-2:], 6, config['train']['gt_heatmap'], device = device).cpu().numpy()[0]

            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)

            image_name = batch["image_meta_dict"]["name"][-1]

            np_heats = outputs.cpu().numpy()
            np_heats = np.clip(np_heats, 0., 1.)[0]  

            GTLandmarks = Landmarks()
            GTLandmarks.load_from_array(landmarks.cpu().numpy()[0])

            gt_corners, gt_centers, gt_top = GTLandmarks(center=True, top_center=True)

            if len(np_heats) == 12:
                np_heats_thoracic = np_heats[:6]
                np_heats_lumbar = np_heats[6:]

                PredLandmarksThoracic = Landmarks(n_landmarks =12)
                PredLandmarksThoracic.load_from_heatmap(np_heats_thoracic)

                pred_corners_thoracic, pred_centers_thoracic, pred_top_thoracic = PredLandmarksThoracic(center=True, top_center=True)
                corners_bin_heat_thoracic , center_bin_heat_thoracic, top_bin_heat_thoracic = PredLandmarksThoracic.get_bin_heatmaps()


                PredLandmarksLumbar = Landmarks(n_landmarks = 5)
                PredLandmarksLumbar.load_from_heatmap(np_heats_lumbar)
                
                pred_corners_lumbar, pred_centers_lumbar, pred_top_lumbar = PredLandmarksLumbar(center=True, top_center=True)
                corners_bin_heat_lumbar , center_bin_heat_lumbar, top_bin_heat_lumbar = PredLandmarksLumbar.get_bin_heatmaps()
                
                pred_corners = pred_corners_thoracic + pred_corners_lumbar
                pred_centers = pred_centers_thoracic + pred_centers_lumbar
                pred_top = pred_top_thoracic + pred_top_lumbar

                corners_bin_heat = corners_bin_heat_thoracic + corners_bin_heat_lumbar
                center_bin_heat = center_bin_heat_thoracic + center_bin_heat_lumbar
                top_bin_heat = top_bin_heat_thoracic + top_bin_heat_lumbar

            else:
                PredLandmarks = Landmarks()
                PredLandmarks.load_from_heatmap(np_heats)

                pred_corners, pred_centers, pred_top = PredLandmarks(center=True, top_center=True)

                corners_bin_heat , center_bin_heat, top_bin_heat = PredLandmarks.get_bin_heatmaps()

            accuracy_centers = accuracy(pred_centers, gt_centers) if len(pred_centers) == 17 else 0
            accuracy_top = accuracy(pred_top, gt_top) if len(pred_top) == 2 else 0

            if len(pred_top) > 1:
                top_line = [pred_top[0][1], pred_top[1][1]]
            else:
                top_line = [pred_top[0][1]]
                if top_line[0] > 150:
                    top_line = [0]
            
            #prepare file to save a figure with both heatmaps (bin and not bin) 
            heatmap_dir = path.join(results_dir, "testing_heatmaps", image_name)
            invalid_heatmap_dir = path.join(results_dir, "testing_heatmaps", "invalid", image_name)  
            titles = ["Top Left", "Top Right", "Bottom Left", "Bottom Right"]          

            #if the nb of landmarks is incorrect, save the heatmaps and pass to the next image
            if len(pred_corners[0]) != 17 or len(pred_corners[1]) != 17 or len(pred_corners[2]) != 17 or len(pred_corners[3]) != 17 :
                os.makedirs(invalid_heatmap_dir, exist_ok=True) 
                show_results("Center points", center_bin_heat, np_heats[4], gt_heatmaps[4], gt_centers, image_plot, pred_centers, accuracy_centers, invalid_heatmap_dir, top_line = top_line)
                show_results("Top points", top_bin_heat, np_heats[5], gt_heatmaps[5], gt_top, image_plot, pred_top, accuracy_top, invalid_heatmap_dir)

                for i in range (4):
                    show_results(titles[i], corners_bin_heat[i], np_heats[i], gt_heatmaps[i], gt_corners[i], image_plot, pred_corners[i], 0, invalid_heatmap_dir)
                continue
             
            os.makedirs(heatmap_dir, exist_ok=True)
            show_results("Center points", center_bin_heat, np_heats[4], gt_heatmaps[4], gt_centers, image_plot, pred_centers, accuracy_centers, heatmap_dir, top_line = top_line)
            show_results("Top points", top_bin_heat, np_heats[5], gt_heatmaps[5], gt_top, image_plot, pred_top, accuracy_top, heatmap_dir)

            
            #compute accuracy for each image with valid number of points             
            accuracy_corners = []
            accuracy_corners_lumbar = []
            accuracy_corners_thoracic = []
            distances_corners = []
            distances_corners_lumbar = []
            distances_corners_thoracic = []

            for i in range(4):
                accuracy_corners.append(accuracy(np.array(pred_corners[i]), np.array(gt_corners[i])))
                accuracy_corners_lumbar.append(accuracy(np.array(pred_corners[i])[:12], np.array(gt_corners[i])[:12]))
                accuracy_corners_thoracic.append(accuracy(np.array(pred_corners[i])[12:], np.array(gt_corners[i])[12:]))
                distances_corners.append(distance(np.array(pred_corners[i]), np.array(gt_corners[i])))
                distances_corners_lumbar.append(distance(np.array(pred_corners[i])[:12], np.array(gt_corners[i])[:12]))
                distances_corners_thoracic.append(distance(np.array(pred_corners[i])[12:], np.array(gt_corners[i])[12:]))

            global_accuracy = sum(accuracy_corners)/4
            global_accuracy_lumbar = sum(accuracy_corners_lumbar)/4
            global_accuracy_thoracic = sum(accuracy_corners_thoracic)/4

            test_accuracy += global_accuracy
            test_accuracy_lumbar += global_accuracy_lumbar
            test_accuracy_thoracic += global_accuracy_thoracic

            accuracies.append(global_accuracy)
            lumbar_accuracies.append(global_accuracy_lumbar)
            thoracic_accuracies.append(global_accuracy_thoracic)

            global_distance = sum(distances_corners)/4
            global_distance_lumbar = sum(distances_corners_lumbar)/4
            global_distance_thoracic = sum(distances_corners_thoracic)/4

            distances.append(global_distance)
            lumbar_distances.append(global_distance_lumbar)
            thoracic_distances.append(global_distance_thoracic)

            accuracy_dict = {"global_accuracy" : global_accuracy, "lumbar_accuracy" : global_accuracy_lumbar, "thoracic_accuracy" : global_accuracy_thoracic}
            distance_dict = {"global_distance" : global_distance, "lumbar_distance" : global_distance_lumbar, "thoracic_distance" : global_distance_thoracic}

            for i in range(4):
                show_results(titles[i], corners_bin_heat[i], np_heats[i], gt_heatmaps[i], gt_corners[i], image_plot, pred_corners[i], accuracy_corners[i], heatmap_dir, global_accuracy = global_accuracy)

            # Save images with landmarks
            show_final(gt_corners, pred_corners, image_plot, heatmap_dir, accuracy_dict, distance_dict)

            #reassemble the landmarks
            pred_landmarks = []
            for i in range(17):
                pred_landmarks.append([pred_corners[0][i][0], pred_corners[0][i][1]])
                pred_landmarks.append([pred_corners[1][i][0], pred_corners[1][i][1]])
                pred_landmarks.append([pred_corners[2][i][0], pred_corners[2][i][1]])
                pred_landmarks.append([pred_corners[3][i][0], pred_corners[3][i][1]])

            pred_landmarks = torch.tensor(pred_landmarks)
            final_coords.append(pred_landmarks)
            image = inputs[0]

            # reverse transformations   
            test_transfo = testing_transforms(config["transforms"])
            image_meta_dict, landmarks_meta_dict = (
                batch["image_meta_dict"],
                batch["landmarks_meta_dict"],
            )
            new_batch = {
                "image": image,
                "landmarks": pred_landmarks,
                "image_meta_dict": image_meta_dict,
                "landmarks_meta_dict": landmarks_meta_dict,
            }
            new_batch = test_transfo.inverse(new_batch)
            image = new_batch["image"]
            pred_landmarks = new_batch["landmarks"]

            output_dataset.append({"image": image[0], "landmarks": pred_landmarks, "image_meta_dict": image_meta_dict, "landmarks_meta_dict": landmarks_meta_dict})  
            

    save_dataset(output_dataset, os.path.join(results_dir, "testing_images"), "pred")

    # get the indices of 5 bests accuracy
    bests_acc = np.argsort(accuracies)[-5:][::-1]
    worsts_acc = np.argsort(accuracies)[:5]

    bests_lumbar_acc = np.argsort(lumbar_accuracies)[-5:][::-1]
    worsts_lumbar_acc = np.argsort(lumbar_accuracies)[:5]

    bests_thoracic_acc = np.argsort(thoracic_accuracies)[-5:][::-1]
    worsts_thoracic_acc = np.argsort(thoracic_accuracies)[:5]

    #histogram of the accuracies
    accuracy_inter = np.zeros(11)
    for acc in accuracies:
        accuracy_inter[floor(acc/10)] += 1    

    logger.info(f"Accuracy histogram: {accuracy_inter}")
    #save histogram
    plt.figure()
    plt.bar(range(10), accuracy_inter[:10], align='edge', width = 0.9)
    plt.bar(10, accuracy_inter[10], align='center', width = 0.2, color = 'green')
    plt.xticks(range(0,11), range(0, 101, 10))
    plt.title("Histogram of the accuracies")
    plt.xlabel("Accuracy")
    plt.ylabel("Number of images")
    plt.savefig(os.path.join(results_dir, "accuracy_histogram.png"))

    accuracy_lumbar_inter = np.zeros(11)
    for acc in lumbar_accuracies:
        accuracy_lumbar_inter[floor(acc/10)] += 1

    logger.info(f"Lumbar accuracy histogram: {accuracy_lumbar_inter}")
    #save histogram
    plt.figure()
    plt.bar(range(10), accuracy_lumbar_inter[:10], align='edge', width = 0.9)
    plt.bar(10, accuracy_lumbar_inter[10], align='center', width = 0.2, color = 'green')
    plt.xticks(range(0,11), range(0, 101, 10))
    plt.title("Histogram of the lumbar accuracies")
    plt.xlabel("Accuracy")
    plt.ylabel("Number of images")
    plt.savefig(os.path.join(results_dir, "lumbar_accuracy_histogram.png"))

    accuracy_thoracic_inter = np.zeros(11)
    for acc in thoracic_accuracies:
        accuracy_thoracic_inter[floor(acc/10)] += 1

    logger.info(f"Thoracic accuracy histogram: {accuracy_thoracic_inter}")
    #save histogram
    plt.figure()
    plt.bar(range(10), accuracy_thoracic_inter[:10], align='edge', width = 0.9)
    plt.bar(10, accuracy_thoracic_inter[10], align='center', width = 0.2, color = 'green')
    plt.xticks(range(0,11), range(0, 101, 10))
    plt.title("Histogram of the thoracic accuracies")
    plt.xlabel("Accuracy")
    plt.ylabel("Number of images")
    plt.savefig(os.path.join(results_dir, "thoracic_accuracy_histogram.png"))

    best_names = []
    for i in bests_acc:
        best_names.append(output_dataset[i]["image_meta_dict"]["name"][-1])

    worst_names = []
    for i in worsts_acc:
        worst_names.append(output_dataset[i]["image_meta_dict"]["name"][-1])

    logger.info(f"Best accuracy images: {best_names}")
    logger.info(f"Worst accuracy images: {worst_names}")

    os.makedirs(os.path.join(results_dir, "best_accuracy"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "worst_accuracy"), exist_ok=True)

    folder_path = os.path.join(results_dir, "testing_heatmaps")
    for name in best_names:
        #replace spaces in name with \[space]
        #find corresponding folder
        print(name)
        folder = os.path.join(folder_path, name)
        print(folder)
        #copy all files in the new folder
        os.system(f'cp -r "{folder}" {os.path.join(results_dir, "best_accuracy")}')
    for name in worst_names:
        #find corresponding folder
        folder = os.path.join(folder_path, name)
        #copy all files in the new folder
        os.system(f'cp -r "{folder}" {os.path.join(results_dir, "worst_accuracy")}')


    best_lumbar_names = []
    for i in bests_lumbar_acc:
        best_lumbar_names.append(output_dataset[i]["image_meta_dict"]["name"][-1])

    worst_lumbar_names = []
    for i in worsts_lumbar_acc:
        worst_lumbar_names.append(output_dataset[i]["image_meta_dict"]["name"][-1])

    logger.info(f"Best lumbar accuracy images: {best_lumbar_names}")
    logger.info(f"Worst lumbar accuracy images: {worst_lumbar_names}")

    os.makedirs(os.path.join(results_dir, "best_accuracy", "lumbar"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "worst_accuracy", "lumbar"), exist_ok=True)

    folder_path = os.path.join(results_dir, "testing_heatmaps")
    for name in best_lumbar_names:
        #replace spaces in name with \[space]
        #find corresponding folder
        folder = os.path.join(folder_path, name)
        #copy all files in the new folder
        os.system(f'cp -r "{folder}" {os.path.join(results_dir, "best_accuracy", "lumbar")}')
    for name in worst_lumbar_names:
        #find corresponding folder
        folder = os.path.join(folder_path, name)
        #copy all files in the new folder
        os.system(f'cp -r "{folder}" {os.path.join(results_dir, "worst_accuracy", "lumbar")}')

    best_thoracic_names = []
    for i in bests_thoracic_acc:
        best_thoracic_names.append(output_dataset[i]["image_meta_dict"]["name"][-1])

    worst_thoracic_names = []
    for i in worsts_thoracic_acc:
        worst_thoracic_names.append(output_dataset[i]["image_meta_dict"]["name"][-1])

    logger.info(f"Best thoracic accuracy images: {best_thoracic_names}")
    logger.info(f"Worst thoracic accuracy images: {worst_thoracic_names}")

    os.makedirs(os.path.join(results_dir, "best_accuracy", "thoracic"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "worst_accuracy", "thoracic"), exist_ok=True)

    folder_path = os.path.join(results_dir, "testing_heatmaps")
    for name in best_thoracic_names:
        #replace spaces in name with \[space]
        #find corresponding folder
        folder = os.path.join(folder_path, name)
        #copy all files in the new folder
        os.system(f'cp -r "{folder}" {os.path.join(results_dir, "best_accuracy", "thoracic")}')
    for name in worst_thoracic_names:
        #find corresponding folder
        folder = os.path.join(folder_path, name)
        #copy all files in the new folder
        os.system(f'cp -r "{folder}" {os.path.join(results_dir, "worst_accuracy", "thoracic")}')


    n_valid_images = len(final_coords)
    logger.warning(f"Number of dropped predictions: {len(test_loader) - n_valid_images}/{len(test_loader)}")

    test_accuracy_valid = test_accuracy/ n_valid_images
    test_accuracy_all = test_accuracy / len(test_loader)

    test_accuracy_lumbar_valid = test_accuracy_lumbar/ n_valid_images
    test_accuracy_lumbar_all = test_accuracy_lumbar / len(test_loader)

    test_accuracy_thoracic_valid = test_accuracy_thoracic/ n_valid_images
    test_accuracy_thoracic_all = test_accuracy_thoracic / len(test_loader)

    logger.success(
        f"Testing complete: \n"
        f"\tAccuracy: {test_accuracy_all:.2f}% \n"
        f"\t\t Thoracic: {test_accuracy_thoracic_all:.2f}% ; Lumbar: {test_accuracy_lumbar_all:.2f}% \n"
        f"\tAccuracy (valid images): {test_accuracy_valid:.2f}% \n"
        f"\t\t Thoracic: {test_accuracy_thoracic_valid:.2f}% ; Lumbar: {test_accuracy_lumbar_valid:.2f}% \n"
        f"\tMean distance: {sum(distances)/len(distances):.2f} \n"
        f"\t\t Thoracic: {sum(thoracic_distances)/len(thoracic_distances):.2f} ; Lumbar: {sum(lumbar_distances)/len(lumbar_distances):.2f}"
    )
    

