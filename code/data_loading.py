import os
import cv2
import numpy as np

def load_images_from_folder(folder, img_size=(64, 64)):
    images = []
    labels = []
    for subdir, _, files in os.walk(folder):
        for file in files:
            img_path = os.path.join(subdir, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, img_size)
                images.append(img)
                label = int(file.split('_')[0])  # Assuming label is the first part of the filename before '_'
                labels.append(label)
    return np.array(images), np.array(labels)