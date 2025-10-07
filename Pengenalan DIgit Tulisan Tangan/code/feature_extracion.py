import cv2
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler

def extract_features_sift(images):
    sift = cv2.SIFT_create()
    descriptors_list = []
    for img in images:
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if descriptors is not None:
            descriptors_list.append(descriptors.flatten())
        else:
            descriptors_list.append(np.zeros(sift.descriptorSize()))
    return pad_descriptors(descriptors_list, sift.descriptorSize())

def extract_features_orb(images):
    orb = cv2.ORB_create()
    descriptors_list = []
    for img in images:
        keypoints, descriptors = orb.detectAndCompute(img, None)
        if descriptors is not None:
            descriptors_list.append(descriptors.flatten())
        else:
            descriptors_list.append(np.zeros(32))
    return pad_descriptors(descriptors_list, 32)

def extract_features_hog(images):
    hog_features = []
    for img in images:
        features, _ = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        hog_features.append(features)
    return np.array(hog_features)

def pad_descriptors(descriptors, size):
    max_length = max([desc.shape[0] for desc in descriptors])
    padded_descriptors = []
    for desc in descriptors:
        if desc.shape[0] < max_length:
            padding = np.zeros(max_length - desc.shape[0])
            desc = np.hstack((desc, padding))
        padded_descriptors.append(desc)
    return np.array(padded_descriptors)

def preprocess_and_extract_features(feature_extractor, images):
    images_normalized = [cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8') for img in images]
    descriptors_list = feature_extractor(images_normalized)
    scaler = StandardScaler().fit(descriptors_list)
    descriptors_list = scaler.transform(descriptors_list).astype(np.float32)
    return descriptors_list
