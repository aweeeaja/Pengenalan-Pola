# main.py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from data_loading import load_images_from_folder
from feature_extracion import extract_features_sift, extract_features_orb, extract_features_hog, preprocess_and_extract_features
from model_training import split_and_evaluate, plot_confusion_matrix

# Load images and labels
data_folder = 'D:/file adit/tugas kuliah/sem 6/PePo/Tugas Besar/DataSet_05/DataSet_05'  # Change to the path to your dataset
X, y = load_images_from_folder(data_folder)

# Extract SIFT features
# X_sift = preprocess_and_extract_features(extract_features_sift, X)

# Extract ORB features
X_orb = preprocess_and_extract_features(extract_features_orb, X)

# Extract HOG features
X_hog = preprocess_and_extract_features(extract_features_hog, X)

# Flatten images for no feature extraction scenario
X_flattened = [img.flatten() for img in X]
X_flattened = np.array(X_flattened).astype(np.float32)
scaler_flattened = StandardScaler().fit(X_flattened)
X_flattened = scaler_flattened.transform(X_flattened).astype(np.float32)

# Evaluate with different ratios
ratios = [0.4, 0.3, 0.2]

for ratio in ratios:
    print(f"Evaluasi dengan test size: {ratio}")

    # print("SIFT + SVM/KNN/RandomForest/GradientBoosting")
    # results_sift = split_and_evaluate(X_sift, y, test_size=ratio)
    # for model, metrics in results_sift.items():
    #     print(f"SIFT + {model} - Accuracy: {metrics['accuracy']}, Precision: {metrics['precision']}, Recall: {metrics['recall']}, F1: {metrics['f1']}")
    #     plot_confusion_matrix(metrics['confusion_matrix'], f"SIFT + {model}")

    print("ORB + SVM/KNN/RandomForest/GradientBoosting")
    results_orb = split_and_evaluate(X_orb, y, test_size=ratio)
    for model, metrics in results_orb.items():
        print(f"ORB + {model} - Accuracy: {metrics['accuracy']}, Precision: {metrics['precision']}, Recall: {metrics['recall']}, F1: {metrics['f1']}")
        plot_confusion_matrix(metrics['confusion_matrix'], f"ORB + {model}")

    print("HOG + SVM/KNN/RandomForest/GradientBoosting")
    results_hog = split_and_evaluate(X_hog, y, test_size=ratio)
    for model, metrics in results_hog.items():
        print(f"HOG + {model} - Accuracy: {metrics['accuracy']}, Precision: {metrics['precision']}, Recall: {metrics['recall']}, F1: {metrics['f1']}")
        plot_confusion_matrix(metrics['confusion_matrix'], f"HOG + {model}")

    print("No Feature Extraction + SVM/KNN/RandomForest/GradientBoosting")
    results_flattened = split_and_evaluate(X_flattened, y, test_size=ratio)
    for model, metrics in results_flattened.items():
        print(f"No Feature Extraction + {model} - Accuracy: {metrics['accuracy']}, Precision: {metrics['precision']}, Recall: {metrics['recall']}, F1: {metrics['f1']}")
        plot_confusion_matrix(metrics['confusion_matrix'], f"No Feature Extraction + {model}")
