"""
Image Classification Project Using OpenCV and scikit-learn
=============================================================
Fashion-MNIST dataset classification pipeline, following the notebook
structure: load -> visualize -> preprocess -> extract HOG features ->
train SVM -> evaluate -> visualize predictions.
"""



import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from skimage.feature import hog

RANDOM_STATE = 42

# 1. Load the dataset

print("Loading dataset...")
df = pd.read_csv("fashion-mnist_test.csv")   # change path if needed

y = df["label"].values
X = df.drop(columns=["label"]).values.astype(np.uint8)
X = X.reshape(-1, 28, 28)   # (N, 28, 28)

print(f"Dataset shape: {X.shape}, labels: {y.shape}")

# 2. Train/test split    (only a test CSV was provided, so we split it ourselves)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# 3. Visualize the Initial Dataset

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def plot_initial_images(images, labels, class_names):
    fig, axes = plt.subplots(1, 10, figsize=(20, 3))
    for i in range(10):
        ax = axes[i]
        ax.imshow(images[i], cmap='gray')
        ax.set_title(class_names[labels[i]])
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("initial_images.png", dpi=150)
    print("Saved initial_images.png")

plot_initial_images(X_train, y_train, class_names)

# 4. Preprocessing of Data

# Normalize pixel value to 0-1

X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape to add a channel -  dimension

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

print(f"Processed training data shape: {X_train.shape}")
print(f"Processed testing data shape: {X_test.shape}")

# 5. Extracting Features(HOG)

def extract_hog_features(images):
    hog_features = []
    for image in images:
        features = hog(
            image, pixels_per_cell=(4, 4), cells_per_block=(2, 2),
            visualize=False, channel_axis=-1
        )
        hog_features.append(features)
    return np.array(hog_features)

print("Extracting HOG features from training set...")
X_train_hog = extract_hog_features(X_train)
print("Extracting HOG features from testing set...")
X_test_hog = extract_hog_features(X_test)

print(f"HOG feature vector length: {X_train_hog.shape[1]}")

# 6. Training Classifier(SVM)

print("Training SVM classifier (linear kernel)...")
svm = SVC(kernel='linear')
svm.fit(X_train_hog, y_train)

train_accuracy = svm.score(X_train_hog, y_train)
print(f"Training accuracy: {train_accuracy * 100:.2f}%")

# 7. Evaluating Model

test_accuracy = svm.score(X_test_hog, y_test)
print(f"Testing accuracy: {test_accuracy * 100:.2f}%")

# 8. Visualizing the Output Prediction

y_pred = svm.predict(X_test_hog)

def plot_output_images(images, true_labels, predicted_labels, class_names):
    fig, axes = plt.subplots(1, 10, figsize=(20, 3))
    for i in range(10):
        ax = axes[i]
        ax.imshow(images[i].reshape(28, 28), cmap='gray')
        ax.set_title(f"True: {class_names[true_labels[i]]}\nPred: {class_names[predicted_labels[i]]}", fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("output_predictions.png", dpi=150)
    print("Saved output_predictions.png")

plot_output_images(X_test, y_test, y_pred, class_names)

print("\nDone.")