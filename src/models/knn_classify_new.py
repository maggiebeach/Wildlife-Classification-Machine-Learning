
from PIL import Image
import pandas as pd
import joblib
import numpy as np
import PIL
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import time


def prepare_data(dataset):
    images = []
    labels = []
    for image_batch, label_batch in dataset:
        for image, label in zip(image_batch, label_batch):
            images.append(image.numpy().flatten())
            labels.append(label.numpy())
    return np.array(images), np.array(labels)


data_path = "/scratch/CS4232_wildlife_classification_project/wildlife_classification/src/cache/datasets/preprocessed_data"
batch_size = 32
img_height = 360
cropped_img_height = 335
img_width = 480
train_ratio = 0.2
rng_seed = 4232

train_dataset = tf.keras.utils.image_dataset_from_directory(
data_path,
validation_split=train_ratio,
subset="training",
shuffle=True,
seed=rng_seed,
image_size=(img_height, img_width),
batch_size=batch_size
)


class_names = train_dataset.class_names
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# Prep the data for KNN
train_images, train_labels = prepare_data(train_dataset)

# normalize the features
scaler = StandardScaler()
train_images = scaler.fit_transform(train_images)
    
# Load the trained KNN model
knn_model = joblib.load('knn_model.pkl')

# Preprocess the new image
img_path = '../data/test/cattle/cattle.jpg'
img = Image.open(img_path).convert('RGB')
img = img.resize((480, 360))

img_array = np.array(img).flatten().reshape(1, -1)

# Use the same StandardScaler used during training for consistent scaling
scaler = StandardScaler()
scaler.fit(train_images)  # Assuming train_images is the scaled training data
img_array = scaler.transform(img_array)

# Predict using the KNN model
predicted_class_knn = knn_model.predict(img_array)
print("KNN Predicted Class for cow ([1] is cattle):", predicted_class_knn)

# Preprocess the new image
img_path = '../data/test/blackagouti/blackagouti2.JPG'
img = Image.open(img_path).convert('RGB')
img = img.resize((480, 360))

img_array = np.array(img).flatten().reshape(1, -1)

# Use the same StandardScaler used during training for consistent scaling
scaler = StandardScaler()
scaler.fit(train_images)  # Assuming train_images is the scaled training data
img_array = scaler.transform(img_array)

# Predict using the KNN model
predicted_class_knn = knn_model.predict(img_array)
print("KNN Predicted Class for black agouti ([1] is cattle):", predicted_class_knn)

