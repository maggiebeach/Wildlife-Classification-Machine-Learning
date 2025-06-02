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


#Prepping image data
def prepare_data(dataset):
    images = []
    labels = []
    for image_batch, label_batch in dataset:
        for image, label in zip(image_batch, label_batch):
            images.append(image.numpy().flatten())
            labels.append(label.numpy())
    return np.array(images), np.array(labels)

def main():
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

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=train_ratio,
        subset="validation",
        shuffle=True,
        seed=rng_seed,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    class_names = train_dataset.class_names
    print(class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    # Prep the data for KNN
    train_images, train_labels = prepare_data(train_dataset)
    test_images, test_labels = prepare_data(test_dataset)

    # normalize the features
    scaler = StandardScaler()
    train_images = scaler.fit_transform(train_images)
    test_images = scaler.transform(test_images)

    # Get accuracy values for multiple K-values
    k_values = [5]
    train_accuracies = []
    test_accuracies = []

    for k in k_values:
        start = time.time()
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_images, train_labels)

        # Training accuracy
        train_predicted_labels = knn.predict(train_images)
        train_accuracy = accuracy_score(train_labels, train_predicted_labels)
        train_accuracies.append(train_accuracy)
        end = time.time()
        train = end -start
        print(train)
	
        start=time.time()
        test_predicted_labels = knn.predict(test_images)
        test_accuracy = accuracy_score(test_labels, test_predicted_labels)
        test_accuracies.append(test_accuracy)
        end=time.time()
        test=end-start
        print(test)
        img_path = '../data/test/whitelipped/whitelipped2.JPG'
        img = Image.open(img_path).convert('RGB')
        img = img.resize((480, 360))

        img_array = np.array(img).flatten().reshape(1, -1)

        scaler = StandardScaler()
        img_array = scaler.fit_transform(img_array)

        predicted_class = knn.predict(img_array)
        print("KNN Predicted Class:", predicted_class)

        joblib.dump(knn, 'knn_model.pkl')
        print(f"K = {k} -- Training Accuracy: {train_accuracy * 100:.2f}% -- Testing Accuracy: {test_accuracy * 100:.2f}%")



if __name__ == "__main__":
    main()
