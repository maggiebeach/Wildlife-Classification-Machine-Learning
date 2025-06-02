
from pathlib import Path
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.decomposition import PCA


def load_images_from_folder(folder_path, batch_size=750, save_path=None):
    images = []
    labels = []
    label_encoding = {}

    if save_path and os.path.exists(save_path):
        # If save_path is provided and exists, load the saved data
        print(f"Loading data from {save_path}")
        data = np.load(save_path, allow_pickle=True)
        images, labels, label_encoding = data['images'], data['labels'], data['label_encoding']
    else:
        for idx, label in enumerate(os.listdir(folder_path)):
            label_encoding[idx] = label
            label_path = os.path.join(folder_path, label)
            if os.path.isdir(label_path):
                filenames = os.listdir(label_path)
                for start in range(0, len(filenames), batch_size):
                    end = min(start + batch_size, len(filenames))
                    batch_filenames = filenames[start:end]

                    batch_images = []
                    for filename in tqdm(batch_filenames, desc=f"Processing {label}"):
                        img_path = os.path.join(label_path, filename)
                        img = image.load_img(img_path, target_size=(299, 299))
                        img_array = image.img_to_array(img)
                        img_array = preprocess_input(img_array)
                        batch_images.append(img_array)
                        labels.append(label)

                    images.append(np.array(batch_images))


        images = np.concatenate(images, axis=0)

        if save_path:
            print(f"Saving data to {save_path}")
            np.savez(save_path, images=images, labels=labels, label_encoding=label_encoding)


    for encoded_label, dir_name in label_encoding.items():
        print(f"Encoded label {encoded_label} corresponds to directory: {dir_name}")

    return images, np.array(labels), label_encoding



def extract_features_combined(images, model, num_components=100):
    features_list = []

    print("Shape of images:", np.array(images).shape)

    for img in tqdm(images, desc="Extracting features"):
        img = np.expand_dims(img, axis=0)
        features = model.predict(img)

        # Apply Global Average Pooling
        features = np.mean(features, axis=(1, 2))
        features_list.append(features.flatten())

    # Combine features from all images
    all_features = np.array(features_list)

    print("Shape of features before PCA:", all_features.shape)

    # Use PCA for dimensionality reduction
#    pca = PCA(n_components=num_components)
 #   features_reduced = pca.fit_transform(all_features)

   #return features_reduced
    return all_features


def save_to_csv(features, labels, csv_filename='extracted_features_gl.csv'):
    df = pd.DataFrame(features)
    df['label'] = labels

    df.to_csv(csv_filename, index=False)
    print(f"Features and labels saved to {csv_filename}")

def main():
    data_folder = '/scratch/CS4232_wildlife_classification_project/wildlife_classification/src/cache/datasets/preprocessed_data'
    #data_folder = '/scratch/CS4232_wildlife_classification_project/maggie_workspace/wildlife_classification/src/data/test/rf'
    save_path = 'images.npz'
    images, labels, _ = load_images_from_folder(data_folder, save_path=save_path)

    base_model = InceptionV3(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('mixed7').output)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    # Extract features individually
    features = extract_features_combined(images, model)

    # Save to CSV or use the features for further analysis
    save_to_csv(features, encoded_labels)
    decoded_labels = label_encoder.inverse_transform(encoded_labels)
    print(decoded_labels)

if __name__ == "__main__":
    main()






