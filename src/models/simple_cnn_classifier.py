import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from keras import layers
from keras.models import Sequential

def main():

    # variables for loading the dataset
    data_path = "/scratch/CS4232_wildlife_classification_project/wildlife_classification/src/cache/datasets/preprocessed_data"
    batch_size = 32
    # images are 1920x1440
    # during model preprocessing scale down by a factor of 4, then crop off the bottom 25 pixels
    img_height = 360
    cropped_img_height = 335
    img_width = 480
    input_img_shape = (batch_size, cropped_img_height, img_width, 3)

    train_ratio = 0.2
    rng_seed = 4232

    # load total dataset first
    total_dataset = tf.keras.utils.image_dataset_from_directory(
        directory = data_path,
        shuffle = True,
        seed = rng_seed,
        image_size = (img_height, img_width),
        batch_size = batch_size)

    # set up training dataset with a 80% split
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        directory = data_path,
        validation_split = train_ratio,
        subset = "training",
        shuffle = True,
        seed = rng_seed,
        image_size = (img_height, img_width),
        batch_size = batch_size
    )
    
    # set up test dataset with a 20% split
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        directory = data_path,
        validation_split = train_ratio,
        subset = "validation",
        shuffle = True,
        seed = rng_seed,
        image_size = (img_height, img_width),
        batch_size = batch_size
    )

    # store class labels
    class_names = train_dataset.class_names
    num_classes = len(class_names)

    # set up dataset caching, so that its loaded into memory on the first epoch and only has to be read from the disk once
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    # factor which images are randomly transformed by for data augmentation (a value of 0.1 means that they will be transformed from 0.9-1.1 their current value)
    augmentation_factor = 0.1

    # rate at which nodes are randomly dropped out in the dropout layers (a rate of 0.1 means that 10% of nodes are randomly dropped out)
    dropout_rate = 0.4

    # store number of classes
    num_classes = len(class_names)

    # create model based on a simple CNN architecture, based off of https://www.tensorflow.org/tutorials/images/cnn
    model_1 = create_model(img_height, img_width, input_img_shape, augmentation_factor, dropout_rate, num_classes)
    
    # print model structure to console
    model_1.summary()

    # compile model & getting ready for training
    model_1.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    # number of training iterations (epochs). 10 for right now, as we currently arent seeing much benefit past 10
    epochs = 10
    history = model_1.fit(
        train_dataset,
        validation_data = test_dataset,
        epochs = epochs)

    # boilerplate code to print a epoch vs. accuracy graph with matplotlib. sourced from https://www.tensorflow.org/tutorials/images/classification
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.tight_layout(pad = 1.5)
    plt.savefig("simple_classifier_training_results_no_dropout.PNG")

    # train final model off of the whole dataset for 4 epochs
    model_output = create_model(img_height,
        img_width,
        input_img_shape,
        augmentation_factor,
        dropout_rate,
        num_classes)

    # overfitting begins at 4 epochs
    final_epochs = 4

    model_output.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    model_output.fit(
        total_dataset,
        epochs = final_epochs)

    output_model_path = "classifier_model.keras"
    model_output.save(output_model_path)

    return 0

def create_model(img_height, img_width, input_img_shape, augmentation_factor, dropout_rate, num_classes):
    model = Sequential([
    # crop the bottom 25 pixels off of the image
    layers.Cropping2D(cropping=((0, 25), (0,0)), input_shape = (img_height, img_width, 3)),
    # data augmentation to apply some randomness and reduce overfitting
    layers.RandomFlip("horizontal", input_shape = input_img_shape),
    # rescale RGB from 0-255 to 0-1
    layers.Rescaling(1./255, input_shape = input_img_shape),
    # 3 convolution -> pooling layers
    layers.Conv2D(16, 3, padding = 'same', activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding = 'same', activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding = 'same', activation = 'relu'),
    layers.MaxPooling2D(),
    # dropout layer to strengthen classifier connections
    layers.Dropout(dropout_rate),
    # flatten highly dimensional outputs from the above layers
    layers.Flatten(),
    # fully connected layer (loads of parameters)
    layers.Dense(512, activation = 'relu'),
    # output layer
    layers.Dense(num_classes)])
    return model

if __name__ == "__main__":
    main()