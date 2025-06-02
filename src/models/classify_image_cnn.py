import numpy as np
import tensorflow as tf
import PIL
import sys

from tensorflow import keras
from keras import models
from keras import utils


# prediction method sourced from https://www.tensorflow.org/tutorials/images/classification

def main():

    # Script Parameters

    # path of keras saved model data (parameters, weights, layers)
    model_path = "/scratch/CS4232_wildlife_classification_project/ethan_workspace/classifier_model.keras"
    # path of an image to classify (in this example, a collared peccary)
    image_path = sys.argv[1]

    img_height = 360
    img_width = 480

    labels = ['black_agouti', 'cattle', 'collared_peccary', 'empty', 'lowland_tapir', 'rodent', 'salvins_curassow', 'south_american_coati', 'southern_tamandua', 'spixs_guan', 'spotted_paca', 'unknown', 'unknown_armadillo', 'unknown_bird', 'white-lipped_peccary']

    # Image Loading

    # load image to classify into memory
    input_image = utils.load_img(image_path, target_size=(img_height, img_width))

    # convert to a 1 image batch
    image_array = utils.img_to_array(input_image)
    image_array = tf.expand_dims(image_array, 0)

    # Classification

    # load model from file
    classifier = models.load_model(model_path)

    # classify image and get best prediction
    predictions = classifier.predict(image_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(labels[np.argmax(score)], 100 * np.max(score))
    )

    return 0

if __name__ == "__main__":
    main()