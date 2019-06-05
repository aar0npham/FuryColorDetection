from Pillow import Image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import itemfreq
from . import knn as knn_classifier


def color_histogram_of_test_image(test_src_image):

    # load the image
    image = test_src_image

    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    features = []
    feature_data = ''
    counter = 0
    for (chan, color) in zip(chans, colors):
        counter = counter + 1

        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # find the peak pixel values for R, G, and B
        elem = np.argmax(hist)

        if counter == 1:
            blue = str(elem)
        elif counter == 2:
            green = str(elem)
        elif counter == 3:
            red = str(elem)
            feature_data = red + ',' + green + ',' + blue
            # print(feature_data)

    with open(os.path.join('data', 'test.data'), 'w') as myfile:
        myfile.write(feature_data)


def color_histogram_of_training_image(img_name):

    # detect image color by using image file name to label training data
    if 'red' in img_name:
        data_source = 'red'
    elif 'yellow' in img_name:
        data_source = 'yellow'
    elif 'green' in img_name:
        data_source = 'green'
    elif 'orange' in img_name:
        data_source = 'orange'
    elif 'white' in img_name:
        data_source = 'white'
    elif 'black' in img_name:
        data_source = 'black'
    elif 'blue' in img_name:
        data_source = 'blue'
    elif 'violet' in img_name:
        data_source = 'violet'

    # load the image
    image = cv2.imread(img_name)

    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    features = []
    feature_data = ''
    counter = 0
    for (chan, color) in zip(chans, colors):
        counter = counter + 1

        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # find the peak pixel values for R, G, and B
        elem = np.argmax(hist)

        if counter == 1:
            blue = str(elem)
        elif counter == 2:
            green = str(elem)
        elif counter == 3:
            red = str(elem)
            feature_data = red + ',' + green + ',' + blue

    with open(os.path.join('data', 'training.data'), 'a') as myfile:
        myfile.write(feature_data + ',' + data_source + '\n')


def training():
    # red color training images
    for f in os.listdir(os.path.join('data', 'red')):
        color_histogram_of_training_image(os.path.join('data', 'red') + f)

    # yellow color training images
    for f in os.listdir(os.path.join('data', 'yellow')):
        color_histogram_of_training_image(os.path.join('data', 'yellow') + f)

    # green color training images
    for f in os.listdir(os.path.join('data', 'green')):
        color_histogram_of_training_image(os.path.join('data', 'green') + f)

    # orange color training images
    for f in os.listdir(os.path.join('data', 'orange')):
        color_histogram_of_training_image(os.path.join('data', 'orange') + f)

    # white color training images
    for f in os.listdir(os.path.join('data', 'white')):
        color_histogram_of_training_image(os.path.join('data', 'white') + f)

    # black color training images
    for f in os.listdir(os.path.join('data', 'black')):
        color_histogram_of_training_image(os.path.join('data', 'black') + f)

    # blue color training images
    for f in os.listdir(os.path.join('data', 'blue')):
        color_histogram_of_training_image(os.path.join('data', 'blue') + f)
