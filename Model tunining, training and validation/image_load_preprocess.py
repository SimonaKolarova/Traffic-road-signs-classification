import os
import sys
import cv2
import tensorflow as tf

def load_preprocess_images(data_dir, n_classes, img_height, img_width, type):
    """
    Loads image data from directory `data_dir`.

    Resizes images to required height and width, normalises their RGB values and 
    transforms them to numpy arrays with dimensions `img_height` x `img_width` x 3.

    If `type` is "train", returns tuple `(images, labels)`, where `images` is a list of all
    of the images and `labels` is a list of integer labels, representing the class of each image.

    If `type` is "validation", returns a list of `images`.
    """

    if type == 'train':
        # Initiate lists
        images = []
        labels = []

        # Full directory path
        main_dir = os.path.abspath(os.curdir) 

        for i in range(n_classes):
            os.chdir(os.path.join(data_dir, str(i)))  # Open directory i
            dir_images = os.listdir()  # Create a list of all images in directory

            for j in range(len(dir_images)):
                image = cv2.imread(dir_images[j])  # Read image from file
                image = tf.keras.preprocessing.image.img_to_array(image)  # Transform image to numpy array
                image = tf.image.resize(image, (img_width, img_height))  # Reshape images
                image = image/255  # Normalize image RGB values
                images.append(image) 
                labels.append(i)

            os.chdir(main_dir)
    
        return (images, labels)

    if type == 'validation':
        # Initiate list
        images = []

        # Full directory path
        main_dir = os.path.abspath(os.curdir) 

        os.chdir(data_dir)
        dir_images = os.listdir()  # Create a list of all images in directory

        for i in range(len(dir_images)):
            image = cv2.imread(dir_images[i])  # Read image from file
            image = tf.keras.preprocessing.image.img_to_array(image)  # Transform image to numpy array
            image = tf.image.resize(image, (img_width, img_height))  # Reshape images
            image = image/255  # Normalize image RGB values
            images.append(image) 

        os.chdir(main_dir)
    
        return images