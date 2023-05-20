import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import tensorflow
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def get_img_array(img_path:str, size = (224,224))->np.ndarray:
    """Prepares the image to be analysed by Grad-CAM

    Parameters
    ----------
    img_path : str,
        Path of the image
    size : (int,int)
        Output image size, by default (224,224)
    Returns
    -------
    np.ndarray
        Image converted into an array
    """
    img = load_img(img_path, target_size = size)
    # `array` is a float32 Numpy array of shape (224, 224, 3)
    array = img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 224, 224, 3)
    array = np.expand_dims(array, axis=0)
    # preprocess input with the MobileNet preprocesser
    img_array = preprocess_input(array)
    
    return img_array