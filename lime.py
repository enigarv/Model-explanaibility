import os
import numpy as np
import pandas as pd
import cv2
import random
import argparse
from lime import lime_image
import matplotlib.pyplot as plt

def explainer(image:np.ndarray, model, positive_only:bool, num_samples:int=500,
                                             num_superpixel:int=5)->np.ndarray:
    """It builds the LIME explainer in order to understand the model prediction on a 
       specific image. Returns an ImageExplanation object with the corresponding explanations
       that it is used compute the mask for the image.

    Parameters
    ----------
    image : 3d numpy array
         A 3 dimension RGB image image that you want LIME to explain.
    model : _type_
        Classifier prediction probability function, which takes a numpy array 
        and outputs prediction probabilities
    positive_only : bool
        if True, only take superpixels that positively contribute to the
        prediction of the label.
    num_samples : int, optional
        Size of the neighborhood to learn the linear model, it's the amount of
        artificial data points similar to our input that will be generated by
        LIME, by default 500
    num_superpixel : int, optional
        number of superpixels to include in explanation, by default 10
            Returns
    -------
    np.ndarray
         where image is a 3d numpy array and mask is a 2d numpy array that can
         be used with skimage.segmentation.mark_boundaries
    """
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image[0].astype('double'),
                                            model.predict, 
                                            hide_color = None, 
                                            num_samples = num_samples)
    
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], 
                                                positive_only = positive_only, 
                                                num_features = num_superpixel,
                                                hide_rest = True)
    
    return temp, mask