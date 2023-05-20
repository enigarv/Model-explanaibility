import numpy as np
from IPython.display import Image, display
from skimage.segmentation import mark_boundaries
import matplotlib.cm as cm
import tensorflow as tf

def make_gradcam_heatmap(img_array:np.ndarray, model, layer_name:str)->np.ndarray:
    """It generates the heatmap that allows to understand which areas are
        of interest to the model

    Parameters
    ----------
    img_array : np.ndarray
        Image to explain
    model : _type_
        Classifier prediction probability function, which takes a numpy array 
        and outputs prediction probabilities
    layer_name : str
        Name of the layer to explain

    Returns
    -------
    np.ndarray
        Heatmap regarding the importance of each channel
    """
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    pred_index=None
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

def save_and_display_gradcam(img_path:str, heatmap:np.ndarray, size =(224,224),
                             cam_path="cam.jpg", alpha=0.4)->np.ndarray:
    """Overlays the output of function `make_gradcam_heatmap`on the image
        of interest
    Parameters
    ----------
    img_path : str
        Path of the image
    heatmap : np.ndarray
        Output of the function `make_gradcam_heatmap`
    size : (int,int)
        Output image size
    cam_path : str, optional
        Image name and extension, by default "cam.jpg"
    alpha : float, optional
        Superimposition parameter, by default 0.4

    Returns
    -------
    np.ndarray
        A 3 dimension RGB image
    """
    # Load the original image
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    #display(Image(cam_path))

    return superimposed_img.resize(size)

def combo(img_path:str, model,  layer_name:str, img_size = (224,224))->np.ndarray:
    """Function combining `make_gradcam_heatmap` and `save_and_display_gradcam`

    Parameters
    ----------
    img_path : str
       Path of the image
    model : _type_
        Classifier prediction probability function, which takes a numpy array 
        and outputs prediction probabilities
    layer_name : str
        Name of the layer to explain
    img_size : int,int, optional
        Output image size, by default (224,224)

    Returns
    -------
    np.ndarray
        Image explained by Grad-CAM
    """    
    img_array = get_img_array(img_path, size = img_size)
    # Create heatmap
    heatmap = make_gradcam_heatmap(img_array, model, layer_name)
    # Generate grad-CAM based on heatmap
    output = save_and_display_gradcam(img_path, heatmap)

    return output