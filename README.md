# Model-explanaibility
**Model Explainability** is a broad concept of analyzing and understanding the results provided by ML/DL models. It is most often used in the context of “black-box” models, for which it is difficult to demonstrate, how did the model arrive at a specific decision.

In this repository I show how to use two well-known model explainability frameworks: LIME and Grad-CAM.

## LIME
LIME is **model-agnostic**, meaning that it can be applied to any machine learning model. The technique attempts to understand the model by perturbing the input of data samples and understanding how the predictions change. LIME provides local model interpretability modifiyng a single data sample by tweaking the feature values and observes the resulting impact on the output.

## Grad-CAM
Gradient-weighted Class Activation Mapping (Grad-CAM), uses the gradients of any target concept (e.g. ‘cat’ in a convolutional classification network) flowing into the final convolutional layer to produce a coarse **localization map** highlighting the important regions in the image for predicting the concept. Unlike previous approaches, Grad-CAM is applicable to a wide variety of CNN model-families (e.g. CNNs with fully-connected layers or CNNs used for structured outputs, all without architectural changes or re-training. 


