import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model

def vgg_features(input_tensor):
    # Load pre-trained VGG-19 model
    vgg_model = VGG19(include_top=False, weights='imagenet')

    # Set the desired layers for computing features
    feature_layers = ['block3_conv3', 'block4_conv3', 'block5_conv3']

    # Create a new model with selected feature layers as outputs
    vgg_features_model = Model(inputs=vgg_model.input,
                               outputs=[vgg_model.get_layer(layer).output for layer in feature_layers])

    # Preprocess the input tensor for VGG-19
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(input_tensor)

    # Compute VGG features for the preprocessed input
    vgg_features_output = vgg_features_model(preprocessed_input)

    return vgg_features_output
