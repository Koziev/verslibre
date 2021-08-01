import io
import os
import numpy as np
import glob
import random
import argparse

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19

from google_images_search import GoogleImagesSearch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verslibre generator v.2')
    parser.add_argument('--query', type=str, required=True)
    parser.add_argument('--output_path', default='../../../tmp/generated_illustration.png', type=str)
    parser.add_argument('--tmp_dir', default='../../../tmp', type=str)
    parser.add_argument('--models_dir', default='../../../models', type=str)
    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--project_cx', type=str, required=True)
    parser.add_argument('--niter', type=int, default=10000)

    args = parser.parse_args()
    tmp_dir = args.tmp_dir
    models_dir = args.models_dir

    api_key = args.api_key
    project_cx = args.project_cx
    tmp_dir = args.tmp_dir
    styles_dir = os.path.join(models_dir, 'image_styles')

    # you can provide API key and CX using arguments,
    # or you can set environment variables: GCS_DEVELOPER_KEY, GCS_CX
    gis = GoogleImagesSearch(api_key, project_cx)

    query = args.query

    # define search params:
    _search_params = {
        'q': query,
        'num': 1,
        'safe': 'high',
        'fileType': 'jpg',  #'jpg|gif|png',
        'imgType': 'clipart',  #'clipart|face|lineart|news|photo',
        'imgSize': 'imgSizeUndefined',  #'huge|icon|large|medium|small|xlarge|xxlarge',
        'imgDominantColor': 'white', # 'black|blue|brown|gray|green|pink|purple|teal|white|yellow',
        'rights': 'cc_publicdomain'  #'cc_publicdomain|cc_attribute|cc_sharealike|cc_noncommercial|cc_nonderived'
    }

    # this will only search for images:
    #gis.search(search_params=_search_params)

    # this will search and download:
    #gis.search(search_params=_search_params, path_to_dir='/home/inkoziev/polygon/text_generator/tmp')

    # this will search, download and resize:
    #gis.search(search_params=_search_params, path_to_dir='/path/', width=500, height=500)

    # search first, then download and resize afterwards:
    gis.search(search_params=_search_params)
    for image in gis.results():
        #image.resize(500, 500)
        image.download(tmp_dir)
        base_image_path = image.path
        print('base_image_path={}'.format(base_image_path))
        break

    sx = []
    for f in glob.glob(os.path.join(styles_dir, '*.jpg'), recursive=False):
        sx.append(f)
    style_reference_image_path = random.choice(sx)
    #style_reference_image_path = "/home/inkoziev/Pictures/примитивизм_3.jpg"
    print('style_reference_image_path={}'.format(style_reference_image_path))

    # Weights of the different loss components
    total_variation_weight = 1e-6  # 1e-6
    style_weight = 1e-6
    content_weight = 2.5e-8

    # Dimensions of the generated picture.
    width, height = keras.preprocessing.image.load_img(base_image_path).size
    img_nrows = 400
    img_ncols = int(width * img_nrows / height)

    """
    ## Image preprocessing / deprocessing utilities
    """

    def preprocess_image(image_path):
        # Util function to open, resize and format pictures into appropriate tensors
        img = keras.preprocessing.image.load_img(
            image_path, target_size=(img_nrows, img_ncols)
        )
        img = keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = vgg19.preprocess_input(img)
        return tf.convert_to_tensor(img)


    def deprocess_image(x):
        # Util function to convert a tensor into a valid image
        x = x.reshape((img_nrows, img_ncols, 3))
        # Remove zero-center by mean pixel
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        # 'BGR'->'RGB'
        x = x[:, :, ::-1]
        x = np.clip(x, 0, 255).astype("uint8")
        return x


    """
    ## Compute the style transfer loss

    First, we need to define 4 utility functions:

    - `gram_matrix` (used to compute the style loss)
    - The `style_loss` function, which keeps the generated image close to the local textures
    of the style reference image
    - The `content_loss` function, which keeps the high-level representation of the
    generated image close to that of the base image
    - The `total_variation_loss` function, a regularization loss which keeps the generated
    image locally-coherent
    """


    # The gram matrix of an image tensor (feature-wise outer product)

    def gram_matrix(x):
        x = tf.transpose(x, (2, 0, 1))
        features = tf.reshape(x, (tf.shape(x)[0], -1))
        gram = tf.matmul(features, tf.transpose(features))
        return gram


    # The "style loss" is designed to maintain
    # the style of the reference image in the generated image.
    # It is based on the gram matrices (which capture style) of
    # feature maps from the style reference image
    # and from the generated image

    def style_loss(style, combination):
        S = gram_matrix(style)
        C = gram_matrix(combination)
        channels = 3
        size = img_nrows * img_ncols
        return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))


    # An auxiliary loss function
    # designed to maintain the "content" of the
    # base image in the generated image

    def content_loss(base, combination):
        return tf.reduce_sum(tf.square(combination - base))


    # The 3rd loss function, total variation loss,
    # designed to keep the generated image locally coherent

    def total_variation_loss(x):
        a = tf.square(
            x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, 1:, : img_ncols - 1, :]
        )
        b = tf.square(
            x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, : img_nrows - 1, 1:, :]
        )
        return tf.reduce_sum(tf.pow(a + b, 1.25))


    """
    Next, let's create a feature extraction model that retrieves the intermediate activations
    of VGG19 (as a dict, by name).
    """

    # Build a VGG19 model loaded with pre-trained ImageNet weights
    model = vgg19.VGG19(weights="imagenet", include_top=False)

    # Get the symbolic outputs of each "key" layer (we gave them unique names).
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    # Set up a model that returns the activation values for every layer in
    # VGG19 (as a dict).
    feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)

    """
    Finally, here's the code that computes the style transfer loss.
    """

    # List of layers to use for the style loss.
    style_layer_names = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1",
        "block5_conv1",
    ]
    # The layer to use for the content loss.
    content_layer_name = "block5_conv2"


    def compute_loss(combination_image, base_image, style_reference_image):
        input_tensor = tf.concat(
            [base_image, style_reference_image, combination_image], axis=0
        )
        features = feature_extractor(input_tensor)

        # Initialize the loss
        loss = tf.zeros(shape=())

        # Add content loss
        layer_features = features[content_layer_name]
        base_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]
        loss = loss + content_weight * content_loss(base_image_features, combination_features)
        # Add style loss
        for layer_name in style_layer_names:
            layer_features = features[layer_name]
            style_reference_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl = style_loss(style_reference_features, combination_features)
            loss += (style_weight / len(style_layer_names)) * sl

        # Add total variation loss
        loss += total_variation_weight * total_variation_loss(combination_image)
        return loss


    """
    ## Add a tf.function decorator to loss & gradient computation

    To compile it, and thus make it fast.
    """


    @tf.function
    def compute_loss_and_grads(combination_image, base_image, style_reference_image):
        with tf.GradientTape() as tape:
            loss = compute_loss(combination_image, base_image, style_reference_image)
        grads = tape.gradient(loss, combination_image)
        return loss, grads


    """
    ## The training loop

    Repeatedly run vanilla gradient descent steps to minimize the loss, and save the
    resulting image every 100 iterations.

    We decay the learning rate by 0.96 every 100 steps.
    """

    optimizer = keras.optimizers.SGD(
        keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96
        )
    )

    base_image = preprocess_image(base_image_path)
    style_reference_image = preprocess_image(style_reference_image_path)
    combination_image = tf.Variable(preprocess_image(base_image_path))

    iterations = args.niter
    print('Start generating image for query="{}"...'.format(query))
    for i in range(1, iterations + 1):
        loss, grads = compute_loss_and_grads(
            combination_image, base_image, style_reference_image
        )
        optimizer.apply_gradients([(grads, combination_image)])
        if i % 100 == 0:
            print("Iteration %d: loss=%.2f" % (i, loss))
            if i == iterations:
                img = deprocess_image(combination_image.numpy())
                fname = args.output_path
                print('Storing image as "{}"'.format(fname))
                keras.preprocessing.image.save_img(fname, img)
