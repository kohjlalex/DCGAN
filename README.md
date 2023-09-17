# Generation of images for model trained on fashion_mnist dataset

# MNIST Fashion Dataset
We will be using the in-built fashion mnist dataset that contains 10,000 gray-scale images, each being 28 by 28 pixels Each image is associated with a label from a a totel of 10 (0-9) possible classes. The clothing or accessory associated with the specific number is indicated below:

0 T-shirt/top
1 Trouser
2 Pullover
3 Dress
4 Coat
5 Sandal
6 Shirt
7 Sneaker
8 Bag
9 Ankle boot

# Preprocessing data
Each of image is represented as a matrix where pixel values range from 0 to 255 since the images are 8-bit grayscale images.
Black colour has pixel value of 0
White colour has pixel value of 255

We will need to centre the pixel values around 0 by making the mean pixel value for each channel to be approximately zero.

To do so, we need to subtract 255/2=127.5 from every pixel value in the dataset before dividing it by 127.5 to scale the centered pixel values by 127.5 to be in the range of -1 to 1.
Centering the data around zero can help with training stability, especially when using activation functions like tanh in neural networks.

The reason for doing this is that many activation functions, like tanh or sigmoid, work well when their inputs are in the range of -1 to 1. This also makes the data distribution more amenable for GAN training because it allows the generator to produce outputs that can have values in a similar range.

# Overview of Model

## Input: 
We will be using a 100-dimensional random noise vector as input and will serve as a source of variability for generating diverse images. It will be a 1-dimensional tensor with 100 elements. Each element in the tensor corresponds to a random number sampled from the normal Gaussian distribution with a mean of 0 and a standard deviation of 1.

## Generator:  
The first layer added to the network is a dense layer with 77256 units. This dense layer takes the random noise as input and transforms it into a higher-dimensional representation. This is often called the "latent space" or "feature space."

Batch normalization is applied to help stabilize and speed up the training process. It normalizes the activations of the previous layer, reducing the risk of vanishing gradients during training.
 
LeakyReLU Activation: Leaky Rectified Linear Unit (LeakyReLU) is used as the activation function. LeakyReLU introduces a small gradient for negative values, which can help the model during training by preventing neurons from becoming "dead."

Reshape the vector of random numbers into a matrix format since the dense layers output in vector format, but we will need matrix format for the subsequent convolutional layers 
  
Convolutional Layers:  
Progressively increase the spatial resolution of the feature maps in the generator network. This gradual upsampling process is crucial for generating high-resolution images with fine details. The combination of convolutional layers, batch normalization, and activation functions helps the generator learn to produce realistic and detailed images as it goes through these layers.

## Discriminator

Matching Generator and Discriminator Architectures: In GANs, it's common practice to design the generator and discriminator with architectures that are inversely related. The generator starts with low-resolution data and upscales it to generate high-resolution images, while the discriminator takes high-resolution images and downscales them to make judgments. This architectural symmetry helps the GAN capture meaningful features and textures at different resolutions. In summary, the downsampling in the discriminator allows it to learn hierarchical features, reduces computational complexity, acts as regularization, and enables the discrimination of high-level image structures. These features contribute to the overall training stability and quality of generated images in a GAN.

The size of the image goes from 28 --> 14 --> 7 which is opposite of generator which does upscaling from 7 --> 14 --> 28
H_out = [(H_in - K_height) / S_height] + 1

Dropout is also used as a simple way to reduce overfitting by dropping out a % of neurons during training. 

The final Flatten() layer flattens the 7x7x128 tensor (from the previous convolutional layers) into a 1D vector. This 1D vector is then passed through a fully connected (Dense) layer with one neuron, which produces a single scalar output. This scalar output represents the discriminator's confidence or probability that the input image is real. In other words, it's a single value indicating how real or fake the discriminator thinks the input image is. Hence, we will only use 1 neuron at output layer. 

# Loss 
BinaryCrossentropy is used as a means to measure the loss and train the model. 

The Adam is an improvement on the gradient descent and is chosen as the optimizer. Learning rate is also set at 0.00001 for both generator and discriminator. 

# Training
Train for 100 epochs

Visualisation of training:
Note that the below are all newly generated images by our model's Generator based on the MNIST training dataset it was trained on.
<img width="662" alt="Screenshot 2023-09-17 at 6 08 44 PM" src="https://github.com/kohjlalex/DCGAN/assets/109453797/65486403-360b-44b0-bfcf-49ab91fc1ba3">
<img width="662" alt="Screenshot 2023-09-17 at 6 09 35 PM" src="https://github.com/kohjlalex/DCGAN/assets/109453797/7e1b8149-3777-49f7-bcb7-b15ef6f8a933">
<img width="662" alt="Screenshot 2023-09-17 at 6 09 57 PM" src="https://github.com/kohjlalex/DCGAN/assets/109453797/bae07b9b-bb63-4fc7-a5b7-3ffc523c72a3">
<img width="662" alt="Screenshot 2023-09-17 at 6 10 14 PM" src="https://github.com/kohjlalex/DCGAN/assets/109453797/760307a9-a411-4059-8bd6-59a2c41df3b7">
<img width="662" alt="Screenshot 2023-09-17 at 6 10 29 PM" src="https://github.com/kohjlalex/DCGAN/assets/109453797/7b4012cb-d1e6-4973-8c36-08ff0a15155f">
