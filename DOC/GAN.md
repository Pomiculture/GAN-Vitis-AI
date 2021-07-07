# GANs (Generative Adversarial Networks)
We are going to present in this section the Deep learning model we want to run on the accelerator card.

## Functioning
A GAN (Generative Adversarial Network) is a Deep Learning neural network able to generate content. The concept was first [introduced in 2014 by Ian Goodfellow](https://arxiv.org/abs/1406.2661 "Generative Adversarial Networks").
It is composed of two entities : a generative model and a discriminative one. The first one generates fake content while the second one is a binary classifier that determines whether the input content is fake or real. [The goal of the generator is to fool the discriminator, while the latter tries not to be mistaken](https://towardsdatascience.com/understanding-generative-adversarial-networks-gans-cd6e4651a29 "Understanding Generative Adversarial Networks (GANs)"). While the dicriminator is a convolutional neural network, the generator is a [deconvolutional neural network](https://searchenterpriseai.techtarget.com/definition/deconvolutional-networks-deconvolutional-neural-networks "Deconvolutional networks").

In our case, the content is about images. What is interesting about the GAN is that the generator can not only transform existing images, but also create new ones from noise data. We chose to implement this second case, where the training is unsupervised, as we just have to know that the images from the training set are considered as real images, while the outputs of the generator are fake ones. The input is a [Gaussian distribution](https://www.probabilitycourse.com/chapter4/4_2_3_normal.php "Normal distribution"). 

![GAN Network 2](IMAGES/gan_2.png)

The training step follows the concept of the [Zero-sum game](https://www.investopedia.com/terms/z/zero-sumgame.asp "Zero-Sum Game"). The discriminator alternately receives real images from the training set, and fake ones created by the generator that takes in noise to generate new instances. The discriminator evaluate the authenticity of each input image to determine whether it belongs to the actual training dataset or not. It returns a probability, with 1 representing a prediction of authenticity and 0 representing fake. Therefore, [the discriminator has a feedback loop with the ground truth images, while the generator is in a feedback loop with the discriminator](https://wiki.pathmind.com/generative-adversarial-network-gan "A Beginner's Guide to Generative Adversarial Networks (GANs)").

![GAN Network](IMAGES/gan.png)

## Specific application
It is quite easy to find some examples online. For instance, we took the [Keras](https://keras.io/ "Keras") model from the [Hands-on Machine Learning GAN tutorial by Aurélien Géron](https://github.com/ageron/handson-ml2/blob/master/17_autoencoders_and_gans.ipynb "GAN with Keras"). In this example, the model is trained with the [Fashion MNIST dataset](https://www.kaggle.com/zalando-research/fashionmnist "Fashion MNIST"), which contains a training set of 60,000 samples and a test set of 10,000 samples of articles of clothing. These are 28x28 grayscale images associated with a label from 10 classes (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).
With such a rich dataset and such small data, we can quickly obtain convincing results in a matter of a few training iterations.

![Fashion MNIST data](IMAGES/fashion_mnist.png)

## Adjustments to fit the target platform
At the time of publication of this project, the [Vitis AI tutorials](https://github.com/Xilinx/Vitis-Tutorials/tree/master/Machine_Learning "Vitis AI tutorials"), [Vitis AI examples](https://github.com/Xilinx/Vitis-AI/blob/master/demo/Vitis-AI-Library/README.md "Vitis AI examples") and [Model Zoo list of pre-compiled models](https://github.com/Xilinx/Vitis-AI/tree/master/models/AI-Model-Zoo "Model Zoo") don't propose any GAN model but rather deal with object detectors and classifiers.

In the end, we are only interested in the generative model, as we only want to produce fake but convincing images from input noise. 

![Keras model summary](IMAGES/summary.png)

To be able to run the generator on the target platform, we first need to proceed to several changes in comparison to the original model :
- As we are using the Vitis AI TensorFlow framework, we had to replace the calls to ```keras``` by the submodule ```tf.keras``` from the TensorFlow package for compatibility reasons because we are then going to convert the Keras model to a TensorFlow frozen graph;
- As specified in the [current Vitis AI documentation](https://www.xilinx.com/html_docs/xilinx2019_2/vitis_doc/tensorflow_1x.html "vai_q_tensorflow"), the Vitis AI quantizer needs to receive as input 4-dimensional input node(s). That is why we changed the 2D input noise to a 4D matrix, and therefore added a layer at the entry of the generator model to reshape the input so as to keep the same behaviour as in the original model;
- Due to the fact that [the Vitis AI compiler doesn't support SELU activations](https://www.xilinx.com/html_docs/xilinx2019_2/vitis_doc/compiling_model.html#lah1605503013889__section_a3t_f2y_ykb "Supported OPs and DPU Limitations
Currently Supported Operators"), we replaced them by RELU ones, which are quite close. 

![Activation functions](IMAGES/activations.png)

## References
- [Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition, Aurélien Géron, O'REILLY, 2019](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ "Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow")
- [Generative Adversarial Networks, Ian J. Goodfellow, 2014](https://arxiv.org/abs/1406.2661 "Generative Adversarial Networks")
- **Illustrations**
  - [Generative Adversarial Networks for beginners, O'REILLY](https://www.oreilly.com/content/generative-adversarial-networks-for-beginners/ "Generative Adversarial Networks for beginners")
  - [An intuitive introduction to Generative Adversarial Networks (GANs), Thalles Silva](https://www.freecodecamp.org/news/an-intuitive-introduction-to-generative-adversarial-networks-gans-7a2264a81394/ "An intuitive introduction to Generative Adversarial Networks (GANs)")
