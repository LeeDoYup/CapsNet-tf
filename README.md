# Capsule Networks (CapsNet) in tensorflow

Tensorflow implementation of [Dynamic Routing Betwwen Capsules](https://arxiv.org/abs/1710.09829) (Capsule Networks, CapsNet).

Capsule is a vector that represents features with instantiation vector and its norm means the existence probability of the feature.

That is, features are not represented by single neurons, but capsule vectors.

It seems to overcome the limitations of max-pooing.

That is, capsules assure positional "equivalence", not "invariance", and considers spatial relationship between features.

Each capsule is learnt by "dynamic rounting", means "agreement" between low-level capsules.


# Model Description
In this implementation, CapsNet has 3 hidden layers: 1) original conv (256), 2) Primary Capsules, 3) Digit Capsules.

- 1) Original Conv: 256 filters (9x9), strides=1, Valid padding, ReLU
- 2) Primary Capsules: 32 number of 8D capsules, 9x9 and strides=2 conv filters 
- 3) Digit Capsules: 10 number of 16D capsules. *learnt by "Dynamic Routing"

![Model Structure](./assets/CapsNet_Architecture.png)

# Implementation Graph (Tensorboard)

![Graph](./assets/capsnet_graph.png)

# Results

## MNIST classification (without augmentation)

### Accuracy

After 10 epochs with 64 batch_size, test accuracy was about 0.975 %.
![mnist_accuracy_without_aug](./assets/mnist_test_result.jpeg)

With data deformation in training, test accuracy was about 0.988 %
![mnist_accuracy_with_aug](./assets/result_deform.jpeg)


### Losses

![mnist_loss_sum_without_aug](./assets/mnist_loss_summary.png)


### Reconstruction

Test samples (original test images), Reconstruction by masking with y-label, and Reconstruction by masking without y-label.


![recon_result](./assets/recon_result.jpeg)


### Tweak Instanitation Vectors 

Tweak test samples are below. In a digit capsule (16-D), each neuron represents a instantiation parameters such as thinkness, inclination, and so on.

![tweak_result1](./assets/tweak_result_1.png)

![tweak_result2](./assets/tweak_result_2.png)

![tweak_result3](./assets/tweak_result_3.png)


## multi MNIST classification and reconstruction

Now, codes are completed and model is training. Below figure is sample result with small number of training.

![multi_result](./assets/multi_MNIST_sample.jpeg)



# File Descriptions
- main.py : Main function of implementations, contained argument parsers, model construction, and test.
- model.py : CapsNet class
- download.py : Files for downloading MNIST. 
- ops.py : Some operation functions with tensorflow. *ConvCaps Layer implementation included*
- utils.py : Some functions dealing with image preprocessing.


# Prerequisites (my environments)
- Python 2.7
- Tensorflow > 0.14
If other libraries are needed, all libraries are available on pip install --upgrade "library_name"



# Usage

## Download dataset

First, you have to download MNIST dataset.

    $ python download.py mnist

If you want to uses other dataset, make image_load function and loaded them on self.x_data, self.y_data, self.x_test, self.y_test.


## Train CapsNet Model

To train a model with downloaded dataset ( (...) is default setting ):

    $ python main.py --train (--epoch=10) (--batch_size=64) (--learning_rate==0.001)

If you (want to) have validation dataset and save models with lowest validation loss,

    $ python main.py --train --validation_check=True
    
If you want to use data deformation method in the paper, use "--data_deformation" setting in train step.

    $ python main.py --train --data_deformation

Also, you can adjust various hyper-parameters for learning. You can check FLAGS in "main.py"


## Test Trained Model

After training model, you can uses the model for test its performance.

There are three setting for test. 0) accuracy test (default), 1) tweak_test, 2) reconstruction_test

    $ python main.py --train=(True or False) --validation_check=(True or False) --test
    
    $ python main.py --test --tweak_test --reconstruction_test

Then, 0) test performances printed, 1) reconstruction samples are saved in './samples', 2) tweak image results are save in './tweak'.


## Use Tensorboard

Tensorboard's writer files are in './logs'. You can uses it on localhost (default port: 6006).

    $ tensorboard --logdir='./logs' (--port=6006)

## Multi_MNIST dataset

In order to train and test with multi_MNIST dataset (two-digit-overlapped-data), use '--multi_MNIST" setting.

The setting automatically change input_shape as 36x36 with data deformation (from 28x28).

    $ tensorboard --train --multi_MNIST
    
    $ tensorboard --test --multi_MNIST


# For Understanding

- Many variables were tiled for parallel computing of tf.matmul by GPU.
- All representations of the tensor shape are parameterized for model generalization.
