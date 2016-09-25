# Kaggle-StateFarmDistractedDriverDetection
State Farm hosted this Kaggle challenge (https://www.kaggle.com/c/state-farm-distracted-driver-detection), hoping to test whether dashboard cameras can automatically detect drivers engaging in distracted behaviors. Dataset consists of 22424 2D dashboard camera images, categorized into 10 classes:

* c0: safe driving
* c1: texting - right
* c2: talking on the phone - right
* c3: texting - left
* c4: talking on the phone - left
* c5: operating the radio
* c6: drinking
* c7: reaching behind
* c8: hair and makeup
* c9: talking to passenger

Even though my ability to test out complex models on large size images are limited by only having a MacBook Air, I am still glad that I took part in this competition and learned a lot, such as the use of Keras library,  convolutional neural networks, the pros and cons of various activation function et al. 

Here are a summary of my progress through this challenge. (Unsuccessful ideas and experiments are not documented). 

1. Used Convolutional Neural Network (CNN), structure followed from ZFTurbo's keras sample script on 
   https://www.kaggle.com/zfturbo/state-farm-distracted-driver-detection/keras-sample/run/202460

   Trained on 48 * 64 Grayscale images

   Achieved Loss 1.57377
2. Used CNN, structure followed Keras example, a simple deep CNN on the CIFAR10 small images dataset.
   https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py

   Trained on 48 * 64 Grayscale images
   1. Added and adjusted weight regularizer.
   
      Achieved Loss 1.12804
   2. Adjusted data augmentation parameters.
   
      Achieved Loss 0.89433

   Trained on 72 * 96 Grayscale images
   
   1. Adjusted data augmentation parameters.
      
      Achieved Loss 0.79676
   
