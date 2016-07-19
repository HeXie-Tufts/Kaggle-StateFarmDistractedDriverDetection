# Kaggle-StateFarmDistractedDriverDetection
My attempts for this challenge. 

1. Used CNN, structure followed from ZFTurbo's keras sample script on 
   https://www.kaggle.com/zfturbo/state-farm-distracted-driver-detection/keras-sample/run/202460

   Trained on 48 * 64 Grayscale images

   Achieved Loss 1.57377
2. Used CNN, structure followed Keras example, a simple deep CNN on the CIFAR10 small images dataset.
   https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py

   Trained on 48 * 64 Grayscale images
   1. Added weight regularizer
      Achieved Loss 1.12804
   2. Adjusted data agumentation parameters
      Achieved Loss 1.07232
   
