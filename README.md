# Kaggle-StateFarmDistractedDriverDetection
My attempts for this challenge. 

1. Used CNN, structure followed from ZFTurbo's keras sample script on 
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
   
