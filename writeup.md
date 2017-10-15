# Traffic Sign Classification using CNNs

In this project I have used Python3,Numpy,Matplotlib,OpenCV,sklearn,Pickle,Tensorflow to build and train a Convolutional Neural Network that can
classify traffic sign from a given image with 95% accuracy. I have used the following approach to build this classifier

1. Loaded the German Traffic signs image dataset as train and test numpy arrays with labels.
2. Explored the dataset using numpy and matplotlib to find the number of unique classes, number of training and testing observations, 
   dimensions of the images and to get a visual feel of the images.
3. Plotted the distribution of images based on the output classes to find how the data is distributed.
4. Pre-processed the dataset by converting the images to grayscale and normalizing them such that they have zero mean and equal variance
5. From the distribution of images, I observed that the data for some classes is far too less compared to other classes, hence I have 
   augmented the dataset to generate some fake images from the existing images by doing some random scaling, translation, rotation, 
   increasing brightness and smoothing the images.
6. After adding the augmented data to the original dataset shuffled the dataset and split them into training and validation sets using sklearn
7. Saved the dataset to disk using pickle to avoid regenerating the augmented data.
8. Randomly checked the images after pre-processing and plotted the distribution.
9. Designed the Convolutional Neural Network with multiple Convolution, pooling and fully connected layers.
10. Experimented with multiple architectures, dropout, multiple kernel sizes, different types of padding and trained with different combinations
    of Epochs, batchsizes and learning rates.
11. Arrived at a working design with optimal hyper parameters that gave a training accuracy of 99% and Testing accuracy of 95%
12. Tested the model on new images found on web.
