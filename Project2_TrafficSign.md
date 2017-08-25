#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image_bar_train]: ./examples/bar_Train.png "Training data set"
[image_bar_valid]: ./examples/bar_valid.png "validation data set"
[image_train]: ./examples/trainingdata.png "Sample images from traing data set"
[image_norm]: ./examples/normalization.png "Image normalization"
[image_aug]: ./examples/augument.png "Example augumented images"
[image_learn]: ./examples/learning.png "Learning curve"
[image_web]: ./examples/webimages.png "Web Images"
[image_webnorm]: ./examples/webimages_norm.png "Web Images after normalization"
[image_top0]: ./examples/topprob_image0.png "Top 5 Probabilities, Image 0"
[image_top1]: ./examples/topprob_image1.png "Top 5 Probabilities, Image 1"
[image_feature1]: ./examples/feature1.png "Feature Map convolution layer 1:  Speed limit (20km/h)"
[image_feature1_conv2]: ./examples/feature1_conv2.png "Feature Map convolution layer 2: Speed limit (20km/h)"
[image_feature2]: ./examples/feature2.png "Feature Map convolution layer 1: No entry)"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of validation examples = 12630
* Number of testing examples = 4410
* Image data shape = (32, 32, 3)
* Number of unique classes in the data set = 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set, where two bar charts show the number of images in each of the unique class in the training and validation data sets.
![alt text][image_bar_train]
![alt text][image_bar_valid]
and here is the visualization of randomly selected images from each of the 43 classes in the training data set
![alt text][image_train]


###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to normalize the data set with the mean, max, and min intensities in the training data set with the code:
X_train = (X_train_orig - X_train_orig.mean()) / (np.max(X_train_orig) - np.min(X_train_orig))
X_test = (X_test_orig - X_train_orig.mean()) / (np.max(X_train_orig) - np.min(X_train_orig))
X_valid = (X_valid_orig - X_train_orig.mean()) / (np.max(X_train_orig) - np.min(X_train_orig))

Here is an example of a traffic sign image before and after normalization randomly selected from the training data set.
![alt text][image_norm]

I decided to not to change the images into grayscale because I thought the colors would be helpful to classify the different traffic signs.

I decided to generate additional data because I find that in the samples belongs to the classes with less examples is more likely to be classfied incorrectly. For example, one of the two class 0 'Speed limit (20km/h)' images were predicated as class 1'Speed limit (30km/h)' in my first try with the LeNet architecuture, and there are only 180 images in the training set for class 0.

To add more data to the the data set, for each of the classes that have less than 750 images, I added more images based on transition, with randomly shift between [-2,2] pixels in x and y direction and rotation of the images between negative 3 to positive 3 degrees. The difference between the original data set and the augmented data set is the location of the traffic signs in the images and the angle to capture the pictures. 

Here is an example of an original image and an augmented image:

![alt text][image_aug]

I tried re-train with the augumented training samples, however, from my expriment, there were not much improvement on the validation and testing accuracies. I would like to try again off line to add more valuable augumented images and look at its effectiveness to help training the classifier. The result of this report was got without the auguemented images.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My model architecture was based on the LeNet network, and I added several fully connected layers to it. I also added the drop out layer for the fully connected layers to reduce the possibility of over fitting.
My final model consisted of the following layers:

| Layer         	|     Description	                      |
|:---------------------:|:-------------------------------------------:|
| Input                 | 32x32x3 RGB image   		              |
| Convolution 1 5x5     | 1x1 stride, valid padding, outputs 28x28x16 |
| RELU		        | RELU activation 			      |	
| Max pooling	      	| 2x2 filter and 2x2 stride,  outputs 14x14x16| 
| Convolution 2 5x5	| 2x2 filter and 2x2 stride,  outputs 10x10x32|
| RELU		        | RELU activation 			      |	
| Max pooling	      	| 2x2 filter and 2x2 stride,  outputs 5x5x32  |
| Fully connected 1	| input depth 5x5x32 = 800, output depth 1024, RELU activation and drop out with keep_prob = 0.9 |  
| Fully connected 2	| input depth 1024, output depth 512, RELU activation and drop out with keep_prob = 0.9          |  
| Fully connected 3	| input depth 512, output depth 256, RELU activation and drop out with keep_prob = 0.9           | 
| Fully connected 4	| input depth 256, output depth 128 , RELU activation and drop out with keep_prob = 0.9          |
| Fully connected 5 	| input depth 128, output depth 43, softmax                                                      |   
   

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used the Adam optimizer (already implemented in the LeNet lab). I tried with different batch size, epochs, learning rate, and initialization of the weight and bias paramters.
The final settings used were:
batch size: 150 
epochs: 100 - 
learning rate: 0.0005 
weights: xavier_initializer
biass: all zeros
dropout keep probability: 0.9

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* Training Accuracy = 0.99954
* Validation Accuracy = 0.949
* Test Accuracy = 0.959

The result shows that the model has a very good generalization and perfom wells on classification of the test data set. 

I first tried with the leNet network and the result on the validation set accuracy is about 0.92. I added more fully connected layers to the network which I think the data set is enough to train a deeper networtk so that it could learn the differences between differnet classes. I added the drop out layer at the end of the fully connected layers to avoid over-fitting.
For the number of epochs, I tried with larger numbers and look the learning curve of validation accuracies to decide the number of epochs. Generally, 100 is a good number without validation accuracies dropping too much(please see the learning curve below, where the blue curve is the traing accuracy and the orange curve is the validation accuracy); for the learning rate, because of the osciallation of the accuracies on the training data, I reduced the learning rate from 0.001 to 0.0005; for initialization of the weights, I adopted the xavier_initializer following recommendations from cs231n; while the default keep probability is 0.5, I found a larger probability is better for my network in the sense of training and validation accuracies. I think the reason is that drop too much connections make this network under-fitting. 
![alt text][image_learn]

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 12 German traffic signs that I found on the web:
![alt text][image_web]

The 12 images are first normalized with the same method as above for the validation and the testing data set. The following shows the images after the normalization. 
![alt text][image_webnorm]

The two images from class 0, the 'speed limit 20km/h' might be difficult to classify because there are very few number of training images in this class and it is hard for the network to learn this class. The image for class 30 'Beware of ice/snow' might also be difficult because this traffic sign is actually covered with snow in the images and is very blurred.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:
Traffic Signal:  [0 'Speed limit (20km/h)'], Predicated As:  [0 'Speed limit (20km/h)']
Traffic Signal:  [40 'Roundabout mandatory'], Predicated As:  [40 'Roundabout mandatory']
Traffic Signal:  [11 'Right-of-way at the next intersection'], Predicated As:  [11 'Right-of-way at the next intersection']
Traffic Signal:  [14 'Stop'], Predicated As:  [14 'Stop']
Traffic Signal:  [34 'Turn left ahead'], Predicated As:  [38 'Keep right']
Traffic Signal:  [17 'No entry'], Predicated As:  [17 'No entry']
Traffic Signal:  [25 'Road work'], Predicated As:  [25 'Road work']
Traffic Signal:  [0 'Speed limit (20km/h)'], Predicated As:  [1 'Speed limit (30km/h)']
Traffic Signal:  [30 'Beware of ice/snow'], Predicated As:  [30 'Beware of ice/snow']
Traffic Signal:  [13 'Yield'], Predicated As:  [13 'Yield']
Traffic Signal:  [18 'General caution'], Predicated As:  [18 'General caution']
Traffic Signal:  [17 'No entry'] Predicated As:  [17 'No entry']

The model was able to correctly guess 11 of the 12 traffic signs, which gives an accuracy of 91%. This compares favorably to the accuracy on the test set of 95%. One image that was not classified is the class 34 image for the 'turn left signal' and it was classfied as class 38 'keep right' signal. These two signs do have a lot of similarities, with the direction of the arrows different. The deep nerual network was able to classify the image of the blurred "beware of ice/snow" traffic signal and the two images of the "speed limit 20km/h" signal.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)
The code for making predictions on my final model is located in the 103th-107th cell of the Ipython notebook.

Here is the top 5 predication probabilities of the 12 images.
NFO:tensorflow:Restoring parameters from ./TrafficNet
TopKV2(values=array([
       [  1.00000000e+00,   4.11557385e-14,   1.53433804e-18,
          1.25508735e-19,   1.68883285e-20],
       [  8.57113123e-01,   5.67428768e-02,   4.41951714e-02,
          1.63362492e-02,   8.65457021e-03],
       [  1.00000000e+00,   8.73178297e-22,   1.04739759e-26,
          4.92090316e-27,   3.57731833e-28],
       [  1.00000000e+00,   2.44513372e-12,   4.91105529e-13,
          1.07904415e-13,   2.04702063e-14],
       [  9.99348223e-01,   6.51740062e-04,   1.05283093e-09,
          1.08202163e-11,   4.93816015e-15],
       [  1.00000000e+00,   2.83569907e-25,   8.59564822e-31,
          9.25595157e-35,   2.99911413e-36],
       [  1.00000000e+00,   3.34274116e-22,   1.79111752e-23,
          1.12804980e-24,   6.13768420e-26],
       [  9.89101112e-01,   1.08449729e-02,   3.32006312e-05,
          1.90584124e-05,   1.04063975e-06],
       [  7.23656476e-01,   2.50214487e-01,   2.59854309e-02,
          1.33909256e-04,   3.90221976e-06],
       [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00],
       [  1.00000000e+00,   1.18427458e-26,   1.35073411e-29,
          6.94266994e-31,   4.18497298e-36],
       [  1.00000000e+00,   3.16922185e-34,   1.30399746e-37,
          0.00000000e+00,   0.00000000e+00]], dtype=float32), indices=array([
       [ 0,  1,  8, 28, 24],
       [40, 38, 39,  4,  8],
       [11, 18, 30, 27, 12],
       [14,  3,  1,  9, 26],
       [38, 34, 40, 35, 37],
       [17, 34, 30, 14, 22],
       [25, 30, 20, 13, 24],
       [ 1,  0,  8,  4, 18],
       [30, 11, 23, 28, 19],
       [13,  0,  1,  2,  3],
       [18, 25, 33, 31, 21],
       [17, 34, 22,  0,  1]], dtype=int32))
[0, 40, 11, 14, 34, 17, 25, 0, 30, 13, 18, 17]

Look at the top 5 probabilities, we can see that for image 0, the true class is class 0, and the predication probability of it belonging to class 0 is 1. It does not have any confusion of this predication. The following image shows the bar plot for the top 5 probabilities of image 0.
![alt text][image_top0]

for image 1, the true class is 40 'Roundabout mandatory', the prediction has highest probability for the correct class 40.  it has have non-zero probabilities predicting it to class 38 'keep right' and '39' keep left. Looking at the 'keep left' and 'keep right' traffic signals, they do have similarites with 'roundabout mandatory". All three images have circles and arrows. The following image shows the bar plot for the top 5 probabilities of image 1.  
![alt text][image_top1]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
The feature maps of the first two layers for the test image 0 "Speed limit (20km/h)" are shown below, from which we can see that the network will first learn the most apparent edges in the imagesand adding more and more details in later layers that may not be recognizable by humans. 
![alt text][image_feature1]
![alt text][image_feature1_conv2]


The feature maps of the first layer for the test image 5 "No entry" are shown below, which also shows the edges are learned first in the neural network.
![alt text][image_feature2]



