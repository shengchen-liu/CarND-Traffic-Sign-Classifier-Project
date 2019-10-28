# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # "Image References"

[image0]: ./images/distribution.png "Distribution"
[image1]: ./images/exploration.png "Visualization"
[image2]: ./images/grayscale.png "Grayscaling"
[image3]: ./images/augmentation.png "Augmentation"
[image4]: ./images/lenet-5.png "LeNet"
[image5]: ./data/new_signs/01-Speed-limit-30-km-h.jpg "Traffic Sign 1"
[image6]: ./data/new_signs/03-Speed-limit-60.jpg "Traffic Sign 2"
[image7]: ./data/new_signs/11-Right-of-way-at-the-next-intersection.jpg "Traffic Sign 3"
[image8]: ./data/new_signs/14-Stop.jpg "Traffic Sign 4"
[image10]: ./data/new_signs/33-Turn-right-ahead.jpg "Traffic Sign 5"
[image9]: ./images/RGB.png "RGB"




## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/shengchen-liu/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using Python, Numpy methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It shows random samples from training set (one for each class)

![alt text][image1]

Here is a class distribution in training dataset.  From the distribution we can tell the dataset is heavily imbalanced.  This will have a impact on the accuracy of our model.

![alt text][image0]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As the first step, I decided to convert the images from RGB to YUV and only use the Y channel.  Following this paper [[Sermanet, LeCun\]](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), gray scale images lead to the best performance.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image9]

![alt text][image2]

As the second step, I normalized the image data because it can help overcome overfitting.  Means of the image was calculated and was used for normalization.

I decided to generate additional data because the training data is imbalance.

To add more data to the the data set, I used the following techniques: rotation, zoom, width shift and height shift.

Here is an example of an original image and augmented images:

![alt text][image9]

![alt text][image3]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image |
| C1: Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x6 |
| RELU					|												|
| S2: Max pooling	   | 2x2 stride,  outputs 14x14x6 		|
| C3: Convolution 5x5	| 1x1 stride, valid padding, outputs 10x10x16 |
| S4: Max pooling	| 2x2 stride,  outputs 5x5x16 |
| C5: Fully connection	| outputs 120     |
| F6: Fully connection | outputs 84 |
| Output | output 43 |

![](images/lenet-5.png)

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Optimizer: AdamOptimizer

Batch size: 0.001

Number of epochs: 30

Learning rate: 0.001

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy: 0.979
* validation set accuracy: 0.961 
* test set accuracy: 0.931


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] 
![alt text][image6] 
![alt text][image7] 
![alt text][image8] 
![alt text][image10]

The last image might be difficult to classify because the arrow direction is hard to detect because the augmentation has random rotations.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Speed limit (30km/h) | Speed limit (30km/h) |
|         Speed limit (60km/h)          | Speed limit (60km/h) |
| Right-of-way at the next intersection	| Right-of-way at the next intersection	|
| Stop	    |                 Stop                  |
|           Turn right ahead            | Ahead only  |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This less accurate comparing with the test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Top 5 model predictions for image 0 (Target is Speed limit (60km/h))
Prediction = Speed limit (60km/h) with confidence 0.9999886751174927
Prediction = Speed limit (80km/h) with confidence 1.1331878340570256e-05
Prediction = Speed limit (50km/h) with confidence 4.370902945138866e-12
Prediction = Speed limit (100km/h) with confidence 2.6082038005054153e-12
Prediction = No vehicles with confidence 1.3342442656359005e-14

Top 5 model predictions for image 1 (Target is Turn right ahead)
Prediction = Ahead only with confidence 0.9946415424346924
Prediction = Turn right ahead with confidence 0.0051369620487093925
Prediction = Speed limit (100km/h) with confidence 0.00012270403385628015
Prediction = Speed limit (60km/h) with confidence 6.474601104855537e-05
Prediction = Priority road with confidence 3.2849751733010635e-05

Top 5 model predictions for image 2 (Target is Speed limit (30km/h))
Prediction = Speed limit (30km/h) with confidence 0.9444583058357239
Prediction = Speed limit (60km/h) with confidence 0.05179717391729355
Prediction = End of speed limit (80km/h) with confidence 0.0015145906945690513
Prediction = Speed limit (80km/h) with confidence 0.0011030563618987799
Prediction = Speed limit (20km/h) with confidence 0.00046880898298695683

Top 5 model predictions for image 3 (Target is Right-of-way at the next intersection)
Prediction = Right-of-way at the next intersection with confidence 0.5807305574417114
Prediction = Beware of ice/snow with confidence 0.35899877548217773
Prediction = Road work with confidence 0.046194855123758316
Prediction = Ahead only with confidence 0.008914738893508911
Prediction = Road narrows on the right with confidence 0.0015767159638926387

Top 5 model predictions for image 4 (Target is Stop)
Prediction = Stop with confidence 0.999962329864502
Prediction = Roundabout mandatory with confidence 2.274571124871727e-05
Prediction = Turn right ahead with confidence 1.0225890036963392e-05
Prediction = Keep right with confidence 2.8250333343748935e-06
Prediction = Turn left ahead with confidence 4.803612227988197e-07




