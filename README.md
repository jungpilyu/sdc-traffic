## Project: Build a Traffic Sign Classification Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[dataset]: ./examples/dataset.png
[newimage1]: ./examples/ni0.png
[newimage2]: ./examples/ni1.png
[newimage3]: ./examples/ni2.png
[newimage4]: ./examples/ni3.png
[newimage5]: ./examples/ni4.png
[histogram]: ./examples/histo.png
[newimages]: ./examples/newimages.png
[priorityroad]: ./examples/priorityroad.png
[endofspeed]: ./examples/endofallspeed.png
[turnleft]: ./examples/turnleft.png
[training]: ./examples/train_curve.png

Overview
---
This writeup was written as a partial fillfulment of the requirements for the Nono degree of "Self-driving car engineer" at the Udacity. The goal of this project are the following:
- Consolidate what have been learned about deep neural networks and convolutional neural networks.
- Program traffic signs classifier, train and validate it so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
- Try out the model on images of German traffic signs found on the web.

The project instructions and starter code can be download [here](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).
The project environment can be created with [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md).

The Project
---
This writeup explains the points detailed in [rubric points](https://review.udacity.com/#!/rubrics/481/view) by providing the description in each step and links to other supporting documents and to images to demonstrate how the code works with examples.

The classifier in the project are coded in the following pipeline:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images

### Required file Submission
All required files are submitted in the directory which contains this writeup.
The files are:

* `Traffic_Sign_Classifier.ipynb` : the Ipython notebook with the code
* `Traffic_Sign_Classifier.html` : the code exported as an html file
* `README.md` : this writeup report


### Dataset Exploration
1. In the program, the dataset is summarized as follows.
```
Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (34799, 32, 32, 3)
Number of classes = 43
```
2. An exploratory visualization on the dataset is shown in the below.

![dataset]
The histogram of traffic sign distribution is also shown in the below.
![histogram]
### Design and Test a Model Architecture
1. The preprocessing techniques used are the following:
 * The dataset image are all converted to grayscale version. This reduces the required computer memory and execution time. Besides, it helps the classifier to perform better (but not significantly). This is done by calling a helper function (`X_train = grayscale(X_train)`).
 * The pixel values are normalized in **MinMax Scaling** between [-0.5, 0.5]. This technique helps **gradient descent algorithms** to well converge without overfitting. The image pixel values are scaled by calling a helper function (`X_train = min_max_normal(X_train)`).
2. The used model is based on the **LeNet** architecture. On top of the **LeNet**, the dropout and $L_2$ regularizaion techniques are incorporated into the model. The Architecture of the model is:
 * 1st convolutional layer: Input = 32x32x1. Output = 28x28x6
 * 1st Max pooling layer: Input = 28x28x6. Output = 14x14x6
 * 1st dropout layer
 * 2nd convolutional layer: Input = 14x14x6. Output = 10x10x16
 * 2nd Max pooling layer: Input = 10x10x16. Output = 5x5x16
 * 2dn dropout layer
 * Flatten: Input = 5x5x16. Output = 400
 * 3rd Fully connected layer: Input = 400. Output = 120
 * 3rd dropout layer
 * 4th Fully connected layer: Input = 120. Output = 84
 * 4th dropout layer
 * Final a softmax layer: Input = 84. Output = 43
3. The model was trained by `Adamoptimizer` optimizer. The other parameters are as follows.
 * batch size = 64
 * number of epochs = 20
 * Learning rate = 0.002
 * $L_2$ regularization parameter = 1e-6
 * weight generation in normal distribution with $(\mu, \sigma) = (0, 0.05)$
 * `keep_prob` in dropout = 0.7 during training
4. The final test on the validation and test set attains the accuracy of **0.952** and **0.937** respectively which meets the required performance in rubric. The converging curve in the training and validation phase is shown in the below.
![training]

### Test a Model on New Images
1. New images are acquired in the web. They are shown in the below.
![newimages]
 * new image 1:  This sign not included in the training data set is very similar to the ***Priority road*** and ***End of all speed and passing limits*** sign 
 * new image 2: The top portion of this sign image was truncated. Althought it was not included in the training data set, it is interesting to know which sign the classifier choose.
 * new image 3:  This sign was not included in the training data set but is very similar to the ***Speed limit (30km/h)*** sign.
 * new image 4:  This is the typical ***Speed limit (30km/h)*** sign included in the training data set. This sign is considered to be easily recognized by the classifier.
 * new image 5: This sign was not included in the training data set but it is interesting to see the classifier's choice.
2. The performance of the model when tested on the captured images is shown in the below. Only 1 of 5 is correct(20% accuracy). This is evident since the remaining 4 signs are not included in the training dataset and the classifier saw them for the first time.
 * new image 1 predicts ***End of all speed and passing limits*** ![endofspeed]. The diagonal line was caught by the classifier to predict this sign.
 * new image 2 predicts ***Priority road*** ![priorityroad]. This prediction is somewhat disappointing. However, other training results showed different choice such as ***children crossing*** or ***beware ice/snow***. This result is assumed to be due to the over-training in the training dataset and it is worthy of revisiting later.
 * new image 3 predicts ***Speed limit (20km/h)*** which is similar to new image2.
 * new image 4 predicts ***Speed limit (30km/h)*** which is correct.
 * new image 5 predicts *** Turn left ahead*** ![turnleft]. This is somewhat imaginable result among 43 traffic signs.
3. The top five softmax probabilities of the predictions on the captured images are shown in the below. The classifer has the tendancy to be quite sure of its choice. So, the first choice's probability is almost 1 discarding other possibilities about signs. This is considered due to the over-training with the finite 34 traffic sign training data set. Remind that the validation accuracy has attained **0.95**.
 * new image1
	* Proba =  1.0 Label =  32 ( End of all speed and passing limits )
	* Proba =  0.0 Label =  0 ( Speed limit (20km/h) )
	* Proba =  0.0 Label =  1 ( Speed limit (30km/h) )
	* Proba =  0.0 Label =  2 ( Speed limit (50km/h) )
	* Proba =  0.0 Label =  3 ( Speed limit (60km/h) )
 * new image2
	* Proba =  1.0 Label =  12 ( Priority road )
	* Proba =  1.61327e-26 Label =  32 ( End of all speed and passing limits )
	* Proba =  0.0 Label =  0 ( Speed limit (20km/h) )
	* Proba =  0.0 Label =  1 ( Speed limit (30km/h) )
	* Proba =  0.0 Label =  2 ( Speed limit (50km/h) )
 * new image3
	* Proba =  1.0 Label =  0 ( Speed limit (20km/h) )
	* Proba =  0.0 Label =  1 ( Speed limit (30km/h) )
	* Proba =  0.0 Label =  2 ( Speed limit (50km/h) )
	* Proba =  0.0 Label =  3 ( Speed limit (60km/h) )
	* Proba =  0.0 Label =  4 ( Speed limit (70km/h) )
 * new image4
	* Proba =  1.0 Label =  1 ( Speed limit (30km/h) )
	* Proba =  0.0 Label =  0 ( Speed limit (20km/h) )
	* Proba =  0.0 Label =  2 ( Speed limit (50km/h) )
	* Proba =  0.0 Label =  3 ( Speed limit (60km/h) )
	* Proba =  0.0 Label =  4 ( Speed limit (70km/h) )
 * new image5
	* Proba =  1.0 Label =  34 ( Turn left ahead )
	* Proba =  4.52641e-17 Label =  38 ( Keep right )
	* Proba =  2.8213e-17 Label =  12 ( Priority road )
	* Proba =  1.37152e-18 Label =  17 ( No entry )
	* Proba =  7.14262e-22 Label =  32 ( End of all speed and passing limits )

