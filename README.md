**Traffic Sign Recognition** 

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./results/data_exploration/visualize_all_signs.png "Visualization of all traffic signs"
[image2]: ./results/data_exploration/training_set_distribution.png "Distribution of the various images in the training set"
[image3]: ./results/data_preprocessing/preprocessing_pipeline1.png "Preprocessing pipeline"
[image4]: ./results/Webimages/all_images.png "All web images"
[image5]: ./results/Webimages/webimage_results0.png "Reults for first 10 images"
[image6]: ./results/view_activations/webimage_dwnsample_0_pool_l1_new.png "Conv layer 1 Image 0"
[image7]: ./results/view_activations/webimage_dwnsample_0_pool_l2_new.png "Conv layer 2 Image 0"
[image8]: ./results/view_activations/webimage_dwnsample_0_pool_l3_new.png "Conv layer 3 Image 0"
[image9]: ./results/view_activations/webimage_dwnsample_0_fully_connected_layers.png "Fully connected layers Image 0"

### Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32 pixels
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set.

Here is an image of one of each of the individual traffic signs in the training dataset

![alt text][image1]

Next we look at the distribution of the number of each individual traffic signs. This is depicted as a bar chart showing the number of samples of each traffic sign in the training dataset. Class id 0-42 represent the 43 individual types of traffic signs in our database.

![alt text][image2]

### Design and Test a Model Architecture

## Input preprocessing description
Here are the steps implemented for data preprocessing.
1. Calculate the Y component of the YCrCb transform for the image
2. Apply localized histogram equalization
3. Normalize the RGB channels by subtracting the mean and dividing by the variance
4. Concatenate the histogram equalized grayscale image as the fourth channel to the normalized RGB image

This creates a 4 channel image frame that the CNN expects. Following is the visualization of the preprocessing steps.
![alt text][image3]

For all the traffic signs that are circular in shape, I have written a script ("autocrop2.py") which finds the relative location of the circular traffic sign with respect to the image and zooms and crops the image in order to maximize the coverage of the traffic sign for that training image. This was done to try and remove the noise in form of background of the traffic sign for the training phase. I will add the images describing the autocropping feature a later point. I tried to write a similar script to locate the triangular signs but that proved to be a challenge for small image sizes with low resolution.

## Final Model Architecture
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x32x32 RGB+Processed grayscale   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, output channels = 36	|
| RELU					|												|
| Average pooling	      	| 2x2 stride, same padding
| Convolution 3x3     	| 1x1 stride, valid padding, output channels = 64	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding
| Convolution 5x5     	| 1x1 stride, valid padding, output channels = 64	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding
| Flattened 		| 1400x1
| Fully connected layer		|input layer size = 1400, output layer size = 400
| RELU
| Fully connected layer		|input layer size = 400, output layer size = 200
| RELU
| Fully connected layer		|input layer size = 200, output layer size = 43

## Training of the model
To train the model, I used an ADAM optimizer with learning rate of 0.001, once the neural network had a validation accuracy of greater than 98%, I throttled the learning rate by a couple orders of magnitude in hope to finetune the network.

My final model results were:
* training set accuracy of 1.0
* validation set accuracy of 0.981 
* test set accuracy of 0.967

Multiple architectures exploring the possibilities of connecting different number of layers with various parameters were created and are now commented out in the code. As I was working on developing these it became apparent that the image preprocessing had a significant impact on the performance of CNN algorithms. Also, adding non-linearity like RELU drastically increases the performance. I tried some other functions to add non-linearity, but RELU worked almost as well as any other option that I tried.


LeNet was used as the starting point for the CNN
* It was selected because it has already proved to be sucessful for low resolution image recognition tasks.
* Traffic signs unlike digits contain information in the color channels and so I decided to use a more beefed up version of LeNet architecture which has 4 channels BGR+G as input. Also, to increase the information content of the traffic sign image and make detection more robust to variable background lighting, reflections, distortions etc the grayscale image is passed through a localized histogram equalization step.


### Test a Model on New Images

Here are 55 German traffic signs that I found on the web. A lot of the new images are snapshots taken while navigating through the streets of Hamburg, Germany on Google Drive. Some of them are from stock photo websites that were available for free use, some of these stock photos actually have trademark logos on the input image, but most of these become minor artifacts when we downsample the image to 32x32. Also, just for curiosity, some of the images are snapshots taken from German traffic sign lists which can be considered ideal test inputs. I wanted to see, for sanity check that the Neural Net trained on real traffic sign images, with various levels of rotation, zoom, lighthing reflections and general degradation due to exposure to elements, can work with the theoretical depictions of the images.

Here is an image that contains all 55 of the web images downloaded and used for further. 
![alt text][image4]

The model classifies all the 55 test images correctly!

Following are the results for the first 10 test images. Result images for the rest are available in the './results/Webimages/' folder
![alt text][image5]

### Visualizing the Neural Network
Using the visualization function provided by Udacity, and modifying it to able to visualize activations of the fully connected layers. We can see how our input images propogate down the Neural Net to result in the end prediction.

Here are a series of visualizations for the activations of the neural net to the first test image downloaded from the internet.
1. Output of 1st convolution layer at the end of the pooling (36 channels)
![alt text][image6]

2. Output of 2nd convolution layer at the end of the pooling (64 channels)
![alt text][image7]

3. Output of 3rd convolution layer at the end of the pooling (64 channels)
![alt text][image8]

4. Output for all the fully connected layers including the flattened output of the 3rd convolution layer and all the hidden layers till the prediction of the class of the traffic sign. Note softmax function has not been applied to the final output just yet.
![alt text][image9]

Similar plots for some other traffic sign images downloaded from the net are available in the results folder. Due to size restriction on the repository to submit to Udacity for review, all the images are not uploaded on the repository. Instead, I have put all the images for a few selected web images. They are up to 2 different traffic signs for each shape available round, triangle, diamond and 1 available for the heptagon stop sign.

