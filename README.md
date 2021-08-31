# Facial-Keypoints-Detection
## Introduction
Its objective is to predict keypoints positions on face images.This can be used as a building block in several applications,such as :
* tracking faces in images and vedio
* analysis facial expressions
* detecting dysmorphic facial signs for medical diagnosis
* biometrics/face recognition

Each predicted keypoint is specified by an (x,y) real-valued pair in the space of pixel indices,The input image is given in the last field of the data files, and consists of a list of pixels (ordered by row), as integers in (0,255). The images are 96x96 pixels.

Data files are as follows:
* training.csv: list of training 7049 images. Each row contains the (x,y) coordinates for 15 keypoints, and image data as row-ordered list of pixels.
* test.csv: list of 1783 test images. Each row contains ImageId and image data as row-ordered list of pixels
* submissionFileFormat.csv: list of 27124 keypoints to predict. Each row contains a RowId, ImageId, FeatureName, Location. 
FeatureName are "left_eye_center_x," "right_eyebrow_outer_end_y," etc. Location is what you need to predict. 

## Data preprocessing
* Check missing data and fill na with forward element
* Visualiaze single picture with highlighting keypoints

## Build and Train model
* Build CNN model with four Convolution2D layers、three BatchNormalization layers、four MaxPool2D layers and Flatten layer、Dense layer、Dropout layer and output Dense layer.
* Model compiling use adam optimizer、mean_squared_error loss function and accuracy metrics
* The validation accuracy is up to 72.2%

## Test predict and save to Submission.csv

# Citing
https://www.kaggle.com/c/facial-keypoints-detection/overview 
