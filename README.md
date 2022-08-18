# Facial-Expression-Recognition

[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/ShreyasiPeriketi/Facial-Expression-Recognition)

<!-- TABLE OF CONTENTS -->
## Table of Contents

- [Introduction](#introduction)
  - [Scope of the work](#scope-of-the-work)
  - [Product Scenarios](#product-scenarios)
- [Requirement Analysis](#requirement-analysis)
  - [Functional Requirements](#functional-requirements)
  - [Non-Functional Requirements](#non-functional-requirements)
- [Implementation](#implementation)
  - [Work done](#work-done)
  - [Results](#results)
  - [Conclusions and Future work](#conclusions-and-future-work)
  - [References](#references)
 
## Introduction
 
Facial expression recognition is an evolving technology in the field of
human-computer interaction. Facial expression recognition has its branches
spread across various applications such as virtual reality, webinar
technologies, online surveys and many other fields. Even though high
advancements have been witnessed in this field, there are several diplomacies
that exist.

Image processing is the field of signal processing where both the input and
output signals are images. One of the most important applications of Image
processing is Facial expression recognition. Our emotion is revealed by the
expressions in our face.

### Scope of the work

Through this project, we put forward a solution to recognize emotions by
understanding different facial expressions.The facial features are identified by
different operations provided by OpenCV and the region consisting of parts of the
face are made to surround or enclose by a contour. This region, enclosed by the
contour is used as an input to Convolutional Neural Network (CNN). The CNN
model created consists of six activation layers, of which four are convolution
layers and two are fully controlled layers. Each layer is designed to undergo
several training techniques. The main objective of this project is to demonstrate the
accuracy of Convolutional Neural Network model designed.

### Product Scenarios

This project is to develop Automatic Facial Expression
Recognition System which can take human facial images containing some
expression as input and recognize and classify it into seven different
expression class such as :
* Neutral
* Angry
* Disgust
* Fear
* Happy
* Sadness
* Surprise

## Requirement Analysis

There are few requirements to implement this project they are as follows:


### Functional Requirements

Software Requirements:
* Operating System: Any OS with clients to access the internet
* Network: Wi-Fi Internet or cellular Network
* Visual Studio: Create and design Data Flow and Context Diagram and to be able to
* GitHub: Versioning Control
* Packages: All the Machine Learning files and libraries to perform the task.

Hardware Requirements:
* Processor: Intel or Ryzen
* RAM: 8000 MB
* Space on disk: minimum 1000 MB
* Device: Any device that can access the internet


### Non-Functional Requirements

* For this project the model requires a GPU for better performance. Since we
  don’t have a GPU in our development machines, we opted to use Visual Studio, a
  Cloud platform where you can run and deploy your machine learning and deep
  learning models.
* Another requirement is a proper network connection.

## Implementation

To run the application we require dataset, python, libraries such as NumPy,
Pandas, Matplotlib, Seaborn and Sklearn and Pickle.

### Work done

1. Quick Data Visualization
   Displaying some images for every different expression. To count number of train images for each expression:
    * 4830 sad images
    * 4965 neutral images
    * 7215 happy images
    * 436 disgust images
    * 3171 suprise images
    * 4097 fear images
    * 3995 angry images

2. Training and Validation
   During training, to minimize the losses of Neural Network, an algorithm called
   Mini-Batch Gradient Descent has been used. MiniBatch Descent is a type of
   Gradient Descent algorithm used for finding the weights or co-efficient of artificial
   neural networks by splitting the training dataset into small batches. This algorithm
   provides more efficiency whilst training data.

3. Analyzing and Testing
   * Flask App and HTML template:
     An HTML file is created and designed, such that the body is coded to return the
     operations of the flask app. Therefore, an HTML template is created for the Flask
     app with a window of defined height and width that reads and runs the actions
     intended by the flask app.
  
   * Real-time Classification
     In this project, OpenCV’s Haar cascade is used to for the detection and extraction
     of the region containing the face from the video feed of webcam through the flask
     app.
  
   * Creating a Class to Output Model Predictions
     A class is created to Output the predictions of the model. This class is made to
     connect the predictions of the model to the Flask App and is displayed on top of
     the contour rectangle surrounding the face region.
  
### Results

We got outputs at each step of the training phase. All those outputs were saved into
the 'history' variable. We can use it to plot the evolution of the loss and accuracy on
both the train and validation datasets.

### Conclusions and Future work

Using the FER-2013 dataset, a test accuracy of 62.7% is attained with this
designed CNN model. The achieved results are satisfactory as the average
accuracies on the FER-2013 dataset is 65% +/- 5% and therefore, this CNN model
is nearly accurate. For an improvement in this project and its outcomes, it is
recommended to add new parameters wherever useful in the CNN model and
removing unwanted and not-so useful parameters. Adjusting the learning rate
and adapting with the location might help in improving the model.
Accommodating the system to adapt to a low graded illumination setup and
nullify noises in the image can also add onto the efforts to develop the CNN
model. Increasing the layers in the CNN model might not deviate from the
achieved accuracy, but the number of epochs can be set to higher number, to
attain a higher accurate output. Though, increasing the number of epochs to a
certain limit, will increase the accuracy, but increasing the number of epochs to a
higher value will result in over- fitting.
The similar CNN model can be trained and tested for other available datasets and
checked for its accuracy.

### References

* Sharif M., Mohsin S., Hanan R., Javed M. and Raza M., ”Using nose Heuristics for Efficient face Recognition”, Sindh Univ. Res. Jour. (Sci. Ser.) Vol.43 (1-A), 63-68,   (2011)
* Maryam Murtaza, Muhammad Sharif, Mudassar Raza, Jamal Hussain Shah, “Analysis of Face Recognition under Varying Facial Expression: A Survey”, The International Arab   Journal of Information Technology (IAJIT) Volume 10, No.4 , July 2013
* https://medium.com/neurohive-computer-vision/state-of-the-art-facial-expression-recognition-model-introducing-of-covariances-9718c3cca996/
* https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/ -
