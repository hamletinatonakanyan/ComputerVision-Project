# ComputerVision-Project

Image classification through Convolutional Neural Networks
_________________________________________________________

Task: Classification of cars based on images and information.
_________________________________________________________________
Input data:
Contains:
1. data folder with separate folders of car images by IDs.
2. CSV file with information about different cars, including it's IDs.

Brand, color, year_data:
Contain images with appropriate subfolders for further model building
_________________________________________________________________
Steps:

1. Data Cleaning: 

   a) from information CSV file chose feature, based on which cars' images will be classified,

   b) split data to train(90%) and test(10%) parts, each of which will contain folders with cars' images named by chosen feature's values,

Data file system should be in the following hierarchy:

[folder name]

    |-train
        |-[class 0]
        |-[class 1]
        |-[class 2]
        ...
        |-[class n]
    |-test
        |-[class 0]
        |-[class 1]
        |-[class 2]
        ...
        |-[class n]

2. Model building:
   Chose feature and build the CNN classification model by that feature in the output layer.
   
3. Python files:
   a) "data_cleaning_exploring.py" --> unzipping inputted image folder, explore CSV file through pandas Dataframe, 
      matplotlib and seaborn plotting techniques, make folders hierarchy based on different features
   b) "Image_classification_functions.py" --> functions for model building
   c) "Image_classification_train.py" --> combining all functions in one training process to train the model
   d) "model_train_argparser.py" --> make ArgumentParser and pass parameters for training as arguments.
   

   
