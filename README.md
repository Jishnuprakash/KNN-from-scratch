# KNN from scratch
 Implementing K-Nearest neighbours for multi class classification from scratch

## About
K-Nearest neighbhour is the world's simplest algorithm for multi-class classification. In this project, KNN is implemented using numpy. You can run and compare KNN with Cosine similarity, Euclidean distance and Manhattan distance. 

## Requirements
python 3, numpy (pip install numpy), matplotlib (pip install matplotlib)

## Folder Structure
Please make sure, the files knn.py and iris.csv are in the same location/folder. 

## How to run the code?
After installing python and numpy, it is very easy to run the knn.py file. 
- Open the code in any Python IDE and click on run
or
- Open command prompt from project folder and run "python /knn.py"


# Code in Detail
## Importing libraries
Importing numpy library for array operations and numpy random module for shuffling the data
Set random seed to constant, to get same accuracy everytime

## Reading the data
readData Method is used to read iris.csv file and convert to numpy array.

## KNN
Class KNN represents the K-nearest neighbhour algorithm. it has following methods, distance, 
predict, and accuracy.

## distance
distance method is used for computing Cosine similarity/ Euclidean distance/ Manhattan distance of input test data and train data.

## predict
predict method is used to execute K-NN algorithm.

## accuracy
accuracy method is used to compute accuracy from predictions and target variables.
