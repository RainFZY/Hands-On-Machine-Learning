# k-Nearest-Neighbor

This repository contains files required for the Implementation of K-NN classifier on a randomly generated gaussian data on two types of classes using MATLAB. The k-Nearest Neighbor algorithm basically works by looking around k (where k is an integer from 0 to total number of examples) of the nearest neighbors around a given test point and then asking the question "To what category those neighborhood data points belong to?". By "nearest", here we mean Minimum Euclidean distance between the reference point and the other points in the neighborhood. (Note: Some other implementations also use Hamming distance)

The training phase of the algorithm consists only of storing the feature vectors and class labels of the training samples. In the testing phase, an unlabeled vector (a query or test point) is classified by assigning the label which is most frequent among the k training samples nearest to that query point. Here, for simpilicity, I have considered a gaussian generated data with just two features and two classes. The value of k has been chosen as 3 ( Recommended to choose an odd value for 'k' in order to break the ties between two classes in case of binary classification). A total number of 200 examples have been chosen with 100 in each of the two classes. This is just a fundamental implementation of k-Nearest Neighbors. I will try to add advanced implementations to the repository as well.

Scatter Plots:
========
Before Classification:

![1](https://user-images.githubusercontent.com/30439795/38649663-9cc0d876-3dbd-11e8-8833-9791932efe83.png)

After Classification:

![2](https://user-images.githubusercontent.com/30439795/38649686-bb6d0894-3dbd-11e8-8c21-9735cbeb2142.png)
========
