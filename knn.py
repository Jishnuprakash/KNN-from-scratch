# %%
import numpy as np
import matplotlib.pyplot as plt

# Load train and test data
def readData(data):
    """
    Reads data and converts to numpy arrays 
    """
    dat = np.array([i.split(',') for i in open(data,"r").read().splitlines()])
    dat = dat[1:].astype(float)
    return dat

data = readData("iris.csv")
np.random.seed(3)
np.random.shuffle(data)

train_data = np.array(data[:100])
test_data = np.array(data[100:])

#%%
# K-Nearest Neighbour for Multi-class classification
class KNN():
    """
    K- Nearest Neighbour

    Class variable: -
    data - train data
    dist - distance used for classification

    Methods: -
    distance, predict, accuracy
    """

    def __init__(self, data, dist='cos'):
        """
        initialise KNN class
        data -> train data 
        dist -> cos / euclidean / manhattan
        """
        self.data = data
        self.dist = dist

    def distance(self, X):
        """
        Compute distance
        cos, euclidean, manhattan
        """
        distances = []
        if self.dist == 'cos':
            for i in self.data:
                score = np.dot(i[0:4],X) / (np.linalg.norm(i[0:4]) * np.linalg.norm(X))
                distances.append(score)

        elif self.dist == 'euclidean':
            for i in self.data:
                score = np.linalg.norm(i[0:4] - X)
                distances.append(score)
        
        elif self.dist == 'manhattan':
            for i in self.data:
                score = sum(abs(val1-val2) for val1, val2 in zip(i[0:4], X))
                distances.append(score)
        return distances

    def predict(self, x, k):
        """
        Predict x with given value of k 
        """
        # Get labels from train data
        labels = self.data[:, 4]
        #Calculate distance with each data point
        distance = self.distance(x)
        #Find index of distances
        if self.dist == 'cos':
            distanceIndex = np.argsort(distance)[::-1]
        else:
            distanceIndex = np.argsort(distance)
        #Fetch item from labels
        preds = [labels[i] for i in distanceIndex[:k]]
        #Find count of predictions
        counts = [preds.count(i) for i in preds]
        #Return prediction with maximum count
        return preds[np.argmax(counts)]

    def accuracy(self, test_data, K):
        """
        Computes accuracy of predictions and target labels
        """
        pred = [self.predict(i[0:4], K) for i in test_data] 
        #computing accuracy
        acc = sum(1 for x,y in zip(test_data[:,4], pred) if x == y) / float(len(test_data)) 
        return round(acc*100, 2)

def plotAccuracy(kRange, accuracies):
    # Plot the results
    plt.plot(kRange, accuracies, 'b')
    plt.xlabel("Parameter k")
    plt.ylabel("Accuracy of k-NN")
    plt.title("k-NN accuracy versus k")
    plt.show()

#%%

kRange = range(1,100,2)
# Compute accuracies
dist = ['cos', 'euclidean', 'manhattan']

for i in dist:
    knn = KNN(train_data, dist=i)
    accuracies = [knn.accuracy(test_data, i) for i in kRange]
    print("\nDistance used:", i)
    # Find the best k for the current validation dataset
    print("Best k: ", kRange[np.argmax(accuracies)], "with accuracy:", max(accuracies))
    plotAccuracy(kRange, accuracies)


# %%
