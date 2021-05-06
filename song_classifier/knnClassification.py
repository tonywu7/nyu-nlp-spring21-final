import sklearn
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import numpy
import scipy



def kmeans(tfidfVectors):
    model = sklearn.cluster.KMeans(n_clusters=4)
    labels = model.fit_predict(tfidfVectors)

    return labels

#def kmeansClassifier(tfidfVectzors, kmeansLabels, testSongs):
    


def knnTrain(tfidfVectors, labels):

    model = KNeighborsClassifier(n_neighbors=7)
    model.fit(tfidfVectors, labels)

    return model

def knnClassify(model,testVectors):

    labels = model.predict(testVectors)

    return labels


def main():
    model = knnTrain(tfidfTrainingVectors, trainingLabels)

    predictions = knnClassify(model, testVectors)

if __name__ == '__main__': sys.exit(main())











