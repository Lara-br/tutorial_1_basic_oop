### Goal
In this folder you can find the skeleton code for the KNN and KMeans models we coded together in tutorial 1. Of course in the tutorial we coded these mostly from scratch, here you are provided with some more guidance in the class structure. You are HIGHLY encouraged to complete this code on your own if you did not attend tutorial 1. The explanations below specify how these models work. If you need more guidance please refer to google/wikipedia before AI models. Though not always accurate, the wikipedia page for these models provides more than enough information to complete the classes. The completed code can be found under Examples when you are done.

### Explanation
#### k-Nearest Neighbours 
A k-Nearest Neighbours or KNN Model, is a very basic classification model. To fit a KNN its parameters consist of a set of observations of some data and a set of corresponding labels. The requirement of labels makes it a supervised learning model. This is where its training ends. It uses this known data to compute the k (a hyperparameter) nearest labeled points to some new unknown point. It then assigns this new point a label by majority vote. For example, given a new fruit, which has 2 apples and 3 oranges (k=5 in this case) as its nearest neighbours will be labeled an orange.

#### k-Means Clustering
k-Means is a clustering algorithm. This takes a set of unlabeled data and hopes to discern the different classes present in this data. In this case k specifies the amount of classes it tries to find. We first initialize k starting centroids, these can be any k unique random points in our data set. The algorithm from here is specified below. 

1. Assign each point in the data set to one of the centroids. This is done by finding the centroid closest to each point and assigning this point to the corresponding cluster.
2. Compute a new centroid for each cluster. This can be done in different ways but most easily by computing the mean of all the points in the cluster.
3. Repeat steps 1 and 2 until convergence. Convergence can be checked by comparing the difference between the centroids of two consecutive iterations of the algorithm. If the difference is below some predefined threshold, we return the current clusters otherwise we continue.