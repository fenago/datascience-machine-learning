<img align="right" src="../images/logo-small.png">


Lab : Machine Learning with Python - Part 3
-------------------------------------


In this scenario, we get into machine learning and how to actually implement machine learning models in Python.

We'll walk through the fascinating concepts of ensemble learning and SVMs, which are some of my favourite machine learning areas!

More specifically, we'll cover the following topics:

- Concept of decision trees and its example in Python
- What is ensemble learning
- Support Vector Machine (SVM) and its example using scikit-learn

#### Pre-reqs:
- Google Chrome (Recommended)

#### Lab Environment
Notebooks are ready to run. All packages have been installed. There is no requirement for any setup.

**Note:** Elev8ed Notebooks (powered by Jupyter) will be accessible at the port given to you by your instructor. Password for jupyterLab : `1234`

All Notebooks are present in `work/datascience-machine-learning` folder.

You can access jupyter lab at `<host-ip>:<port>/lab/workspaces/lab_


### Using SVM to cluster people by using scikit-learn

Let's try out some support vector machines here. Fortunately, it's a lot easier to use than it is to understand. We're going to go back to the same example I used for k-means clustering, where I'm going to create some fabricated cluster data about ages and incomes of a hundred random people.

#### Open Notebook
The Notebook opens in a new browser window. You can create a new notebook or open a local one. Check out the local folder `work` for several notebooks. Open and run `SVC.ipynb` in the `work` folder.




If you want to go back to the k-means clustering section, you can learn more about kind of the idea behind this code that generates the fake data. And if you're ready, please consider the following code:

```
import numpy as np 
 
#Create fake income/age clusters for N people in k clusters 
def createClusteredData(N, k): 
    pointsPerCluster = float(N)/k 
    X = [] 
    y = [] 
    for i in range (k): 
        incomeCentroid = np.random.uniform(20000.0, 200000.0) 
        ageCentroid = np.random.uniform(20.0, 70.0) 
        for j in range(int(pointsPerCluster)): 
            X.append([np.random.normal(incomeCentroid, 10000.0),  
            np.random.normal(ageCentroid, 2.0)]) 
            y.append(i) 
    X = np.array(X) 
    y = np.array(y) 
    return X, y 
```

Please note that because we're using supervised learning here, we not only need the feature data again, but we also need the actual answers for our training dataset.


What the createClusteredData() function does here, is to create a bunch of random data for people that are clustered around k points, based on age and income, and it returns two arrays. The first array is the feature array, that we're calling X, and then we have the array of the thing we're trying to predict for, which we're calling y. A lot of times in scikit-learn when you're creating a model that you can predict from, those are the two inputs that it will take, a list of feature vectors, and the thing that you're trying to predict, that it can learn from. So, we'll go ahead and run that.

So now we're going to use the createClusteredData() function to create 100 random people with 5 different clusters. We will just create a scatter plot to illustrate those, and see where they land up:

```
%matplotlib inline 
from pylab import * 
 
(X, y) = createClusteredData(100, 5) 
 
plt.figure(figsize=(8, 6)) 
plt.scatter(X[:,0], X[:,1], c=y.astype(np.float)) 
plt.show() 
```

The following graph shows our data that we're playing with. Every time you run this you're going to get a different set of clusters. So, you know, I didn't actually use a random seed... to make life interesting.

A couple of new things here--I'm using the figsize parameter on plt.figure() to actually make a larger plot. So, if you ever need to adjust the size in matplotlib, that's how you do it. I'm using that same trick of using the color as the classification number that I end up with. So the number of the cluster that I started with is being plotted as the color of these data points. You can see, it's a pretty challenging problem, there's definitely some intermingling of clusters here:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-05-03/steps/9/1.jpg)


Now we can use linear SVC (SVC is a form of SVM), to actually partition that into clusters. We're going to use SVM with a linear kernel, and with a C value of 1.0. C is just an error penalty term that you can adjust; it's 1 by default. Normally, you won't want to mess with that, but if you're doing some sort of convergence on the right model using ensemble learning or train/test, that's one of the things you can play with. Then, we will fit that model to our feature data, and the actual classifications that we have for our training dataset.

```
from sklearn import svm, datasets 
 
C = 1.0 
svc = svm.SVC(kernel='linear', C=C).fit(X, y) 
```

So, let's go ahead and run that. I don't want to get too much into how we're actually going to visualize the results here, just take it on faith that plotPredictions() is a function that can plot the classification ranges and SVC.

It helps us visualize where different classifications come out. Basically, it's creating a mesh across the entire grid, and it will plot different classifications from the SVC models as different colors on that grid, and then we're going to plot our original data on top of that:

```
def plotPredictions(clf): 
    xx, yy = np.meshgrid(np.arange(0, 250000, 10), 
                     np.arange(10, 70, 0.5)) 
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) 
 
    plt.figure(figsize=(8, 6)) 
    Z = Z.reshape(xx.shape) 
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8) 
    plt.scatter(X[:,0], X[:,1], c=y.astype(np.float)) 
    plt.show() 
 
plotPredictions(svc) 
```

So, let's see how that works out. SVC is computationally expensive, so it takes a long time to run:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-05-03/steps/9/2.jpg)

You can see here that it did its best. Given that it had to draw straight lines, and polygonal shapes, it did a decent job of fitting to the data that we had. So, you know, it did miss a few - but by and large, the results are pretty good.


SVC is actually a very powerful technique; it's real strength is in higher dimensional feature data. Go ahead and play with it. By the way if you want to not just visualize the results, you can use the predict() function on the SVC model, just like on pretty much any model in scikit-learn, to pass in a feature array that you're interested in. If I want to predict the classification for someone making $200,000 a year who was 40 years old, I would use the following code:

```
svc.predict([[200000, 40]])
```

This would put that person in, in our case, cluster number 1:

If I had a someone making $50,000 here who was 65, I would use the following code:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-05-03/steps/9/3.jpg)

```
svc.predict([[50000, 65]])
```

This is what your output should now look like:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-05-03/steps/9/4.jpg)


That person would end up in cluster number 2, whatever that represents in this example. So,go ahead and play with it.

### Activity

Now, linear is just one of many kernels that you can use, like I said there are many different kernels you can use. One of them is a polynomial model, so you might want to play with that. Please do go ahead and look up the documentation. It's good practice for you to looking at the docs. If you're going to be using scikit-learn in any sort of depth, there's a lot of different capabilities and options that you have available to you. So, go look up scikit-learn online, find out what the other kernels are for the SVC method, and try them out, see if you actually get better results or not.

This is a little exercise, not just in playing with SVM and different kinds of SVC, but also in familiarizing yourself with how to learn more on your own about SVC. And, honestly, a very important trait of any data scientist or engineer is going to be the ability to go and look up information yourself when you don't know the answers.

