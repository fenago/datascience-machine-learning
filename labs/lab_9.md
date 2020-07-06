### Introduction

In this scenario, we get into machine learning and how to actually implement machine learning models in Python.

We'll walk through the fascinating concepts of ensemble learning and SVMs, which are some of my favourite machine learning areas!

More specifically, we'll cover the following topics:

- Concept of decision trees and its example in Python
- What is ensemble learning
- Support Vector Machine (SVM) and its example using scikit-learn

### Jupyter Notebooks

We will run Jupyter Notebook as a Docker container. This setup will take some time because of the size of the image.

## Login
When the container is running, execute this statement:
`docker logs jupyter 2>&1 | grep -v "HEAD" `{{execute}}


This will show something like:
```
copy/paste this URL into your browser when you connect for the first time, to login with a token:
    http://localhost:8888/?token=f89b02dd78479d52470b3c3a797408b20cc5a11e067e94b8
    THIS IS NOT YOUR TOKEN.  YOU HAVE TO SEARCH THE LOGS TO GET YOUR TOKEN
```

The token is the value behind `/?token=`. You need that for logging in.

**Note:** You can also run following command to get token directly: 
`docker exec -it jupyter bash -c 'jupyter notebook list' | cut -d'=' -f 2 | cut -d' ' -f 1`{{execute}}

Next, you can open the Jupyter Notebook at 
 https://[[HOST_SUBDOMAIN]]-8888-[[KATACODA_HOST]].environments.katacoda.com/

 ### Ensemble learning

 When we talked about random forests, that was an example of ensemble learning, where we're actually combining multiple models together to come up with a better result than any single model could come up with. So, let's learn about that in a little bit more depth. Let's talk about ensemble learning a little bit more.

So, remember random forests? We had a bunch of decision trees that were using different subsamples of the input data, and different sets of attributes that it would branch on, and they all voted on the final result when you were trying to classify something at the end. That's an example of ensemble learning. Another example: when we were talking about k-means clustering, we had the idea of maybe using different k-means models with different initial random centroids, and letting them all vote on the final result as well. That is also an example of ensemble learning.

Basically, the idea is that you have more than one model, and they might be the same kind of model or it might be different kinds of models, but you run them all, on your set of training data, and they all vote on the final result for whatever it is you're trying to predict. And oftentimes, you'll find that this ensemble of different models produces better results than any single model could on its own.


A good example, from a few years ago, was the Netflix prize. Netflix ran a contest where they offered, I think it was a million dollars, to any researcher who could outperform their existing movie recommendation algorithm. The ones that won were ensemble approaches, where they actually ran multiple recommender algorithms at once and let them all vote on the final result. So, ensemble learning can be a very powerful, yet simple tool, for increasing the quality of your final results in machine learning. Let us now try to explore various types of ensemble learning:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-05-03/steps/3/1.png)

Now, there is a whole field of research on ensemble learning that tries to find the optimal ways of doing ensemble learning, and if you want to sound smart, usually that involves using the word Bayes a lot. So, there are some very advanced methods of doing ensemble learning but all of them have weak points, and I think this is yet another lesson in that we should always use the simplest technique that works well for us.


Now these are all very complicated techniques that I can't really get into in the scope of this book, but at the end of the day, it's hard to outperform just the simple techniques that we've already talked about. A few of the complex techniques are listed here:

- **Bayes optical classifier:** In theory, there's something called the Bayes Optimal Classifier that will always be the best, but it's impractical, because it's computationally prohibitive to do it.
- **Bayesian parameter averaging:** Many people have tried to do variations of the Bayes Optimal Classifier to make it more practical, like the Bayesian Parameter Averaging variation. But it's still susceptible to overfitting and it's often outperformed by bagging, which is the same idea behind random forests; you just resample the data multiple times, run different models, and let them all vote on the final result. Turns out that works just as well, and it's a heck of a lot simpler!
- **Bayesian model combination:** Finally, there's something called Bayesian Model Combination that tries to solve all the shortcomings of Bayes Optimal Classifier and Bayesian Parameter Averaging. But, at the end of the day, it doesn't do much better than just cross validating against the combination of models.

Again, these are very complex techniques that are very difficult to use. In practice, we're better off with the simpler ones that we've talked about in more detail. But, if you want to sound smart and use the word Bayes a lot it's good to be familiar with these techniques at least, and know what they are.

So, that's ensemble learning. Again, the takeaway is that the simple techniques, like bootstrap aggregating, or bagging, or boosting, or stacking, or bucket of models, are usually the right choices. There are some much fancier techniques out there but they're largely theoretical. But, at least you know about them now.

It's always a good idea to try ensemble learning out. It's been proven time and time again that it will produce better results than any single model, so definitely consider it!

### Support vector machine overview

Finally, we're going to talk about support vector machines (SVM), which is a very advanced way of clustering or classifying higher dimensional data.

So, what if you have multiple features that you want to predict from? SVM can be a very powerful tool for doing that, and the results can be scarily good! It's very complicated under the hood, but the important things are understanding when to use it, and how it works at a higher level. So, let's cover SVM now.

Support vector machines is a fancy name for what actually is a fancy concept. But fortunately, it's pretty easy to use. The important thing is knowing what it does, and what it's good for. So, support vector machines works well for classifying higher-dimensional data, and by that I mean lots of different features. So, it's easy to use something like k-means clustering, to cluster data that has two dimensions, you know, maybe age on one axis and income on another. But, what if I have many, many different features that I'm trying to predict from. Well, support vector machines might be a good way of doing that.

Support vector machines finds higher-dimensional support vectors across which to divide the data (mathematically, these support vectors define hyperplanes). That is, mathematically, what support vector machines can do is find higher dimensional support vectors (that's where it gets its name from) that define the higher-dimensional planes that split the data into different clusters.


Obviously the math gets pretty weird pretty quickly with all this. Fortunately, the scikit-learn package will do it all for you, without you having to actually get into it. Under the hood, you need to understand though that it uses something called the kernel trick to actually find those support vectors or hyperplanes that might not be apparent in lower dimensions. There are different kernels you can use, to do this in different ways. The main point is that SVM's are a good choice if you have higher- dimensional data with lots of different features, and there are different kernels you can use that have varying computational costs and might be better fits for the problem at hand.

**Note:**

The important point is that SVMs employ some advanced mathematical trickery to cluster data, and it can handle data sets with lots of features. It's also fairly expensive - the "kernel trick" is the only thing that makes it possible.

I want to point out that SVM is a supervised learning technique. So, we're actually going to train it on a set of training data, and we can use that to make predictions for future unseen data or test data. It's a little bit different than k-means clustering and that k-means was completely unsupervised; with a support vector machine, by contrast, it is training based on actual training data where you have the answer of the correct classification for some set of data that it can learn from. So, SVM's are useful for classification and clustering, if you will - but it's a supervised technique!

One example that you often see with SVMs is using something called support vector classification. The typical example uses the Iris dataset which is one of the sample datasets that comes with scikit-learn. This set is a classification of different flowers, different observations of different Iris flowers and their species. The idea is to classify these using information about the length and width of the petal on each flower, and the length and width of the sepal of each flower. (The sepal, apparently, is a little support structure underneath the petal. I didn't know that until now either.) You have four dimensions of attributes there; you have the length and width of the petal, and the length and the width of the sepal. You can use that to predict the species of an Iris given that information.

Here's an example of doing that with SVC: basically, we have sepal width and sepal length projected down to two dimensions so we can actually visualize it:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-05-03/steps/6/1.jpg)

With different kernels you might get different results. SVC with a linear kernel will produce something very much as you see in the preceding image. You can use polynomial kernels or fancier kernels that might project down to curves in two dimensions as shown in the image. You can do some pretty fancy classification this way.

These have increasing computational costs, and they can produce more complex relationships. But again, it's a case where too much complexity can yield misleading results, so you need to be careful and actually use train/test when appropriate. Since we are doing supervised learning, you can actually do train/test and find the right model that works, or maybe use an ensemble approach.

You need to arrive at the right kernel for the task at hand. For things like polynomial SVC, what's the right degree polynomial to use? Even things like linear SVC will have different parameters associated with them that you might need to optimize for. This will make more sense with a real example, so let's dive into some actual Python code and see how it works!

### Using SVM to cluster people by using scikit-learn

Let's try out some support vector machines here. Fortunately, it's a lot easier to use than it is to understand. We're going to go back to the same example I used for k-means clustering, where I'm going to create some fabricated cluster data about ages and incomes of a hundred random people.

#### Open Notebook
The Notebook opens in a new browser window. You can create a new notebook or open a local one. Check out the local folder `work` for several notebooks. Open and run `SVC.ipynb` in the `work` folder.

You can also open the Jupyter Notebook at https://[[HOST_SUBDOMAIN]]-8888-[[KATACODA_HOST]].environments.katacoda.com/notebooks/work/SVC.ipynb


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

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-05-03/steps/9/1.jpg)


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

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-05-03/steps/9/2.jpg)

You can see here that it did its best. Given that it had to draw straight lines, and polygonal shapes, it did a decent job of fitting to the data that we had. So, you know, it did miss a few - but by and large, the results are pretty good.


SVC is actually a very powerful technique; it's real strength is in higher dimensional feature data. Go ahead and play with it. By the way if you want to not just visualize the results, you can use the predict() function on the SVC model, just like on pretty much any model in scikit-learn, to pass in a feature array that you're interested in. If I want to predict the classification for someone making $200,000 a year who was 40 years old, I would use the following code:

```
svc.predict([[200000, 40]])
```

This would put that person in, in our case, cluster number 1:

If I had a someone making $50,000 here who was 65, I would use the following code:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-05-03/steps/9/3.jpg)

```
svc.predict([[50000, 65]])
```

This is what your output should now look like:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-05-03/steps/9/4.jpg)


That person would end up in cluster number 2, whatever that represents in this example. So,go ahead and play with it.

### Activity

Now, linear is just one of many kernels that you can use, like I said there are many different kernels you can use. One of them is a polynomial model, so you might want to play with that. Please do go ahead and look up the documentation. It's good practice for you to looking at the docs. If you're going to be using scikit-learn in any sort of depth, there's a lot of different capabilities and options that you have available to you. So, go look up scikit-learn online, find out what the other kernels are for the SVC method, and try them out, see if you actually get better results or not.

This is a little exercise, not just in playing with SVM and different kinds of SVC, but also in familiarizing yourself with how to learn more on your own about SVC. And, honestly, a very important trait of any data scientist or engineer is going to be the ability to go and look up information yourself when you don't know the answers.

So, you know, I'm not being lazy by not telling you what those other kernels are, I want you to get used to the idea of having to look this stuff up on your own, because if you have to ask someone else about these things all the time you're going to get really annoying, really fast in a workplace. So, go look that up, play around it, see what you come up with.

So, that's SVM/SVC, a very high power technique that you can use for classifying data, in supervised learning. Now you know how it works and how to use it, so keep that in your bag of tricks!
