### Introduction

In this scenario, we talk about a few more data mining and machine learning techniques. We will talk about a really simple technique called k-nearest neighbors (KNN). We'll then use KNN to predict a rating for a movie. After that, we'll go on to talk about dimensionality reduction and principal component analysis. We'll also look at an example of PCA where we will reduce 4D data to two dimensions while still preserving its variance.

We'll then walk through the concept of data warehousing and see the advantages of the newer ELT process over the ETL process. We'll learn the fun concept of reinforcement learning and see the technique used behind the intelligent Pac-Man agent of the Pac-Man game. Lastly, we'll see some fancy terminology used for reinforcement learning.

We'll cover the following topics:

- The concept of k-nearest neighbors
- Implementation of KNN to predict the rating of a movie
- Dimensionality reduction and principal component analysis
- Example of PCA with the Iris dataset
- Data warehousing and ETL versus ELT
- What is reinforcement learning
- The working behind the intelligent Pac-Man game
- Some fancy words used for reinforcement learning

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

 ### K-nearest neighbors - concepts

 Let's talk about a few data mining and machine learning techniques that employers expect you to know about. We'll start with a really simple one called KNN for short. You're going to be surprised at just how simple a good supervised machine learning technique can be. Let's take a look!

KNN sounds fancy but it's actually one of the simplest techniques out there! Let's say you have a scatter plot and you can compute the distance between any two points on that scatter plot. Let's assume that you have a bunch of data that you've already classified, that you can train the system from. If I have a new data point, all I do is look at the KNN based on that distance metric and let them all vote on the classification of that new point.

Let's imagine that the following scatter plot is plotting movies. The squares represent science fiction movies, and the triangles represent drama movies. We'll say that this is plotting ratings versus popularity, or anything else you can dream up:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-07/steps/3/1.jpg)

Here, we have some sort of distance that we can compute based on rating and popularity between any two points on the scatter plot. Let's say a new point comes in, a new movie that we don't know the genre for. What we could do is set K to 3 and take the 3 nearest neighbors to this point on the scatter plot; they can all then vote on the classification of the new point/movie.


You can see if I take the three nearest neighbors (K=3), I have 2 drama movies and 1 science fiction movie. I would then let them all vote, and we would choose the classification of drama for this new point based on those 3 nearest neighbors. Now, if I were to expand this circle to include 5 nearest neighbors, that is K=5, I get a different answer. So, in that case I pick up 3 science fiction and 2 drama movies. If I let them all vote I would end up with a classification of science fiction for the new movie instead.

Our choice of K can be very important. You want to make sure it's small enough that you don't go too far and start picking up irrelevant neighbors, but it has to be big enough to enclose enough data points to get a meaningful sample. So, often you'll have to use train/test or a similar technique to actually determine what the right value of K is for a given dataset. But, at the end of the day, you have to just start with your intuition and work from there.

That's all there is to it, it's just that simple. So, it is a very simple technique. All you're doing is literally taking the k nearest neighbors on a scatter plot, and letting them all vote on a classification. It does qualify as supervised learning because it is using the training data of a set of known points, that is, known classifications, to inform the classification of a new point.


But let's do something a little bit more complicated with it and actually play around with movies, just based on their metadata. Let's see if we can actually figure out the nearest neighbors of a movie based on just the intrinsic values of those movies, for example, the ratings for it, the genre information for it:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-07/steps/3/2.jpg)

In theory, we could recreate something similar to Customers Who Watched This Item Also Watched (the above image is a screenshot from Amazon) just using k-nearest Neighbors. And, I can take it one step further: once I identify the movies that are similar to a given movie based on the k-nearest Neighbors algorithm, I can let them all vote on a predicted rating for that movie.

That's what we're going to do in our next example. So you now have the concepts of KNN, k-nearest neighbors. Let's go ahead and apply that to an example of actually finding movies that are similar to each other and using those nearest neighbor movies to predict the rating for another movie we haven't seen before.

### Using KNN to predict a rating for a movie

Alright, we're going to actually take the simple idea of KNN and apply that to a more complicated problem, and that's predicting the rating of a movie given just its genre and rating information. So, let's dive in and actually try to predict movie ratings just based on the KNN algorithm and see where we get. So, if you want to follow along, go ahead and open up the `KNN.ipynb` and you can play along with me.

#### Open Notebook
The Notebook opens in a new browser window. You can create a new notebook or open a local one. Check out the local folder `work` for several notebooks. Open and run `KNN.ipynb` in the `work` folder.

You can also open the Jupyter Notebook at https://[[HOST_SUBDOMAIN]]-8888-[[KATACODA_HOST]].environments.katacoda.com/notebooks/work/KNN.ipynb

What we're going to do is define a distance metric between movies just based on their metadata. By metadata I just mean information that is intrinsic to the movie, that is, the information associated with the movie. Specifically, we're going to look at the genre classifications of the movie.


Every movie in our MovieLens dataset has additional information on what genre it belongs to. A movie can belong to more than one genre, a genre being something like science fiction, or drama, or comedy, or animation. We will also look at the overall popularity of the movie, given by the number of people who rated it, and we also know the average rating of each movie. I can combine all this information together to basically create a metric of distance between two movies just based on rating information and genre information. Let's see what we get.

We'll use pandas again to make life simple, and if you are following along, again make sure to change the path to the MovieLens dataset to wherever you installed it, which will almost certainly not be what is in this Python notebook.

Please go ahead and change that if you want to follow along. As before, we're just going to import the actual ratings data file itself, which is u.data using the read_csv() function in pandas. We're going to tell that it actually has a tab-delimiter and not a comma. We're going to import the first 3 columns, which represent the user_id, movie_id, and rating, for every individual movie rating in our dataset:

```
import pandas as pd 
 
r_cols = ['user_id', 'movie_id', 'rating'] 
ratings = pd.read_csv('C:\DataScience\ml-100k\u.data', sep='\t', names=r_cols, usecols=range(3)) 
ratings.head()ratings.head() 
```

If we go ahead and run that and look at the top of it, we can see that it's working, here's how the output should look like:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-07/steps/6/1.png)

We end up with a DataFrame that has user_id, movie_id, and rating. For example, user_id 0 rated movie_id 50, which I believe is Star Wars, 5 stars, and so on and so forth.


The next thing we have to figure out is aggregate information about the ratings for each movie. We use the groupby() function in pandas to actually group everything by movie_id. We're going to combine together all the ratings for each individual movie, and we're going to output the number of ratings and the average rating score, that is the mean, for each movie:

```
movieProperties = ratings.groupby('movie_id').agg({'rating': 
 [np.size, np.mean]}) 
movieProperties.head() 
```

Let's go ahead and do that - comes back pretty quickly, here's how the output looks like:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-07/steps/6/2.png)

This gives us another DataFrame that tells us, for example, movie_id 1 had 452 ratings (which is a measure of its popularity, that is, how many people actually watched it and rated it), and a mean review score of 3.8. So, 452 people watched movie_id 1, and they gave it an average review of 3.87, which is pretty good.

Now, the raw number of ratings isn't that useful to us. I mean I don't know if 452 means it's popular or not. So, to normalize that, what we're going to do is basically measure that against the maximum and minimum number of ratings for each movie. We can do that using the lambda function. So, we can apply a function to an entire DataFrame this way.

What we're going to do is use the np.min() and np.max() functions to find the maximum number of ratings and the minimum number of ratings found in the entire dataset. So, we'll take the most popular movie and the least popular movie and find the range there, and normalize everything against that range:

```
movieNumRatings = pd.DataFrame(movieProperties['rating']['size']) 
movieNormalizedNumRatings = movieNumRatings.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))) 
movieNormalizedNumRatings.head() 
```

What this gives us, when we run it, is the following:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-07/steps/6/3.png)

This is basically a measure of popularity for each movie, on a scale of 0 to 1. So, a score of 0 here would mean that nobody watched it, it's the least popular movie, and a score of 1 would mean that everybody watched, it's the most popular movie, or more specifically, the movie that the most people watched. So, we have a measure of movie popularity now that we can use for our distance metric.

Next, let's extract some general information. So, it turns out that there is a u.item file that not only contains the movie names, but also all the genres that each movie belongs to:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-07/steps/6/4.png)

The code above will actually go through each line of u.item. We're doing this the hard way; we're not using any pandas functions; we're just going to use straight-up Python this time. Again, make sure you change that path to wherever you installed this information.

Next, we open our u.item file, and then iterate through every line in the file one at a time. We strip out the new line at the end and split it based on the pipe-delimiters in this file. Then, we extract the movieID, the movie name and all of the individual genre fields. So basically, there's a bunch of 0s and 1s in 19 different fields in this source data, where each one of those fields represents a given genre. We then construct a Python dictionary in the end that maps movie IDs to their names, genres, and then we also fold back in our rating information. So, we will have name, genre, popularity on a scale of 0 to 1, and the average rating. So, that's what this little snippet of code does. Let's run that! And, just to see what we end up with, we can extract the value for movie_id 1:

```
movieDict[1] 
```

Following is the output of the preceding code:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-07/steps/6/5.png)

Entry 1 in our dictionary for movie_id 1 happens to be Toy Story, an old Pixar film from 1995 you've probably heard of. Next is a list of all the genres, where a 0 indicates it is not part of that genre, and 1 indicates it is part of that genre. There is a data file in the MovieLens dataset that will tell you what each of these genre fields actually corresponds to.

For our purposes, that's not actually important, right? We're just trying to measure distance between movies based on their genres. So, all that matters mathematically is how similar this vector of genres is to another movie, okay? The actual genres themselves, not important! We just want to see how same or different two movies are in their genre classifications. So we have that genre list, we have the popularity score that we computed, and we have there the mean or average rating for Toy Story. Okay, let's go ahead and figure out how to combine all this information together into a distance metric, so we can find the k-nearest neighbors for Toy Story, for example.

I've rather arbitrarily computed this ComputeDistance() function, that takes two movie IDs and computes a distance score between the two. We're going to base this, first of all, on the similarity, using a cosine similarity metric, between the two genre vectors. Like I said, we're just going to take the list of genres for each movie and see how similar they are to each other. Again, a 0 indicates it's not part of that genre, a 1 indicates it is.

We will then compare the popularity scores and just take the raw difference, the absolute value of the difference between those two popularity scores and use that toward the distance metric as well. Then, we will use that information alone to define the distance between two movies. So, for example, if we compute the distance between movie IDs 2 and 4, this function would return some distance function based only on the popularity of that movie and on the genres of those movies.

Now, imagine a scatter plot if you will, like we saw back in our example from the previous sections, where one axis might be a measure of genre similarity, based on cosine metric, the other axis might be popularity, okay? We're just finding the distance between these two things:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-07/steps/6/6.png)

For this example, where we're trying to compute the distance using our distance metric between movies 2 and 4, we end up with a score of 0.8


Remember, a far distance means it's not similar, right? We want the nearest neighbors, with the smallest distance. So, a score of 0.8 is a pretty high number on a scale of 0 to 1. So that's telling me that these movies really aren't similar. Let's do a quick sanity check and see what these movies really are:

```
print movieDict[2] 
print movieDict[4] 
```

It turns out it's the movies GoldenEye and Get Shorty, which are pretty darn different movies:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-07/steps/6/7.png)

You know, you have James Bond action-adventure, and a comedy movie - not very similar at all! They're actually comparable in terms of popularity, but the genre difference did it in. Okay! So, let's put it all together!

Next, we're going to write a little bit of code to actually take some given movieID and find the KNN. So, all we have to do is compute the distance between Toy Story and all the other movies in our movie dictionary, and sort the results based on their distance score. That's what the following little snippet of code does. If you want to take a moment to wrap your head around it, it's fairly straightforward.

We have a little getNeighbors() function that will take the movie that we're interested in, and the K neighbors that we want to find. It'll iterate through every movie that we have; if it's actually a different movie than we're looking at, it will compute that distance score from before, append that to the list of results that we have, and sort that result. Then we will pluck off the K top results.


In this example, we're going to set K to 10, find the 10 nearest neighbors. We will find the 10 nearest neighbors using getNeighbors(), and then we will iterate through all these 10 nearest neighbors and compute the average rating from each neighbor. That average rating will inform us of our rating prediction for the movie in question.

**Note:**

As a side effect, we also get the 10 nearest neighbors based on our distance function, which we could call similar movies. So, that information itself is useful. Going back to that "Customers Who Watched Also Watched" example, if you wanted to do a similar feature that was just based on this distance metric and not actual behavior data, this might be a reasonable place to start, right?

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-07/steps/6/8.png)

So, let's go ahead and run this, and see what we end up with. The output of the following code is as follows:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-07/steps/6/9.png)

The results aren't that unreasonable. So, we are using as an example the movie Toy Story, which is movieID 1, and what we came back with, for the top 10 nearest neighbors, are a pretty good selection of comedy and children's movies. So, given that Toy Story is a popular comedy and children's movie, we got a bunch of other popular comedy and children's movies; so, it seems to work! We didn't have to use a bunch of fancy collaborative filtering algorithms, these results aren't that bad.

Next, let's use KNN to predict the rating, where we're thinking of the rating as the classification in this example:

```
avgRating 
```

Following is the output of the preceding code:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-07/steps/6/10.png)

We end up with a predicted rating of 3.34, which actually isn't all that different from the actual rating for that movie, which was 3.87. So not great, but it's not too bad either! I mean it actually works surprisingly well, given how simple this algorithm is!

### Activity

Most of the complexity in this example was just in determining our distance metric, and you know we intentionally got a little bit fancy there just to keep it interesting, but you can do anything else you want to. So, if you want fiddle around with this, I definitely encourage you to do so. Our choice of 10 for K was completely out of thin air, I just made that up. How does this respond to different K values? Do you get better results with a higher value of K? Or with a lower value of K? Does it matter?

If you really want to do a more involved exercise you can actually try to apply it to train/test, to actually find the value of K that most optimally can predict the rating of the given movie based on KNN. And, you can use different distance metrics, I kind of made that up too! So, play around the distance metric, maybe you can use different sources of information, or weigh things differently. It might be an interesting thing to do. Maybe, popularity isn't really as important as the genre information, or maybe it's the other way around. See what impact that has on your results too. So, go ahead and mess with these algorithms, mess with the code and run with it, and see what you can get! And, if you do find a significant way of improving on this, share that with your classmates.

That is KNN in action! So, a very simple concept but it can be actually pretty powerful. So, there you have it: similar movies just based on the genre and popularity and nothing else. Works out surprisingly well! And, we used the concept of KNN to actually use those nearest neighbors to predict a rating for a new movie, and that actually worked out pretty well too. So, that's KNN in action, very simple technique but often it works out pretty darn good!

### Dimensionality reduction and principal component analysis

Alright, time to get all trippy! We're going to talking about higher dimensions, and dimensionality reduction. Sounds scary! There is some fancy math involved, but conceptually it's not as hard to grasp as you might think. So, let's talk about dimensionality reduction and principal component analysis next. Very dramatic sounding! Usually when people talk about this, they're talking about a technique called principal component analysis or PCA, and a specific technique called singular value decomposition or SVD. So PCA and SVD are the topics of this section. Let's dive into it!

#### Dimensionality reduction
So, what is the curse of dimensionality? Well, a lot of problems can be thought of having many different dimensions. So, for example, when we were doing movie recommendations, we had attributes of various movies, and every individual movie could be thought of as its own dimension in that data space.

If you have a lot of movies, that's a lot of dimensions and you can't really wrap your head around more than 3, because that's what we grew up to evolve within. You might have some sort of data that has many different features that you care about. You know, in a moment we'll look at an example of flowers that we want to classify, and that classification is based on 4 different measurements of the flowers. Those 4 different features, those 4 different measurements can represent 4 dimensions, which again, is very hard to visualize.

For this reason, dimensionality reduction techniques exist to find a way to reduce higher dimensional information into lower dimensional information. Not only can that make it easier to look at, and classify things, but it can also be useful for things like compressing data. So, by preserving the maximum amount of variance, while we reduce the number of dimensions, we're more compactly representing a dataset. A very common application of dimensionality reduction is not just for visualization, but also for compression, and for feature extraction. We'll talk about that a little bit more in a moment.

A very simple example of dimensionality reduction can be thought of as k-means clustering:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-07/steps/11/1.png)

So you know, for example, we might start off with many points that represent many different dimensions in a dataset. But, ultimately, we can boil that down to K different centroids, and your distance to those centroids. That's one way of boiling data down to a lower dimensional representation.

### Principal component analysis


Usually, when people talk about dimensionality reduction, they're talking about a technique called principal component analysis. This is a much more-fancy technique, it gets into some pretty involved mathematics. But, at a high-level, all you need to know is that it takes a higher dimensional data space, and it finds planes within that data space and higher dimensions.

These higher dimensional planes are called hyper planes, and they are defined by things called eigenvectors. You take as many planes as you want dimensions in the end, project that data onto those hyperplanes, and those become the new axes in your lower dimensional data space:


You know, unless you're familiar with higher dimensional math and you've thought about it before, it's going to be hard to wrap your head around! But, at the end of the day, it means we're choosing planes in a higher dimensional space that still preserve the most variance in our data, and project the data onto those higher dimensional planes that we then bring into a lower dimensional space, okay?

You don't really have to understand all the math to use it; the important point is that it's a very principled way of reducing a dataset down to a lower dimensional space while still preserving the variance within it. We talked about image compression as one application of this. So you know, if I want to reduce the dimensionality in an image, I could use PCA to boil it down to its essence.

Facial recognition is another example. So, if I have a dataset of faces, maybe each face represents a third dimension of 2D images, and I want to boil that down, SVD and principal component analysis can be a way to identify the features that really count in a face. So, it might end up focusing more on the eyes and the mouth, for example, those important features that are necessary for preserving the variance within that dataset. So, it can produce some very interesting and very useful results that just emerge naturally out of the data, which is kind of cool!

To make it real, we're going to use a simpler example, using what's called the Iris dataset. This is a dataset that's included with scikit-learn. It's used pretty commonly in examples, and here's the idea behind it: So, an Iris actually has 2 different kinds of petals on its flower. One's called a petal, which is the flower petals you're familiar with, and it also has something called a sepal, which is kind of this supportive lower set of petals on the flower.

We can take a bunch of different species of Iris, and measure the petal length and width, and the sepal length and width. So, together the length and width of the petal, and the length and width of the sepal are 4 different measurements that correspond to 4 different dimensions in our dataset. I want to use that to classify what species an Iris might belong to. Now, PCA will let us visualize this in 2 dimensions instead of 4, while still preserving the variance in that dataset. So, let's see how well that works and actually write some Python code to make PCA happen on the Iris dataset.

So, those were the concepts of dimensionality reduction, principal component analysis, and singular value decomposition. All big fancy words and yeah, it is kind of a fancy thing. You know, we're dealing with reducing higher dimensional spaces down to smaller dimensional spaces in a way that preserves their variance. Fortunately, scikit-learn makes this extremely easy to do, like 3 lines of code is all you need to actually apply PCA. So let's make that happen!


### A PCA example with the Iris dataset

Let's apply principal component analysis to the Iris dataset. This is a 4D dataset that we're going to reduce down to 2 dimensions. We're going to see that we can actually still preserve most of the information in that dataset, even by throwing away half of the dimensions. It's pretty cool stuff, and it's pretty simple too. Let's dive in and do some principal component analysis and cure the curse of dimensionality. Go ahead and open up the `PCA.ipynb` file.

#### Open Notebook
The Notebook opens in a new browser window. You can create a new notebook or open a local one. Check out the local folder `work` for several notebooks. Open and run `PCA.ipynb` in the `work` folder.

You can also open the Jupyter Notebook at https://[[HOST_SUBDOMAIN]]-8888-[[KATACODA_HOST]].environments.katacoda.com/notebooks/work/PCA.ipynb

It's actually very easy to do using scikit-learn, as usual! Again, PCA is a dimensionality reduction technique. It sounds very science-fictiony, all this talk of higher dimensions. But, just to make it more concrete and real again, a common application is image compression. You can think of an image of a black and white picture, as 3 dimensions, where you have width, as your x-axis, and your y-axis of height, and each individual cell has some brightness value on a scale of 0 to 1, that is black or white, or some value in between. So, that would be 3D data; you have 2 spatial dimensions, and then a brightness and intensity dimension on top of that.

If you were to distill that down to say 2 dimensions alone, that would be a compressed image and, if you were to do that in a technique that preserved the variance in that image as well as possible, you could still reconstruct the image, without a whole lot of loss in theory. So, that's dimensionality reduction, distilled down to a practical example.

Now, we're going to use a different example here using the Iris dataset, and scikit-learn includes this. All it is is a dataset of various Iris flower measurements, and the species classification for each Iris in that dataset. And it has also, like I said before, the length and width measurement of both the petal and the sepal for each Iris specimen. So, between the length and width of the petal, and the length and width of the sepal we have 4 dimensions of feature data in our dataset.


We want to distill that down to something we can actually look at and understand, because your mind doesn't deal with 4 dimensions very well, but you can look at 2 dimensions on a piece of paper pretty easily. Let's go ahead and load that up:

```
from sklearn.datasets import load_iris 
from sklearn.decomposition import PCA 
import pylab as pl 
from itertools import cycle 
 
iris = load_iris() 
 
numSamples, numFeatures = iris.data.shape 
print numSamples 
print numFeatures 
print list(iris.target_names) 
```

There's a handy dandy load_iris() function built into scikit-learn that will just load that up for you with no additional work; so you can just focus on the interesting part. Let's take a look at what that dataset looks like, the output of the preceding code is as follows:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-07/steps/14/1.png)

You can see that we are extracting the shape of that dataset, which means how many data points we have in it, that is 150, and how many features, or how many dimensions that dataset has, and that is 4. So, we have 150 Iris specimens in our dataset, with 4 dimensions of information. Again, that is the length and width of the sepal, and the length and width of the petal, for a total of 4 features, which we can think of as 4 dimensions.

We can also print out the list of target names in this dataset, which are the classifications,and we can see that each Iris belongs to one of three different species: Setosa, Versicolor, or Virginica. That's the data that we're working with: 150 Iris specimens, classified into one of 3 species, and we have 4 features associated with each Iris.

Let's look at how easy PCA is. Even though it's a very complicated technique under the hood, doing it is just a few lines of code. We're going to assign the entire Iris dataset and we're going to call it X. We will then create a PCA model, and we're going to keep n_components=2, because we want 2 dimensions, that is, we're going to go from 4 to 2.

We're going to use whiten=True, that means that we're going to normalize all the data, and make sure that everything is nice and comparable. Normally you will want to do that to get good results. Then, we will fit the PCA model to our Iris dataset X. We can use that model then, to transform that dataset down to 2 dimensions. Let's go ahead and run that. It happened pretty quickly!

```
X = iris.data 
pca = PCA(n_components=2, whiten=True).fit(X) 
X_pca = pca.transform(X) 
```

Please think about what just happened there. We actually created a PCA model to reduce 4 dimensions down to 2, and it did that by choosing 2 4D vectors, to create hyperplanes around, to project that 4D data down to 2 dimensions. You can actually see what those 4D vectors are, those eigenvectors, by printing out the actual components of PCA. So, PCA stands for Principal Component Analysis, those principal components are the eigenvectors that we chose to define our planes about:

```
print pca.components_ 
```

Output to the preceding code is as follows:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-07/steps/14/2.png)

You can actually look at those values, they're not going to mean a lot to you, because you can't really picture 4 dimensions anyway, but we did that just so you can see that it's actually doing something with principal components. So, let's evaluate our results:

```
print pca.explained_variance_ratio_ 
print sum(pca.explained_variance_ratio_) 
```

The PCA model gives us back something called explained_variance_ratio. Basically, that tells you how much of the variance in the original 4D data was preserved as I reduced it down to 2 dimensions. So, let's go ahead and take a look at that:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-07/steps/14/3.png)

What it gives you back is actually a list of 2 items for the 2 dimensions that we preserved. This is telling me that in the first dimension I can actually preserve 92% of the variance in the data, and the second dimension only gave me an additional 5% of variance. If I sum it together, these 2 dimensions that I projected my data down into, I still preserved over 97% of the variance in the source data. We can see that 4 dimensions weren't really necessary to capture all the information in this dataset, which is pretty interesting. It's pretty cool stuff!

If you think about it, why do you think that might be? Well, maybe the overall size of the flower has some relationship to the species at its center. Maybe it's the ratio of length to width for the petal and the sepal. You know, some of these things probably move together in concert with each other for a given species, or for a given overall size of a flower. So, perhaps there are relationships between these 4 dimensions that PCA is extracting on its own. It's pretty cool, and pretty powerful stuff. Let's go ahead and visualize this.

The whole point of reducing this down to 2 dimensions was so that we could make a nice little 2D scatter plot of it, at least that's our objective for this little example here. So, we're going to do a little bit of Matplotlib magic here to do that. There is some sort of fancy stuff going on here that I should at least mention. So, what we're going to do is create a list of colors: red, green and blue. We're going to create a list of target IDs, so that the values 0, 1, and 2 map to the different Iris species that we have.

What we're going to do is zip all this up with the actual names of each species. The for loop will iterate through the 3 different Iris species, and as it does that, we're going to have the index for that species, a color associated with it, and the actual human-readable name for that species. We'll take one species at a time and plot it on our scatter plot just for that species with a given color and the given label. We will then add in our legend and show the results:

```
colors = cycle('rgb') 
target_ids = range(len(iris.target_names)) 
pl.figure() 
for i, c, label in zip(target_ids, colors, iris.target_names): 
    pl.scatter(X_pca[iris.target == i, 0], X_pca[iris.target == i, 1], 
        c=c, label=label) 
pl.legend() 
pl.show()
```

The following is what we end up with:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-07/steps/14/4.png)

That is our 4D Iris data projected down to 2 dimensions. Pretty interesting stuff! You can see it still clusters together pretty nicely. You know, you have all the Virginicas sitting together, all the Versicolors sitting in the middle, and the Setosas way off on the left side. It's really hard to imagine what these actual values represent. But, the important point is, we've projected 4D data down to 2D, and in such a way that we still preserve the variance. We can still see clear delineations between these 3 species. A little bit of intermingling going on in there, it's not perfect you know. But by and large, it was pretty effective.

### Activity

As you recall from explained_variance_ratio, we actually captured most of the variance in a single dimension. Maybe the overall size of the flower is all that really matters in classifying it; and you can specify that with one feature. So, go ahead and modify the results if you are feeling up to it. See if you can get away with 2 dimensions, or 1 dimension instead of 2! So, go change that n_components to 1, and see what kind of variance ratio you get.

What happens? Does it makes sense? Play around with it, get some familiarity with it. That is dimensionality reduction, principal component analysis, and singular value decomposition all in action. Very, very fancy terms, and you know, to be fair it is some pretty fancy math under the hood. But as you can see, it's a very powerful technique and with scikit-learn, it's not hard to apply. So, keep that in your tool chest.

And there you have it! A 4D dataset of flower information boiled down to 2 dimensions that we can both easily visualize, and also still see clear delineations between the classifications that we're interested in. So, PCA works really well in this example. Again, it's a useful tool for things like compression, or feature extraction, or facial recognition as well. So, keep that in your toolbox.

### Data warehousing overview

Next, we're going to talk a little bit about data warehousing. This is a field that's really been upended recently by the advent of Hadoop, and some big data techniques and cloud computing. So, a lot of big buzz words there, but concepts that are important for you to understand.

Let's dive in and explore these concepts! Let's talk about ELT and ETL, and data warehousing in general. This is more of a concept, as opposed to a specific practical technique, so we're going to talk about it conceptually. But, it is something that's likely to come up in the setting of a job interview. So, let's make sure you understand these concepts.

We'll start by talking about data warehousing in general. What is a data warehouse? Well, it's basically a giant database that contains information from many different sources and ties them together for you. For example, maybe you work at a big ecommerce company and they might have an ordering system that feeds information about the stuff people bought into your data warehouse.

You might also have information from web server logs that get ingested into the data warehouse. This would allow you to tie together browsing information on the website with what people ultimately ordered for example. Maybe you could also tie in information from your customer service systems, and measure if there's a relationship between browsing behavior and how happy the customers are at the end of the day.

A data warehouse has the challenge of taking data from many different sources, transforming them into some sort of schema that allows us to query these different data sources simultaneously, and it lets us make insights, through data analysis. So, large corporations and organizations have this sort of thing pretty commonly. We're going into the concept of big data here. You can have a giant Oracle database, for example, that contains all this stuff and maybe it's partitioned in some way, and replicated and it has all sorts of complexity. You can just query that through SQL, structured query language, or, through graphical tools, like Tableau which is a very popular one these days. That's what a data analyst does, they query large datasets using stuff like Tableau.

That's kind of the difference between a data analyst and a data scientist. You might be actually writing code to perform more advanced techniques on data that border on AI, as opposed to just using tools to extract graphs and relationships out of a data warehouse. It's a very complicated problem. At Amazon, we had an entire department for data warehousing that took care of this stuff full time, and they never had enough people, I can tell you that; it's a big job!

You know, there are a lot of challenges in doing data warehousing. One is data normalization: so, you have to figure out how do all the fields in these different data sources actually relate to each other? How do I actually make sure that a column in one data source is comparable to a column from another data source and has the same set of data, at the same scale, using the same terminology? How do I deal with missing data? How do I deal with corrupt data or data from outliers, or from robots and things like that? These are all very big challenges. Maintaining those data feeds is also a very big problem.

A lot can go wrong when you're importing all this information into your data warehouse, especially when you have a very large transformation that needs to happen to take the raw data, saved from web logs, into an actual structured database table that can be imported into your data warehouse. Scaling also can get tricky when you're dealing with a monolithic data warehouse. Eventually, your data will get so large that those transformations themselves start to become a problem. This starts to get into the whole topic of ELT versus ETL thing.


### ETL versus ELT

Let's first talk about ETL. What does that stand for? It stands for extract, transform, and load - and that's sort of the conventional way of doing data warehousing.

Basically, first you extract the data that you want from the operational systems that you want. So, for example, I might extract all of the web logs from our web servers each day. Then I need to transform all that information into an actual structured database table that I can import into my data warehouse.

This transformation stage might go through every line of those web server logs, transform that into an actual table, where I'm plucking out from each web log line things like session ID, what page they looked at, what time it was, what the referrer was and things like that, and I can organize that into a tabular structure that I can then load into the data warehouse itself, as an actual table in a database. So, as data becomes larger and larger, that transformation step can become a real problem. Think about how much processing work is required to go through all of the web logs on Google, or Amazon, or any large website, and transform that into something a database can ingest. That itself becomes a scalability challenge and something that can introduce stability problems through the entire data warehouse pipeline.

That's where the concept of ELT comes in, and it kind of flips everything on its head. It says, "Well, what if we don't use a huge Oracle instance? What if instead we use some of these newer techniques that allow us to have a more distributed database over a Hadoop cluster that lets us take the power of these distributed databases like Hive, or Spark, or MapReduce, and use that to actually do the transformation after it's been loaded"

The idea here is we're going to extract the information we want as we did before, say from a set of web server logs. But then, we're going to load that straight in to our data repository, and we're going to use the power of the repository itself to actually do the transformation in place. So, the idea here is, instead of doing an offline process to transform my web logs, as an example, into a structured format, I'm just going to suck those in as raw text files and go through them one line at a time, using the power of something like Hadoop, to actually transform those into a more structured format that I can then query across my entire data warehouse solution.


Things like Hive let you host a massive database on a Hadoop cluster. There's things like Spark SQL that let you also do queries in a very SQL-like data warehouse-like manner, on a data warehouse that is actually distributed on Hadoop cluster. There are also distributed NoSQL data stores that can be queried using Spark and MapReduce. The idea is that instead of using a monolithic database for a data warehouse, you're instead using something built on top of Hadoop, or some sort of a cluster, that can actually not only scale up the processing and querying of that data, but also scale the transformation of that data as well.

Once again, you first extract your raw data, but then we're going to load it into the data warehouse system itself as is. And, then use the power of the data warehouse, which might be built on Hadoop, to do that transformation as the third step. Then I can query things together. So, it's a very big project, very big topic. You know, data warehousing is an entire discipline in and of itself. We're going to talk about Spark some more in this book very soon, which is one way of handling this thing - there's something called Spark SQL in particular that's relevant.

The overall concept here is that if you move from a monolithic database built on Oracle or MySQL to one of these more modern distributed databases built on top of Hadoop, you can take that transform stage and actually do that after you've loaded in the raw data, as opposed to before. That can end up being simpler and more scalable, and taking advantage of the power of large computing clusters that are available today.

That's ETL versus ELT, the legacy way of doing it with a lot of clusters all over the place in cloud-based computing versus a way that makes sense today, when we do have large clouds of computing available to us for transforming large datasets. That's the concept.

ETL is kind of the old school way of doing it, you transform a bunch of data offline before importing it in and loading it into a giant data warehouse, monolithic database. But with today's techniques, with cloud-based databases, and Hadoop, and Hive, and Spark, and MapReduce, you can actually do it a little bit more efficiently and take the power of a cluster to actually do that transformation step after you've loaded the raw data into your data warehouse.

This is really changing the field and it's important that you know about it. Again, there's a lot more to learn on the subject, so I encourage you to explore more on this topic. But, that's the basic concept, and now you know what people are talking about when they talk about ETL versus ELT.

### Reinforcement learning

Our next topic's a fun one: reinforcement learning. We can actually use this idea with an example of Pac-Man. We can actually create a little intelligent Pac-Man agent that can play the game Pac-Man really well on its own. You'll be surprised how simple the technique is for building up the smarts behind this intelligent Pac-Man. Let's take a look!

So, the idea behind reinforcement learning is that you have some sort of agent, in this case Pac-Man, that explores some sort of space, and in our example that space will be the maze that Pac-Man is in. As it goes, it learns the value of different state changes within different conditions.

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-07/steps/19/1.png)

For example, in the preceding image, the state of Pac-Man might be defined by the fact that it has a ghost to the South, and a wall to the West, and empty spaces to the North and East, and that might define the current state of Pac-Man. The state changes it can take would be to move in a given direction. I can then learn the value of going in a certain direction. So, for example, if I were to move North, nothing would really happen, there's no real reward associated with that. But, if I were to move South I would be destroyed by the ghost, and that would be a negative value.


As I go and explore the entire space, I can build up a set of all the possible states that Pac-Man can be in, and the values associated with moving in a given direction in each one of those states, and that's reinforcement learning. And as it explores the whole space, it refines these reward values for a given state, and it can then use those stored reward values to choose the best decision to make given a current set of conditions. In addition to Pac-Man, there's also a game called Cat & Mouse that is an example that's used commonly that we'll look at later.

The benefit of this technique is that once you've explored the entire set of possible states that your agent can be in, you can very quickly have a very good performance when you run different iterations of this. So, you know, you can basically make an intelligent Pac-Man by running reinforcement learning and letting it explore the values of different decisions it can make in different states and then storing that information, to very quickly make the right decision given a future state that it sees in an unknown set of conditions.

### Q-learning

So, a very specific implementation of reinforcement learning is called Q-learning, and this formalizes what we just talked about a little bit more:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-07/steps/19/2.png)

So, we start off with a Q value of 0 for every possible state that Pac-Man could be in. And, as Pac-Man explores a maze, as bad things happen to Pac-Man, we reduce the Q value for the state that Pac-Man was in at the time. So, if Pac-Man ends up getting eaten by a ghost, we penalize whatever he did in that current state. As good things happen to Pac-Man, as he eats a power pill, or eats a ghost, we'll increase the Q value for that action, for the state that he was in. Then, what we can do is use those Q values to inform Pac-Man's future choices, and sort of build a little intelligent agent that can perform optimally, and make a perfect little Pac-Man. From the same image of Pac-Man that we saw just above, we can further define the current state of Pac-Man by defining that he has a wall to the West, empty space to the North and East, a ghost to the South.

We can look at the actions he can take: he can't actually move left at all, but he can move up, down, or right, and we can assign a value to all those actions. By going up or right, nothing really happens at all, there's no power pill or dots to consume. But if he goes left, that's definitely a negative value. We can say for the state given by the current conditions that Pac-Man is surrounded by, moving down would be a really bad choice; there should be a negative Q value for that. Moving left just can't be done at all. Moving up or right or staying neutral, the Q value would remain 0 for those action choices for that given state.


Now, you can also look ahead a little bit, to make an even more intelligent agent. So, I'm actually two steps away from getting a power pill here. So, as Pac-Man were to explore this state, if I were to hit the case of eating that power pill on the next state, I could actually factor that into the Q value for the previous state. If you just have some sort of a discount factor, based on how far away you are in time, how many steps away you are, you can factor that all in together. So, that's a way of actually building in a little bit of memory into the system. You can "look ahead" more than one step by using a discount factor when computing Q (here s is previous state, s' is current state):

```
Q(s,a) += discount * (reward(s,a) + max(Q(s')) - Q(s,a))
```

So, the Q value that I experience when I consume that power pill might actually give a boost to the previous Q values that I encountered along the way. So, that's a way to make Q-learning even better.

### The exploration problem

One problem that we have in reinforcement learning is the exploration problem. How do I make sure that I efficiently cover all the different states and actions within those states during the exploration phase?

**The simple approach**

One simple approach is to always choose the action for a given state with the highest Q value that I've computed so far, and if there's a tie, just choose at random. So, initially all of my Q values might be 0, and I'll just pick actions at random at first.

As I start to gain information about better Q values for given actions and given states, I'll start to use those as I go. But, that ends up being pretty inefficient, and I can actually miss a lot of paths that way if I just tie myself into this rigid algorithm of always choosing the best Q value that I've computed thus far.

**The better way**

So, a better way is to introduce a little bit of random variation into my actions as I'm exploring. So, we call that an epsilon term. So, suppose we have some value, that I roll the dice, I have a random number. If it ends up being less than this epsilon value, I don't actually follow the highest Q value; I don't do the thing that makes sense, I just take a path at random to try it out, and see what happens. That actually lets me explore a much wider range of possibilities, a much wider range of actions, for a wider range of states more efficiently during that exploration stage.

So, what we just did can be described in very fancy mathematical terms, but you know conceptually it's pretty simple.

### Fancy words

I explore some set of actions that I can take for a given set of states, I use that to inform the rewards associated with a given action for a given set of states, and after that exploration is done I can use that information, those Q values, to intelligently navigate through an entirely new maze for example.

This can also be called a Markov decision process. So again, a lot of data science is just assigning fancy, intimidating names to simple concepts, and there's a ton of that in reinforcement learning.

**Markov decision process**

So, if you look up the definition of Markov decision processes, it is "a mathematical framework for modeling decision making in situations where outcomes are partly random and partly under the control of a decision maker".

- **Decision making:** What action do we take given a set of possibilities for a given state?
- **In situations where outcomes are partly random:** Hmm, kind of like our random exploration there.
- **Partly under the control of a decision maker:** The decision maker is our Q values that we computed.

So, MDPs, Markov decision processes, are a fancy way of describing our exploration algorithm that we just described for reinforcement learning. The notation is even similar, states are still described as s, and s' is the next state that we encounter. We have state transition functions that are defined as Pa for a given state of s and s'. We have our Q values, which are basically represented as a reward function, an Ra value for a given s and s'. So, moving from one state to another has a given reward associated with it, and moving from one state to another is defined by a state transition function:

- States are still described as **s** and **s''**
- State transition functions are described as **Pa(s,s')**
- Our Q values are described as a reward function **Ra(s,s')**

So again, describing what we just did, only a mathematical notation, and a fancier sounding word, Markov decision processes. And, if you want to sound even smarter, you can also call a Markov decision process by another name: a discrete time stochastic control process. That sounds intelligent! But the concept itself is the same thing that we just described.

### Dynamic programming


So, even more fancy words: dynamic programming can be used to describe what we just did as well. Wow! That sounds like artificial intelligence, computers programming themselves, Terminator 2, Skynet stuff. But no, it's just what we just did. If you look up the definition of dynamic programming, it is a method for solving a complex problem by breaking it down into a collection of simpler subproblems, solving each of those subproblems just once, and storing their solutions ideally, using a memory-based data structure.

The next time the same subproblem occurs, instead of recomputing its solution, one simply looks up the previously computed solution thereby saving computation time at the expense of a (hopefully) modest expenditure in storage space:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-07/steps/19/3.png)

We have a complicated exploration phase that finds the optimal rewards associated with each action for a given state. Once we have that table of the right action to take for a given state, we can very quickly use that to make our Pac-Man move in an optimal manner in a whole new maze that he hasn't seen before. So, reinforcement learning is also a form of dynamic programming. Wow!

To recap, you can make an intelligent Pac-Man agent by just having it semi-randomly explore different choices of movement given different conditions, where those choices are actions and those conditions are states. We keep track of the reward or penalty associated with each action or state as we go, and we can actually discount, going back multiple steps if you want to make it even better.

Then we store those Q values that we end up associating with each state, and we can use that to inform its future choices. So we can go into a whole new maze, and have a really smart Pac-Man that can avoid the ghosts and eat them up pretty effectively, all on its own. It's a pretty simple concept, very powerful though. You can also say that you understand a bunch of fancy terms because it's all called the same thing. Q-learning, reinforcement learning, Markov decision processes, dynamic programming: all tied up in the same concept.

I don't know, I think it's pretty cool that you can actually make sort of an artificially intelligent Pac-Man through such a simple technique, and it really does work! If you want to go look at it in more detail, following are a few examples you can look at that have one actual source code you can look at, and potentially play with, Python Markov Decision Process Toolbox: http://pymdptoolbox.readthedocs.org/en/latest/api/mdp.html.

There is a Python Markov decision process toolbox that wraps it up in all that terminology we talked about. There's an example you can look at, a working example of the cat and mouse game, which is similar. And, there is actually a Pac-Man example you can look at online as well, that ties in more directly with what we were talking about. Feel free to explore these links, and learn even more about it.

And so that's reinforcement learning. More generally, it's a useful technique for building an agent that can navigate its way through a possible different set of states that have a set of actions that can be associated with each state. So, we've talked about it mostly in the context of a maze game. But, you can think more broadly, and you know whenever you have a situation where you need to predict behavior of something given a set of current conditions and a set of actions it can take. Reinforcement learning and Q-learning might be a way of doing it. So, keep that in mind!

