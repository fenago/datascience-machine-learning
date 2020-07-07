<img align="right" src="../images/logo-small.png">


Lab : Machine Learning with Python - Part 2
-------------------------------------


In this scenario, we get into machine learning and how to actually implement machine learning models in Python.

We'll also cover a really interesting application of machine learning called decision trees and we'll build a working example in Python that predict shiring decisions in a company. Finally, we'll walk through the fascinating concepts of ensemble learning and SVMs, which are some of my favourite machine learning areas!

More specifically, we'll cover the following topics:

- Concept of K-means clustering
- Example of clustering in Python
- Entropy and how to measure it

#### Pre-reqs:
- Google Chrome (Recommended)

#### Lab Environment
Notebooks are ready to run. All packages have been installed. There is no requirement for any setup.

**Note:** Elev8ed Notebooks (powered by Jupyter) will be accessible at the port given to you by your instructor. Password for jupyterLab : `1234`

All Notebooks are present in `work/datascience-machine-learning` folder.

You can access jupyter lab at `<host-ip>:<port>/lab/workspaces/lab_


### K-Means clustering

Next, we're going to talk about k-means clustering, and this is an unsupervised learning technique where you have a collection of stuff that you want to group together into various clusters. Maybe it's movie genres or demographics of people, who knows? But it's actually a pretty simple idea, so let's see how it works.

K-means clustering is a very common technique in machine learning where you just try to take a bunch of data and find interesting clusters of things just based on the attributes of the data itself. Sounds fancy, but it's actually pretty simple. All we do in k-means clustering is try to split our data into K groups - that's where the K comes from, it's how many different groups you're trying to split your data into - and it does this by finding K centroids.

So, basically, what group a given data point belongs to is defined by which of these centroid points it's closest to in your scatter plot. You can visualize this in the following image:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-05-02/steps/3/1.jpg)

This is showing an example of k-means clustering with K of three, and the squares represent data points in a scatter plot. The circles represent the centroids that the k-means clustering algorithm came up with, and each point is assigned a cluster based on which centroid it's closest to. So that's all there is to it, really. It's an example of unsupervised learning. It isn't a case where we have a bunch of data and we already know the correct cluster for a given set of training data; rather, you're just given the data itself and it tries to converge on these clusters naturally just based on the attributes of the data alone. It's also an example where you are trying to find clusters or categorizations that you didn't even know were there. As with most unsupervised learning techniques, the point is to find latent values, things you didn't really realize were there until the algorithm showed them to you.


For example, where do millionaires live? I don't know, maybe there is some interesting geographical cluster where rich people tend to live, and k-means clustering could help you figure that out. Maybe I don't really know if today's genres of music are meaningful. What does it mean to be alternative these days? Not much, right? But by using k-means clustering on attributes of songs, maybe I could find interesting clusters of songs that are related to each other and come up with new names for what those clusters represent. Or maybe I can look at demographic data, and maybe existing stereotypes are no longer useful. Maybe Hispanic has lost its meaning and there's actually other attributes that define groups of people, for example, that I could uncover with clustering. Sounds fancy, doesn't it? Really complicated stuff. Unsupervised machine learning with K clusters, it sounds fancy, but as with most techniques in data science, it's actually a very simple idea.

Here's the algorithm for us in plain English:

Randomly pick K centroids (k-means): We start off with a randomly chosen set of centroids. So if we have a K of three we're going to look for three clusters in our group, and we will assign three randomly positioned centroids in our scatter plot.
Assign each data point to the centroid it is closest to: We then assign each data point to the randomly assigned centroid that it is closest to.
Recompute the centroids based on the average position of each centroid's points: Then recompute the centroid for each cluster that we come up with. That is, for a given cluster that we end up with, we will move that centroid to be the actual center of all those points.
Iterate until points stop changing assignment to centroids: We will do it all again until those centroids stop moving, we hit some threshold value that says OK, we have converged on something here.
Predict the cluster for new points: To predict the clusters for new points that I haven't seen before, we can just go through our centroid locations and figure out which centroid it's closest to to predict its cluster.

Let's look at a graphical example to make a little bit more sense. We'll call the first figure in the following image as A, second as B, third as C and the fourth as D.

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-05-02/steps/3/2.jpg)

The gray squares in image A represent data points in our scatter plot. The axes represent some different features of something. Maybe it's age and income; it's an example I keep using, but it could be anything. And the gray squares might represent individual people or individual songs or individual something that I want to find relationships between.

So I start off by just picking three points at random on my scatterplot. Could be anywhere. Got to start somewhere, right? The three points (centroids) I selected have been shown as circles in image A. So the next thing I'm going to do is for each centroid I'll compute which one of the gray points it's closest to. By doing that, the points shaded in blue are associated with this blue centroid. The green points are closest to the green centroid, and this single red point is closest to that red random point that I picked out.

Of course, you can see that's not really reflective of where the actual clusters appear to be. So what I'm going to do is take the points that ended up in each cluster and compute the actual center of those points. For example, in the green cluster, the actual center of all data turns out to be a little bit lower. We're going to move the centroid down a little bit. The red cluster only had one point, so its center moves down to where that single point is. And the blue point was actually pretty close to the center, so that just moves a little bit. On this next iteration we end up with something that looks like image D. Now you can see that our cluster for red things has grown a little bit and things have moved a little bit, that is, those got taken from the green cluster.

If we do that again, you can probably predict what's going to happen next. The green centroid will move a little bit, the blue centroid will still be about where it is. But at the end of the day you're going to end up with the clusters you'd probably expect to see. That's how k-means works. So it just keeps iterating, trying to find the right centroids until things start moving around and we converge on a solution.

### Limitations to k-means clustering

So there are some limitations to k-means clustering. Here they are:

- **Choosing K:** First of all, we need to choose the right value of K, and that's not a straightforward thing to do at all. The principal way of choosing K is to just start low and keep increasing the value of K depending on how many groups you want, until you stop getting large reductions in squared error. If you look at the distances from each point to their centroids, you can think of that as an error metric. At the point where you stop reducing that error metric, you know you probably have too many clusters. So you're not really gaining any more information by adding additional clusters at that point.
- **Avoiding local minima:** Also, there is a problem of local minima. You could just get very unlucky with those initial choices of centroids and they might end up just converging on local phenomena instead of more global clusters, so usually, you want to run this a few times and maybe average the results together. We call that ensemble learning. We'll talk about that more a little bit later on, but it's always a good idea to run k-means more than once using a different set of random initial values and just see if you do in fact end up with the same overall results or not.
- **Labeling the clusters:** Finally, the main problem with k-means clustering is that there's no labels for the clusters that you get. It will just tell you that this group of data points are somehow related, but you can't put a name on it. It can't tell you the actual meaning of that cluster. Let's say I have a bunch of movies that I'm looking at, and k-means clustering tells me that bunch of science fiction movies are over here, but it's not going to call them "science fiction" movies for me. It's up to me to actually dig into the data and figure out, well, what do these things really have in common? How might I describe that in English? That's the hard part, and k-means won't help you with that. So again, scikit-learn makes it very easy to do this.

Let's now work up an example and put k-means clustering into action.

### Clustering people based on income and age

Let's see just how easy it is to do k-means clustering using scikit-learn and Python.

#### Open Notebook
The Notebook opens in a new browser window. You can create a new notebook or open a local one. Check out the local folder `work` for several notebooks. Open and run `KMeans.ipynb` in the `work` folder.



We will discuss notebook in the next steps.

The first thing we're going to do is create some random data that we want to try to cluster. Just to make it easier, we'll actually build some clusters into our fake test data. So let's pretend there's some real fundamental relationship between these data, and there are some real natural clusters that exist in it.

So to do that, we can work with this little createClusteredData() function in Python:

```
from numpy import random, array 
 
#Create fake income/age clusters for N people in k clusters 
def createClusteredData(N, k): 
    random.seed(10) 
    pointsPerCluster = float(N)/k 
    X = [] 
    for i in range (k): 
        incomeCentroid = random.uniform(20000.0, 200000.0) 
        ageCentroid = random.uniform(20.0, 70.0) 
        for j in range(int(pointsPerCluster)): 
            X.append([random.normal(incomeCentroid, 10000.0), 
            random.normal(ageCentroid, 2.0)]) 
    X = array(X) 
    return X
```

The function starts off with a consistent random seed so you'll get the same result every time. We want to create clusters of N people in k clusters. So we pass N and k to createClusteredData().


Our code figures out how many points per cluster that works out to first and stores it in pointsPerCluster. Then, it builds up list X that starts off empty. For each cluster, we're going to create some random centroid of income (incomeCentroid) between 20,000 and 200,000 dollars and some random centroid of age (ageCentroid) between the age of 20 and 70.

What we're doing here is creating a fake scatter plot that will show income versus age for N people and k clusters. So for each random centroid that we created, I'm then going to create a normally distributed set of random data with a standard deviation of 10,000 in income and a standard deviation of 2 in age. That will give us back a bunch of age income data that is clustered into some pre-existing clusters that we can chose at random. OK, let's go ahead and run that.

Now, to actually do k-means, you'll see how easy it is.

```
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import scale 
from numpy import random, float 
 
data = createClusteredData(100, 5) 
 
model = KMeans(n_clusters=5) 
 
# Note I'm scaling the data to normalize it! Important for good results. 
model = model.fit(scale(data)) 
 
# We can look at the clusters each data point was assigned to 
print model.labels_  
 
# And we'll visualize it: 
plt.figure(figsize=(8, 6)) 
plt.scatter(data[:,0], data[:,1], c=model.labels_.astype(float)) 
plt.show() 
```

All you need to do is import KMeans from scikit-learn's cluster package. We're also going to import matplotlib so we can visualize things, and also import scale so we can take a look at how that works.

So we use our createClusteredData() function to say 100 random people around 5 clusters. So there are 5 natural clusters for the data that I'm creating. We then create a model, a KMeans model with k of 5, so we're picking 5 clusters because we know that's the right answer. But again, in unsupervised learning you don't necessarily know what the real value of k is. You need to iterate and converge on it yourself. And then we just call model.fit using my KMeans model using the data that we had.


Now the scale I alluded to earlier, that's normalizing the data. One important thing with k-means is that it works best if your data is all normalized. That means everything is at the same scale. So a problem that I have here is that my ages range from 20 to 70, but my incomes range all the way up to 200,000. So these values are not really comparable. The incomes are much larger than the age values. Scale will take all that data and scale it together to a consistent scale so I can actually compare these things as apples to apples, and that will help a lot with your k-means results.

So, once we've actually called fit on our model, we can actually look at the resulting labels that we got. Then we can actually visualize it using a little bit of matplotlib magic. You can see in the code we have a little trick where we assigned the color to the labels that we ended up with converted to some floating point number. That's just a little trick you can use to assign arbitrary colors to a given value. So let's see what we end up with:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-05-02/steps/8/1.jpg)

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-05-02/steps/8/2.jpg)



It didn't take that long. You see the results are basically what clusters I assigned everything into. We know that our fake data is already pre-clustered, so it seems that it identified the first and second clusters pretty easily. It got a little bit confused beyond that point, though, because our clusters in the middle are actually a little bit mushed together. They're not really that distinct, so that was a challenge for k-means. But regardless, it did come up with some reasonable guesses at the clusters. This is probably an example of where four clusters would more naturally fit the data.

### Activity

So what I want you to do for an activity is to try a different value of k and see what you end up with. Just eyeballing the preceding graph, it looks like four would work well. Does it really? What happens if I increase k too large? What happens to my results? What does it try to split things into, and does it even make sense? So, play around with it, try different values of k. So in the n_clusters() function, change the 5 to something else. Run all through it again and see you end up with.

That's all there is to k-means clustering. It's just that simple. You can just use scikit-learn's KMeans thing from cluster. The only real gotcha: make sure you scale the data, normalize it. You want to make sure the things that you're using k-means on are comparable to each other, and the scale() function will do that for you. So those are the main things for k-means clustering. Pretty simple concept, even simpler to do it using scikit-learn.

That's all there is to it. That's k-means clustering. So if you have a bunch of data that is unclassified and you don't really have the right answers ahead of time, it's a good way to try to naturally find interesting groupings of your data, and maybe that can give you some insight into what that data is. It's a good tool to have. I've used it before in the real world and it's really not that hard to use, so keep that in your tool chest.

### Measuring entropy

Quite soon we're going to get to one of the cooler parts of machine learning, at least I think so, called decision trees. But before we can talk about that, it's a necessary to understand the concept of entropy in data science.

So entropy, just like it is in physics and thermodynamics, is a measure of a dataset's disorder, of how same or different the dataset is. So imagine we have a dataset of different classifications, for example, animals. Let's say I have a bunch of animals that I have classified by species. Now, if all of the animals in my dataset are an iguana, I have very low entropy because they're all the same. But if every animal in my dataset is a different animal, I have iguanas and pigs and sloths and who knows what else, then I would have a higher entropy because there's more disorder in my dataset. Things are more different than they are the same.

Entropy is just a way of quantifying that sameness or difference throughout my data. So, an entropy of 0 implies all the classes in the data are the same, whereas if everything is different, I would have a high entropy, and something in between would be a number in between. Entropy just describes how same or different the things in a dataset are.


Now mathematically, it's a little bit more involved than that, so when I actually compute a number for entropy, it's computed using the following expression:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-05-02/steps/12/1.jpg)

So for every different class that I have in my data, I'm going to have one of these p terms, p1, p2, and so on and so forth through pn, for n different classes that I might have. The p just represents the proportion of the data that is that class. And if you actually plot what this looks like for each term- pi* ln * pi, it'll look a little bit something like the following graph:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-05-02/steps/12/2.jpg)

You add these up for each individual class. For example, if the proportion of the data, that is, for a given class is 0, then the contribution to the overall entropy is 0. And if everything is that class, then again the contribution to the overall entropy is 0 because in either case, if nothing is this class or everything is this class, that's not really contributing anything to the overall entropy.

It's the things in the middle that contribute entropy of the class, where there's some mixture of this classification and other stuff. When you add all these terms together, you end up with an overall entropy for the entire dataset. So mathematically, that's how it works out, but again, the concept is very simple. It's just a measure of how disordered your dataset, how same or different the things in your data are.

### Decision trees - Concepts

Believe it or not, given a set of training data, you can actually get Python to generate a flowchart for you to make a decision. So if you have something you're trying to predict on some classification, you can use a decision tree to actually look at multiple attributes that you can decide upon at each level in the flowchart. You can print out an actual flowchart for you to use to make a decision from, based on actual machine learning. How cool is that? Let's see how it works.

I personally find decision trees are one of the most interesting applications of machine learning. A decision tree basically gives you a flowchart of how to make some decision.You have some dependent variable, like whether or not I should go play outside today or not based on the weather. When you have a decision like that that depends on multiple attributes or multiple variables, a decision tree could be a good choice.

There are many different aspects of the weather that might influence my decision of whether I should go outside and play. It might have to do with the humidity, the temperature, whether it's sunny or not, for example. A decision tree can look at all these different attributes of the weather, or anything else, and decide what are the thresholds? What are the decisions I need to make on each one of those attributes before I arrive at a decision of whether or not I should go play outside? That's all a decision tree is. So it's a form of supervised learning.


The way it would work in this example would be as follows. I would have some sort of dataset of historical weather, and data about whether or not people went outside to play on a particular day. I would feed the model this data of whether it was sunny or not on each day, what the humidity was, and if it was windy or not; and whether or not it was a good day to go play outside. Given that training data, a decision tree algorithm can then arrive at a tree that gives us a flowchart that we can print out. It looks just like the following flow chart. You can just walk through and figure out whether or not it's a good day to play outside based on the current attributes. You can use that to predict the decision for a new set of values:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-05-02/steps/14/1.jpg)

How cool is that? We have an algorithm that will make a flowchart for you automatically just based on observational data. What's even cooler is how simple it all works once you learn how it works.

### Decision tree example

Let's say I want to build a system that will automatically filter out resumes based on the information in them. A big problem that technology companies have is that we get tons and tons of resumes for our positions. We have to decide who we actually bring in for an interview, because it can be expensive to fly somebody out and actually take the time out of the day to conduct an interview. So what if there were a way to actually take historical data on who actually got hired and map that to things that are found on their resume?

We could construct a decision tree that will let us go through an individual resume and say, "OK, this person actually has a high likelihood of getting hired, or not". We can train a decision tree on that historical data and walk through that for future candidates. Wouldn't that be a wonderful thing to have?

So let's make some totally fabricated hiring data that we're going to use in this example:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-05-02/steps/14/2.jpg)

In the preceding table, we have candidates that are just identified by numerical identifiers. I'm going to pick some attributes that I think might be interesting or helpful to predict whether or not they're a good hire or not. How many years of experience do they have? Are they currently employed? How many employers have they had previous to this one? What's their level of education? What degree do they have? Did they go to what we classify as a top-tier school? Did they do an internship while they were in college? We can take a look at this historical data, and the dependent variable here is Hired. Did this person actually get a job offer or not based on that information?

Now, obviously there's a lot of information that isn't in this model that might be very important, but the decision tree that we train from this data might actually be useful in doing an initial pass at weeding out some candidates. What we end up with might be a tree that looks like the following:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-05-02/steps/14/3.jpg)

So it just turns out that in my totally fabricated data, anyone that did an internship in college actually ended up getting a job offer. So my first decision point is "did this person do an internship or not?" If yes, go ahead and bring them in. In my experience, internships are actually a pretty good predictor of how good a person is. If they have the initiative to actually go out and do an internship, and actually learn something at that internship, that's a good sign.
Do they currently have a job? Well, if they are currently employed, in my very small fake dataset it turned out that they are worth hiring, just because somebody else thought they were worth hiring too. Obviously it would be a little bit more of a nuanced decision in the real world.
If they're not currently employed, do they have less than one prior employer? If yes, this person has never held a job and they never did an internship either. Probably not a good hire decision. Don't hire that person.
But if they did have a previous employer, did they at least go to a top-tier school? If not, it's kind of iffy. If so, then yes, we should hire this person based on the data that we trained on.
Walking through a decision tree
So that's how you walk through the results of a decision tree. It's just like going through a flowchart, and it's kind of awesome that an algorithm can produce this for you. The algorithm itself is actually very simple. Let me explain how the algorithm works.

At each step of the decision tree flowchart, we find the attribute that we can partition our data on that minimizes the entropy of the data at the next step. So we have a resulting set of classifications: in this case hire or don't hire, and we want to choose the attribute decision at that step that will minimize the entropy at the next step.

At each step we want to make all of the remaining choices result in either as many no hires or as many hire decisions as possible. We want to make that data more and more uniform so as we work our way down the flowchart, and we ultimately end up with a set of candidates that are either all hires or all no hires so we can classify into yes/no decisions on a decision tree. So we just walk down the tree, minimize entropy at each step by choosing the right attribute to decide on, and we keep on going until we run out.

There's a fancy name for this algorithm. It's called ID3 (Iterative Dichotomiser 3). It is what's known as a greedy algorithm. So as it goes down the tree, it just picks the attribute that will minimize entropy at that point. Now that might not actually result in an optimal tree that minimizes the number of choices that you have to make, but it will result in a tree that works, given the data that you gave it.

Random forests technique
Now one problem with decision trees is that they are very prone to overfitting, so you can end up with a decision tree that works beautifully for the data that you trained it on, but it might not be that great for actually predicting the correct classification for new people that it hasn't seen before. Decision trees are all about arriving at the right decision for the training data that you gave it, but maybe you didn't really take into account the right attributes, maybe you didn't give it enough of a representative sample of people to learn from. This can result in real problems.

So to combat this issue, we use a technique called random forests, where the idea is that we sample the data that we train on, in different ways, for multiple different decision trees. Each decision tree takes a different random sample from our set of training data and constructs a tree from it. Then each resulting tree can vote on the right result.

Now that technique of randomly resampling our data with the same model is a term called bootstrap aggregating, or bagging. This is a form of what we call ensemble learning, which we'll cover in more detail shortly. But the basic idea is that we have multiple trees, a forest of trees if you will, each that uses a random subsample of the data that we have to train on. Then each of these trees can vote on the final result, and that will help us combat overfitting for a given set of training data.

The other thing random forests can do is actually restrict the number of attributes that it can choose, between at each stage, while it is trying to minimize the entropy as it goes. And we can randomly pick which attributes it can choose from at each level. So that also gives us more variation from tree to tree, and therefore we get more of a variety of algorithms that can compete with each other. They can all vote on the final result using slightly different approaches to arriving at the same answer.

So that's how random forests work. Basically, it is a forest of decision trees where they are drawing from different samples and also different sets of attributes at each stage that it can choose between.

So, with all that, let's go make some decision trees. We'll use random forests as well when we're done, because scikit-learn makes it really really easy to do, as you'll see soon.

### Decision trees - Predicting hiring decisions using Python

Turns out that it's easy to make decision trees; in fact it's crazy just how easy it is, with just a few lines of Python code. So let's give it a try.

#### Open Notebook
The Notebook opens in a new browser window. You can create a new notebook or open a local one. Check out the local folder `work` for several notebooks. Open and run `DecisionTree.ipynb` in the `work` folder.




I've included a PastHires.csv file with your book materials, and that just includes some fabricated data, that I made up, about people that either got a job offer or not based on the attributes of those candidates.

```
import numpy as np 
import pandas as pd 
from sklearn import tree 
 
input_file = "./PastHires.csv" 
df = pd.read_csv(input_file, header = 0) 
```

We will use pandas to read our CSV in, and create a DataFrame object out of it. Let's go ahead and run our code, and we can use the head() function on the DataFrame to print out the first few lines and make sure that it looks like it makes sense.

```
df.head() 
```

Sure enough we have some valid data in the output:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-05-02/steps/18/1.jpg)

So, for each candidate ID, we have their years of past experience, whether or not they were employed, their number of previous employers, their highest level of education, whether they went to a top-tier school, and whether they did an internship; and finally here, in the Hired column, the answer - where we knew that we either extended a job offer to this person or not.


As usual, most of the work is just in massaging your data, preparing your data, before you actually run the algorithms on it, and that's what we need to do here. Now scikit-learn requires everything to be numerical, so we can't have Ys and Ns and BSs and MSs and PhDs. We have to convert all those things to numbers for the decision tree model to work. The way to do this is to use some short-hand in pandas, which makes these things easy. For example:

```
d = {'Y': 1, 'N': 0} 
df['Hired'] = df['Hired'].map(d) 
df['Employed?'] = df['Employed?'].map(d) 
df['Top-tier school'] = df['Top-tier school'].map(d) 
df['Interned'] = df['Interned'].map(d) 
d = {'BS': 0, 'MS': 1, 'PhD': 2} 
df['Level of Education'] = df['Level of Education'].map(d) 
df.head() 
```

Basically, we're making a dictionary in Python that maps the letter Y to the number 1, and the letter N to the value 0. So, we want to convert all our Ys to 1s and Ns to 0s. So 1 will mean yes and 0 will mean no. What we do is just take the Hired column from the DataFrame, and call map() on it, using a dictionary. This will go through the entire Hired column, in the entire DataFrame and use that dictionary lookup to transform all the entries in that column. It returns a new DataFrame column that I'm putting back into the Hired column. This replaces the Hired column with one that's been mapped to 1s and 0s.

We do the same thing for Employed, Top-tier school and Interned, so all those get mapped using the yes/no dictionary. So, the Ys and Ns become 1s and 0s instead. For the Level of Education, we do the same trick, we just create a dictionary that assigns BS to 0, MS to 1, and PhD to 2 and uses that to remap those degree names to actual numerical values. So if I go ahead and run that and do a head() again, you can see that it worked:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-05-02/steps/18/2.jpg)

All my yeses are 1's, my nos are 0's, and my Level of Education is now represented by a numerical value that has real meaning.


Next we need to prepare everything to actually go into our decision tree classifier, which isn't that hard. To do that, we need to separate our feature information, which are the attributes that we're trying to predict from, and our target column, which contains the thing that we're trying to predict.To extract the list of feature name columns, we are just going to create a list of columns up to number 6. We go ahead and print that out.

```
features = list(df.columns[:6]) 
features 
```

We get the following output:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-05-02/steps/18/3.jpg)

Above are the column names that contain our feature information: Years Experience, Employed?, Previous employers, Level of Education, Top-tier school, and Interned. These are the attributes of candidates that we want to predict hiring on.

Next, we construct our y vector which is assigned what we're trying to predict, that is our Hired column:

```
y = df["Hired"] 
X = df[features] 
clf = tree.DecisionTreeClassifier() 
clf = clf.fit(X,y) 
```

This code extracts the entire Hired column and calls it y. Then it takes all of our columns for the feature data and puts them in something called X. This is a collection of all of the data and all of the feature columns, and X and y are the two things that our decision tree classifier needs.

To actually create the classifier itself, two lines of code: we call tree.DecisionTreeClassifier() to create our classifier, and then we fit it to our feature data (X) and the answers (y)- whether or not people were hired. So, let's go ahead and run that.

Displaying graphical data is a little bit tricky, and I don't want to distract us too much with the details here, so please just consider the following boilerplate code. You don't need to get into how Graph viz works here - and dot files and all that stuff: it's not important to our journey right now. The code you need to actually display the end results of a decision tree is simply:

```
from IPython.display import Image   
from sklearn.externals.six import StringIO   
import pydot  
 
dot_data = StringIO()   
tree.export_graphviz(clf, out_file=dot_data,   
                         feature_names=features)   
graph = pydot.graph_from_dot_data(dot_data.getvalue())   
Image(graph.create_png()) 
```

So let's go ahead and run this.

This is what your output should now look like:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-05-02/steps/18/4.jpg)

There we have it! How cool is that?! We have an actual flow chart here.

Now, let me show you how to read it. At each stage, we have a decision. Remember most of our data which is yes or no, is going to be 0 or 1. So, the first decision point becomes: is Employed? less than 0.5? Meaning that if we have an employment value of 0, that is no, we're going to go left.If employment is 1, that is yes, we're going to go right.

So, were they previously employed? If not go left, if yes go right. It turns out that in my sample data, everyone who is currently employed actually got a job offer, so I can very quickly say if you are currently employed, yes, you're worth bringing in, we're going to follow down to the second level here.

So, how do you interpret this? The gini score is basically a measure of entropy that it's using at each step. Remember as we're going down the algorithm is trying to minimize the amount of entropy. And the samples are the remaining number of samples that haven't beensectioned off by a previous decision.

So say this person was employed. The way to read the right leaf node is the value column that tells you at this point we have 0 candidates that were no hires and 5 that were hires. So again, the way to interpret the first decision point is if Employed? was 1, I'm going to go to the right, meaning that they are currently employed, and this brings me to a world where everybody got a job offer. So, that means I should hire this person.

Now let's say that this person doesn't currently have a job. The next thing I'm going to look at is, do they have an internship. If yes, then we're at a point where in our training data everybody got a job offer. So, at that point, we can say our entropy is now 0 (gini=0.0000), because everyone's the same, and they all got an offer at that point. However, you know if we keep going down(where the person has not done an internship),we'll be at a point where the entropy is 0.32. It's getting lower and lower, that's a good thing.

Next we're going to look at how much experience they have, do they have less than one year of experience? And, if the case is that they do have some experience and they've gotten this far they're a pretty good no hire decision. We end up at the point where we have zero entropy but, all three remaining samples in our training set were no hires. We have 3 no hires and 0 hires. But, if they do have less experience, then they're probably fresh out of college, they still might be worth looking at.

The final thing we're going to look at is whether or not they went to a Top-tier school, and if so, they end up being a good prediction for being a hire. If not, they end up being a no hire. We end up with one candidate that fell into that category that was a no hire and 0 that were a hire. Whereas, in the case candidates did go to a top tier school, we have 0 no hires and 1 hire.

So, you can see we just keep going until we reach an entropy of 0, if at all possible, for every case.

### Ensemble learning – Using a random forest

Now, let's say we want to use a random forest, you know, we're worried that we might be over fitting our training data. It's actually very easy to create a random forest classifier of multiple decision trees.

So, to do that, we can use the same data that we created before. You just need your X and y vectors, that is the set of features and the column that you're trying to predict on:

```
from sklearn.ensemble import RandomForestClassifier 
 
clf = RandomForestClassifier(n_estimators=10) 
clf = clf.fit(X, y) 
 
#Predict employment of an employed 10-year veteran 
print clf.predict([[10, 1, 4, 0, 0, 0]]) 
#...and an unemployed 10-year veteran 
print clf.predict([[10, 0, 4, 0, 0, 0]]) 
```

We make a random forest classifier, also available from scikit-learn, and pass it the number of trees we want in our forest. So, we made ten trees in our random forest in the code above. We then fit that to the model.

You don't have to walk through the trees by hand, and when you're dealing with a random forest you can't really do that anyway. So, instead we use the predict() function on the model, that is on the classifier that we made. We pass in a list of all the different features for a given candidate that we want to predict employment for.

If you remember this maps to these columns: Years Experience, Employed?, Previous employers, Level of Education, Top-tier school, and Interned; interpreted as numerical values. We predict the employment of an employed 10-year veteran. We also predict the employment of an unemployed 10-year veteran. And, sure enough, we get a result:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-05-02/steps/18/5.jpg)

So, in this particular case, we ended up with a hire decision on both. But, what's interesting is there is a random component to that. You don't actually get the same result every time! More often than not, the unemployed person does not get a job offer, and if you keep running this you'll see that's usually the case. But, the random nature of bagging, of bootstrap aggregating each one of those trees, means you're not going to get the same result every time. So, maybe 10 isn't quite enough trees. So, anyway, that's a good lesson to learn here!


### Activity

For an activity, if you want to go back and play with this, mess around with my input data. Go ahead and edit the code we've been exploring, and create an alternate universe where it's a topsy turvy world; for example, everyone that I gave a job offer to now doesn't get one and vice versa. See what that does to your decision tree. Just mess around with it and see what you can do and try to interpret the results.

So, that's decision trees and random forests, one of the more interesting bits of machine learning, in my opinion. I always think it's pretty cool to just generate a flowchart out of thin air like that. So, hopefully you'll find that useful.