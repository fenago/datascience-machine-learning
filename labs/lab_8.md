<img align="right" src="../images/logo-small.png">


Lab : Machine Learning with Python - Part 2
-------------------------------------


In this scenario, we'll cover the following topics:

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



Let's now work up an example and put k-means clustering into action.

### Clustering people based on income and age

Let's see just how easy it is to do k-means clustering using scikit-learn and Python.

#### Open Notebook
The Notebook opens in a new browser window. You can create a new notebook or open a local one. Check out the local folder `work` for several notebooks. Open and run `KMeans.ipynb` in the `work` folder.

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




### Activity

So what I want you to do for an activity is to try a different value of k and see what you end up with. Just eyeballing the preceding graph, it looks like four would work well. Does it really? What happens if I increase k too large? What happens to my results? What does it try to split things into, and does it even make sense? So, play around with it, try different values of k. So in the n_clusters() function, change the 5 to something else. Run all through it again and see you end up with.

That's all there is to k-means clustering. It's just that simple. You can just use scikit-learn's KMeans thing from cluster. The only real gotcha: make sure you scale the data, normalize it. You want to make sure the things that you're using k-means on are comparable to each other, and the scale() function will do that for you. So those are the main things for k-means clustering. Pretty simple concept, even simpler to do it using scikit-learn.



### Decision tree example

Let's say I want to build a system that will automatically filter out resumes based on the information in them. A big problem that technology companies have is that we get tons and tons of resumes for our positions. We have to decide who we actually bring in for an interview, because it can be expensive to fly somebody out and actually take the time out of the day to conduct an interview. So what if there were a way to actually take historical data on who actually got hired and map that to things that are found on their resume?

We could construct a decision tree that will let us go through an individual resume and say, "OK, this person actually has a high likelihood of getting hired, or not". We can train a decision tree on that historical data and walk through that for future candidates. Wouldn't that be a wonderful thing to have?

So let's make some totally fabricated hiring data that we're going to use in this example:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-05-02/steps/14/2.jpg)

In the preceding table, we have candidates that are just identified by numerical identifiers. I'm going to pick some attributes that I think might be interesting or helpful to predict whether or not they're a good hire or not. How many years of experience do they have? Are they currently employed? How many employers have they had previous to this one? What's their level of education? What degree do they have? Did they go to what we classify as a top-tier school? Did they do an internship while they were in college? We can take a look at this historical data, and the dependent variable here is Hired. Did this person actually get a job offer or not based on that information?

Now, obviously there's a lot of information that isn't in this model that might be very important, but the decision tree that we train from this data might actually be useful in doing an initial pass at weeding out some candidates. What we end up with might be a tree that looks like the following:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-05-02/steps/14/3.jpg)


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
