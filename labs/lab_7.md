### Introduction

In this scenario, we get into machine learning and how to actually implement machine learning models in Python.

We'll examine what supervised and unsupervised learning means, and how they're different from each other. We'll see techniques to prevent overfitting, and then look at an interesting example where we implement a spam classifier. We'll analyze what K-Means clustering is a long the way, with a working example that clusters people based on their income and age using scikit-learn!

More specifically, we'll cover the following topics:

- Supervised and unsupervised learning
- Avoiding overfitting by using train/test
- Bayesian methods
- Implementation of an e-mail spam classifier with Naïve Bayes

### Jupyter Notebooks

We will run Jupyter Notebook as a Docker container. This setup will take some time because of the size of the image.

## Login
When the container is running, execute this statement:
`docker logs jupyter 2>&1 | grep -v "HEAD" `


This will show something like:
```
copy/paste this URL into your browser when you connect for the first time, to login with a token:
    http://localhost:8888/?token=f89b02dd78479d52470b3c3a797408b20cc5a11e067e94b8
    THIS IS NOT YOUR TOKEN.  YOU HAVE TO SEARCH THE LOGS TO GET YOUR TOKEN
```

The token is the value behind `/?token=`. You need that for logging in.

**Note:** You can also run following command to get token directly: 
`docker exec -it jupyter bash -c 'jupyter notebook list' | cut -d'=' -f 2 | cut -d' ' -f 1`

Next, you can open the Jupyter Notebook at 
 https://[[HOST_SUBDOMAIN]]-8888-[[KATACODA_HOST]].environments.katacoda.com/

 ### Machine learning and train/test

 So what is machine learning? Well, if you look it up on Wikipedia or whatever, it'll say that it is algorithms that can learn from observational data and can make predictions based on it. It sounds really fancy, right? Like artificial intelligence stuff, like you have a throbbing brain inside of your computer. But in reality, these techniques are usually very simple.

We've already looked at regressions, where we took a set of observational data, we fitted a line to it, and we used that line to make predictions. So by our new definition, that was machine learning! And your brain works that way too.

Another fundamental concept in machine learning is something called train/test, which lets us very cleverly evaluate how good a machine learning model we've made. As we look now at unsupervised and supervised learning, you'll see why train/test is so important to machine learning.

### Unsupervised learning

Let's talk in detail now about two different types of machine learning: supervised and unsupervised learning. Sometimes there can be kind of a blurry line between the two, but the basic definition of unsupervised learning is that you're not giving your model any answers to learn from. You're just presenting it with a group of data and your machine learning algorithm tries to make sense out of it given no additional information:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-05-01/steps/4/1.jpg)

Let's say I give it a bunch of different objects, like these balls and cubes and sets of dice and what not. Let's then say have some algorithm that will cluster these objects into things that are similar to each other based on some similarity metric.

Now I haven't told the machine learning algorithm, ahead of time, what categories certain objects belong to. I don't have a cheat sheet that it can learn from where I have a set of existing objects and my correct categorization of it. The machine learning algorithm must infer those categories on its own. This is an example of unsupervised learning, where I don't have a set of answers that I'm getting it learn from. I'm just trying to let the algorithm gather its own answers based on the data presented to it alone.

The problem with this is that we don't necessarily know what the algorithm will come up with! If I gave it that bunch of objects shown in the preceding image, is it going to group things into things that are round, things that are large versus small, things that are red versus blue, I don't know. It's going to depend on the metric that I give it for similarity between items primarily. But sometimes you'll find clusters that are surprising,and emerged that you didn't expect to see.


So that's really the point of unsupervised learning: if you don't know what you're looking for, it can be a powerful tool for discovering classifications that you didn't even know were there. We call this a latent variable. Some property of your data that you didn't even know was there originally, can be teased out by unsupervised learning.

Let's take another example around unsupervised learning. Say I was clustering people instead of balls and dice. I'm writing a dating site and I want to see what sorts of people tend to cluster together. There are some attributes that people tend to cluster around, which decide whether they tend to like each other and date each other for example. Now you might find that the clusters that emerge don't conform to your predisposed stereotypes. Maybe it's not about college students versus middle-aged people, or people who are divorced and whatnot, or their religious beliefs. Maybe if you look at the clusters that actually emerged from that analysis, you'll learn something new about your users and actually figure out that there's something more important than any of those existing features of your people that really count toward, to decide whether they like each other. So that's an example of supervised learning providing useful results.

Another example could be clustering movies based on their properties. If you were to run clustering on a set of movies from like IMDb or something, maybe the results would surprise you. Perhaps it's not just about the genre of the movie. Maybe there are other properties, like the age of the movie or the running length or what country it was released in, that are more important. You just never know. Or we could analyze the text of product descriptions and try to find the terms that carry the most meaning for a certain category. Again, we might not necessarily know ahead of time what terms, or what words, are most indicative of a product being in a certain category; but through unsupervised learning, we can tease out that latent information.

### Supervised learning

Now in contrast, supervised learning is a case where we have a set of answers that the model can learn from. We give it a set of training data, that the model learns from. It can then infer relationships between the features and the categories that we want, and apply that to unseen new values - and predict information about them.

Going back to our earlier example, where we were trying to predict car prices based on the attributes of those cars. That's an example where we are training our model using actual answers. So I have a set of known cars and their actual prices that they sold for. I train the model on that set of complete answers, and then I can create a model that I'm able to use to predict the prices of new cars that I haven't seen before. That's an example of supervised learning, where you're giving it a set of answers to learn from. You've already assigned categories or some organizing criteria to a set of data, and your algorithm then uses that criteria to build a model from which it can predict new values.

### Evaluating supervised learning

So how do you evaluate supervised learning? Well, the beautiful thing about supervised learning is that we can use a trick called train/test. The idea here is to split our observational data that I want my model to learn from into two groups, a training set and a testing set. So when I train/build my model based on the data that I have, I only do that with part of my data that I'm calling my training set, and I reserve another part of my data that I'm going to use for testing purposes.

I can build my model using a subset of my data for training data, and then I'm in a position to evaluate the model that comes out of that, and see if it can successfully predict the correct answers for my testing data.

So you see what I did there? I have a set of data where I already have the answers that I can train my model from, but I'm going to withhold a portion of that data and actually use that to test my model that was generated using the training set! That it gives me a very concrete way to test how good my model is on unseen data because I actually have a bit of data that I set aside that I can test it with.


You can then measure quantitatively how well it did using r-squared or some other metric, like root-mean-square error, for example. You can use that to test one model versus another and see what the best model is for a given problem. You can tune the parameters of that model and use train/test to maximize the accuracy of that model on your testing data. So this is a great way to prevent overfitting.

There are some caveats to supervised learning. need to make sure that both your training and test datasets are large enough to actually be representative of your data. You also need to make sure that you're catching all the different categories and outliers that you care about, in both training and testing, to get a good measure of its success, and to build a good model.

You have to make sure that you've selected from those datasets randomly, and that you're not just carving your dataset in two and saying everything left of here is training and right here is testing. You want to sample that randomly, because there could be some pattern sequentially in your data that you don't know about.

Now, if your model is overfitting, and just going out of its way to accept outliers in your training data, then that's going to be revealed when you put it against unset scene of testing data. This is because all that gyrations for outliers won't help with the outliers that it hasn't seen before.

Let's be clear here that train/test is not perfect, and it is possible to get misleading results from it. Maybe your sample sizes are too small, like we already talked about, or maybe just due to random chance your training data and your test data look remarkably similar, they actually do have a similar set of outliers - and you can still be overfitting. As you can see in the following example, it really can happen:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-05-01/steps/8/1.jpg)

### K-fold cross validation

Now there is a way around this problem, called k-fold cross-validation, and we'll look at an example of this later in the book, but the basic concept is you train/test many times. So you actually split your data not into just one training set and one test set, but into multiple randomly assigned segments, k segments. That's where the k comes from. And you reserve one of those segments as your test data, and then you start training your model on the remaining segments and measure their performance against your test dataset. Then you take the average performance from each of those training sets' models' results and take their r-squared average score.

So this way, you're actually training on different slices of your data, measuring them against the same test set, and if you have a model that's overfitting to a particular segment of your training data, then it will get averaged out by the other ones that are contributing to k-fold cross-validation.

Here are the K-fold cross validation steps:

- Split your data into K randomly-assigned segments
- Reserve one segment as your test data
- Train on each of the remaining K-1 segments and measure their performance against the test set
- Take the average of the K-1 r-squared scores

This will make more sense later in the book, right now I would just like for you to know that this tool exists for actually making train/test even more robust than it already is. So let's go and actually play with some data and actually evaluate it using train/test next.

### Using train/test to prevent overfitting of a polynomial regression

Let's put train/test into action. So you might remember that a regression can be thought of as a form of supervised machine learning. Let's just take a polynomial regression, which we covered earlier, and use train/test to try to find the right degree polynomial to fit a given set of data.

#### Open Notebook
The Notebook opens in a new browser window. You can create a new notebook or open a local one. Check out the local folder `work` for several notebooks. Open and run `TrainTest.ipynb` in the `work` folder.


You can also open the Jupyter Notebook at https://[[HOST_SUBDOMAIN]]-8888-[[KATACODA_HOST]].environments.katacoda.com/notebooks/work/TrainTest.ipynb


Just like in our previous example, we're going to set up a little fake dataset of randomly generated page speeds and purchase amounts, and I'm going to create a quirky little relationship between them that's exponential in nature.

```
%matplotlib inline 
import numpy as np 
from pylab import * 
 
np.random.seed(2) 
 
pageSpeeds = np.random.normal(3.0, 1.0, 100) 
purchaseAmount = np.random.normal(50.0, 30.0, 100) / pageSpeeds 
 
scatter(pageSpeeds, purchaseAmount) 
```

Let's go ahead and generate that data. We'll use a normal distribution of random data for both page speeds and purchase amount using the relationship as shown in the following screenshot:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-05-01/steps/10/1.jpg)

Next, we'll split that data. We'll take 80% of our data, and we're going to reserve that for our training data. So only 80% of these points are going to be used for training the model, and then we're going to reserve the other 20% for testing that model against unseen data.

We'll use Python's syntax here for splitting the list. The first 80 points are going to go to the training set, and the last 20, everything after 80, is going to go to test set. You may remember this from our Python basics chapter earlier on, where we covered the syntax to do this, and we'll do the same thing for purchase amounts here:

```
trainX = pageSpeeds[:80] 
testX = pageSpeeds[80:] 
 
trainY = purchaseAmount[:80] 
testY = purchaseAmount[80:] 
```

Now in our earlier sections, I've said that you shouldn't just slice your dataset in two like this, but that you should randomly sample it for training and testing. In this case though, it works out because my original data was randomly generated anyway, so there's really no rhyme or reason to where things fell. But in real-world data you'll want to shuffle that data before you split it.


We'll look now at a handy method that you can use for that purpose of shuffling your data. Also, if you're using the pandas package, there's some handy functions in there for making training and test datasets automatically for you. But we're going to do it using a Python list here. So let's visualize our training dataset that we ended up with. We'll do a scatter plot of our training page speeds and purchase amounts.

```
scatter(trainX, trainY) 
```

This is what your output should now look like:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-05-01/steps/11/1.jpg)

Basically, 80 points that were selected at random from the original complete dataset have been plotted. It has basically the same shape, so that's a good thing. It's representative of our data. That's important!

Now let's plot the remaining 20 points that we reserved as test data.

```
scatter(testX, testY) 
```

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-05-01/steps/11/2.jpg)

Here, we see our remaining 20 for testing also has the same general shape as our original data. So I think that's a representative test set too. It's a little bit smaller than you would like to see in the real world, for sure. You probably get a little bit of a better result if you had 1,000 points instead of 100, for example, to choose from and reserved 200 instead of 20.


Now we're going to try to fit an 8th degree polynomial to this data, and we'll just pick the number 8 at random because I know it's a really high order and is probably overfitting.

Let's go ahead and fit our 8th degree polynomial using np.poly1d(np.polyfit(x, y, 8)), where x is an array of the training data only, and y is an array of the training data only. We are finding our model using only those 80 points that we reserved for training. Now we have this p4 function that results that we can use to predict new values:

```
x = np.array(trainX) 
y = np.array(trainY) 
 
p4 = np.poly1d(np.polyfit(x, y, 8)) 
```

Now we'll plot the polynomial this came up with against the training data. We can scatter our original data for the training data set, and then we can plot our predicted values against them:

```
import matplotlib.pyplot as plt 
 
xp = np.linspace(0, 7, 100) 
axes = plt.axes() 
axes.set_xlim([0,7]) 
axes.set_ylim([0, 200]) 
plt.scatter(x, y) 
plt.plot(xp, p4(xp), c='r') 
plt.show() 
```

You can see in the following graph that it looks like a pretty good fit, but you know that clearly it's doing some overfitting:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-05-01/steps/11/3.jpg)


What's this craziness out at the right? I'm pretty sure our real data, if we had it out there, wouldn't be crazy high, as this function would implicate. So this is a great example of overfitting your data. It fits the data you gave it very well, but it would do a terrible job of predicting new values beyond the point where the graph is going crazy high on the right. So let's try to tease that out. Let's give it our test dataset:

```
testx = np.array(testX) 
testy = np.array(testY) 
 
axes = plt.axes() 
axes.set_xlim([0,7]) 
axes.set_ylim([0, 200]) 
plt.scatter(testx, testy) 
plt.plot(xp, p4(xp), c='r') 
plt.show() 
```

Indeed, if we plot our test data against that same function, well, it doesn't actually look that bad.

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-05-01/steps/11/4.jpg)

We got lucky and none of our test is actually out here to begin with, but you can see that it's a reasonable fit, but far from perfect. And in fact, if you actually measure the r-squared score, it's worse than you might think. We can measure that using the r2_score() function from sklearn.metrics. We just give it our original data and our predicted values and it just goes through and measures all the variances from the predictions and squares them all up for you:

```
from sklearn.metrics import r2_score  
r2 = r2_score(testy, p4(testx))  
print r2 
```

We end up with an r-squared score of just 0.3. So that's not that hot! You can see that it fits the training data a lot better:

```
from sklearn.metrics import r2_score  
r2 = r2_score(np.array(trainY), p4(np.array(trainX))) 
print r2 
```

The r-squared value turns out to be 0.6, which isn't too surprising, because we trained it on the training data. The test data is sort of its unknown, its test, and it did fail the test, quite frankly. 30%, that's an F!

So this has been an example where we've used train/test to evaluate a supervised learning algorithm, and like I said before, pandas has some means of making this even easier. We'll look at that a little bit later, and we'll also look at more examples of train/test, including k-fold cross validation, later in the book as well.

### Bayesian methods - Concepts

Did you ever wonder how the spam classifier in your e-mail works? How does it know that an e-mail might be spam or not? Well, one popular technique is something called Naive Bayes, and that's an example of a Bayesian method. Let's learn more about how that works. Let's discuss Bayesian methods.

We did talk about Bayes' theorem earlier in this book in the context of talking about how things like drug tests could be very misleading in their results. But you can actually apply the same Bayes' theorem to larger problems, like spam classifiers. So let's dive into how that might work, it's called a Bayesian method.

So just a refresher on Bayes' theorem -remember, the probability of A given B is equal to the overall probability of A times the probability of B given A over the overall probability of B:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-05-01/steps/15/1.jpg)

How can we use that in machine learning? I can actually build a spam classifier for that: an algorithm that can analyze a set of known spam e-mails and a known set of non-spam e-mails, and train a model to actually predict whether new e-mails are spam or not. This is a real technique used in actual spam classifiers in the real world.


As an example, let's just figure out the probability of an e-mail being spam given that it contains the word "free". If people are promising you free stuff, it's probably spam! So let's work that out. The probability of an email being spam given that you have the word "free" in that e-mail works out to the overall probability of it being a spam message times the probability of containing the word "free" given that it's spam over the probability overall of being free:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-05-01/steps/15/1.jpg)

The numerator can just be thought of as the probability of a message being Spam and containing the word Free. But that's a little bit different than what we're looking for, because that's the odds out of the complete dataset and not just the odds within things that contain the word Free. The denominator is just the overall probability of containing the word Free. Sometimes that won't be immediately accessible to you from the data that you have. If it's not, you can expand that out to the following expression if you need to derive it:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-05-01/steps/15/1.jpg)

This gives you the percentage of e-mails that contain the word "free" that are spam, which would be a useful thing to know when you're trying to figure out if it's spam or not.


What about all the other words in the English language, though? So our spam classifier should know about more than just the word "free". It should automatically pick up every word in the message, ideally, and figure out how much does that contribute to the likelihood of a particular e-mail being spam. So what we can do is train our model on every word that we encounter during training, throwing out things like "a" and "the" and "and" and meaningless words like that. Then when we go through all the words in a new e-mail, we can multiply the probability of being spam for each word together, and we get the overall probability of that e-mail being spam.

Now it's called Naive Bayes for a reason. It's naive is because we're assuming that there's no relationships between the words themselves. We're just looking at each word in isolation, individually within a message, and basically combining all the probabilities of each word's contribution to it being spam or not. We're not looking at the relationships between the words. So a better spam classifier would do that, but obviously that's a lot harder.

So this sounds like a lot of work. But the overall idea is not that hard, and scikit-learn in Python makes it actually pretty easy to do. It offers a feature called CountVectorizer that makes it very simple to actually split up an e-mail to all of its component words and process those words individually. Then it has a MultinomialNB function, where NB stands for Naive Bayes, which will do all the heavy lifting for Naive Bayes for us.

### Implementing a spam classifier with Naive Bayes

Let's write a spam classifier using Naive Bayes. You're going to be surprised how easy this is. In fact, most of the work ends up just being reading all the input data that we're going to train on and actually parsing that data in. The actual spam classification bit, the machine learning bit, is itself just a few lines of code. So that's usually how it works out: reading in and massaging and cleaning up your data is usually most of the work when you're doing data science, so get used to the idea!

#### Open Notebook
The Notebook opens in a new browser window. You can create a new notebook or open a local one. Check out the local folder `work` for several notebooks. Open and run `NaiveBayes.ipynb` in the `work` folder.

You can also open the Jupyter Notebook at https://[[HOST_SUBDOMAIN]]-8888-[[KATACODA_HOST]].environments.katacoda.com/notebooks/work/NaiveBayes.ipynb

So the first thing we need to do is read all those e-mails in somehow, and we're going to again use pandas to make this a little bit easier. Again, pandas is a useful tool for handling tabular data. We import all the different packages that we're going to use within our example here, that includes the os library, the io library, numpy, pandas, and CountVectorizer and MultinomialNB from scikit-learn.

Let's go through this code in detail now. We can skip past the function definitions of readFiles() and dataFrameFromDirectory()for now and go down to the first thing that our code actually does which is to create a pandas DataFrame object.

We're going to construct this from a dictionary that initially contains a little empty list for messages in an empty list of class. So this syntax is saying, "I want a DataFrame that has two columns: one that contains the message, the actual text of each e-mail; and one that contains the class of each e-mail, that is, whether it's spam or ham". So it's saying I want to create a little database of e-mails, and this database has two columns: the actual text of the e-mail and whether it's spam or not.

Now we needed to put something in that database, that is, into that DataFrame, in Python syntax. So we call the two methods append() and dataFrameFromDirectory() to actually throw into the DataFrame all the spam e-mails from my spam folder, and all the ham e-mails from the ham folder.

If you are playing along here, make sure you modify the path passed to the dataFrameFromDirectory() function to match wherever you installed the book materials in your system! And again, if you're on Mac or Linux, please pay attention to backslashes and forward slashes and all that stuff. In this case, it doesn't matter, but you won't have a drive letter, if you're not on Windows. So just make sure those paths are actually pointing to where your spam and ham folders are for this example.


Next, dataFrameFromDirectory() is a function I wrote, which basically says I have a path to a directory, and I know it's given classification, spam or ham, then it uses the readFiles() function, that I also wrote, which will iterate through every single file in a directory. So readFiles() is using the os.walk() function to find all the files in a directory. Then it builds up the full pathname for each individual file in that directory, and then it reads it in. And while it's reading it in, it actually skips the header for each e-mail and just goes straight to the text, and it does that by looking for the first blank line.

It knows that everything after the first empty line is actually the message body, and everything in front of that first empty line is just a bunch of header information that I don't actually want to train my spam classifier on. So it gives me back both, the full path to each file and the body of the message. So that's how we read in all of the data, and that's the majority of the code!

So what I have at the end of the day is a DataFrame object, basically a database with two columns, that contains message bodies, and whether it's spam or not. We can go ahead and run that, and we can use the head command from the DataFrame to actually preview what this looks like:

```
data.head() 
```

The first few entries in our DataFrame look like this: for each path to a given file full of e-mails we have a classification and we have the message body:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-05-01/steps/18/1.jpg)

Alright, now for the fun part, we're going to use the MultinomialNB() function from scikit-learn to actually perform Naive Bayes on the data that we have.

```
vectorizer = CountVectorizer() 
counts = vectorizer.fit_transform(data['message'].values) 
 
classifier = MultinomialNB() 
targets = data['class'].values 
classifier.fit(counts, targets) 
```

This is what your output should now look like:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-05-01/steps/18/2.jpg)

Once we build a MultinomialNB classifier, it needs two inputs. It needs the actual data that we're training on (counts), and the targets for each thing (targets). So counts is basically a list of all the words in each e-mail and the number of times that word occurs.

So this is what CountVectorizer() does: it takes the message column from the DataFrame and takes all the values from it. I'm going to call vectorizer.fit_transform which basically tokenizes or converts all the individual words seen in my data into numbers, into values. It then counts up how many times each word occurs.

This is a more compact way of representing how many times each word occurs in an e-mail. Instead of actually preserving the words themselves, I'm representing those words as different values in a sparse matrix, which is basically saying that I'm treating each word as a number, as a numerical index, into an array. What that does is, just in plain English, it split each message up into a list of words that are in it, and counts how many times each word occurs. So we're calling that counts. It's basically that information of how many times each word occurs in each individual message. Mean while targets is the actual classification data for each e-mail that I've encountered. So I can call classifier.fit() using my MultinomialNB() function to actually create a model using Naive Bayes, which will predict whether new e-mails are spam or not based on the information we've given it.

Let's go ahead and run that. It runs pretty quickly! I'm going to use a couple of examples here. Let's try a message body that just says Free Money now!!! which is pretty clearly spam, and a more innocent message that just says "Hi Bob, how about a game of golf tomorrow?" So we're going to pass these in.

```
examples = ['Free Money now!!!', "Hi Bob, how about a game of golf tomorrow?"] 
example_counts = vectorizer.transform(examples) 
predictions = classifier.predict(example_counts) 
predictions 
```

The first thing we do is convert the messages into the same format that I trained my model on. So I use that same vectorizer that I created when creating the model to convert each message into a list of words and their frequencies, where the words are represented by positions in an array. Then once I've done that transformation, I can actually use the predict() function on my classifier, on that array of examples that have transformed into lists of words, and see what we come up with:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-05-01/steps/18/3.jpg)

```
array(['spam', 'ham'], dtype='|S4') 
```

And sure enough, it works! So, given this array of two input messages, Free Money now!!! and Hi Bob, it's telling me that the first result came back as spam and the second result came back as ham, which is what I would expect. That's pretty cool. So there you have it.

### Activity

We had a pretty small dataset here, so you could try running some different e-mails through it if you want and see if you get different results. If you really want to challenge yourself, try applying train/test to this example. So the real measure of whether or not my spam classifier is good or not is not just intuitively whether it can figure out that Free Money now!!! is spam. You want to measure that quantitatively.

So if you want a little bit of a challenge, go ahead and try to split this data up into a training set and a test dataset. You can actually look up online how pandas can split data up into train sets and testing sets pretty easily for you, or you can do it by hand. Whatever works for you. See if you can actually apply your MultinomialNB classifier to a test dataset and measure its performance. So, if you want a little bit of an exercise, a little bit of a challenge, go ahead and give that a try.

How cool is that? We just wrote our own spam classifier just using a few lines of code in Python. It's pretty easy using scikit-learn and Python. That's Naive Bayes in action, and you can actually go and classify some spam or ham messages now that you have that under your belt. Pretty cool stuff. Let's talk about clustering next.

