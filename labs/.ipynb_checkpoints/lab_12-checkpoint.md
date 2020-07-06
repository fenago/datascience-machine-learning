### Introduction

In this scenario, we're going to talk about the challenges of dealing with real-world data, and some of the quirks you might run into. The chapter starts by talking about the bias-variance trade-off, which is kind of a more principled way of talking about the different ways you might overfit and underfit data, and how it all interrelates with each other. We then talk about the k-fold cross-validation technique, which is an important tool in your chest to combat overfitting, and look at how to implement it using Python.

Next, we analyze the importance of cleaning your data and normalizing it before actually applying any algorithms on it. We see an example to determine the most popular pages on a website which will demonstrate the importance of cleaning data. The chapter also covers the importance of remembering to normalize numerical data. Finally, we look at how to detect outliers and deal with them.

Specifically, this chapter covers the following topics:

- Analyzing the bias/variance trade-off
- The concept of k-fold cross-validation and its implementation
- The importance of cleaning and normalizing data
- An example to determine the popular pages of a website
- Normalizing numerical data
- Detecting outliers and dealing with them

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

 ### Bias/variance trade-off

 One of the basic challenges that we face when dealing with real-world data is overfitting versus underfitting your regressions to that data, or your models, or your predictions. When we talk about underfitting and overfitting, we can often talk about that in the context of bias and variance, and the bias-variance trade-off. So, let's talk about what that means.

So conceptually, bias and variance are pretty simple. Bias is just how far off you are from the correct values, that is, how good are your predictions overall in predicting the right overall value. If you take the mean of all your predictions, are they more or less on the right spot? Or are your errors all consistently skewed in one direction or another? If so, then your predictions are biased in a certain direction.

Variance is just a measure of how spread out, how scattered your predictions are. So, if your predictions are all over the place, then that's high variance. But, if they're very tightly focused on what the correct values are, or even an incorrect value in the case of high bias, then your variance is small.

Let's look at some examples. Let's imagine that the following dartboard represents a bunch of predictions we're making where the real value we're trying to predict is in the center of the bullseye:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-08/steps/3/1.png)

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-08/steps/3/1-1.png)


In reality, you often need to choose between bias and variance. It comes down to over fitting Vs underfitting your data. Let's take a look at the following example:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-08/steps/3/2.png)

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-08/steps/3/3.png)

It's a little bit of a different way of thinking of bias and variance. So, in the left graph, we have a straight line, and you can think of that as having very low variance, relative to these observations. So, there's not a lot of variance in this line, that is, there is low variance. But the bias, the error from each individual point, is actually high.

Now, contrast that to the overfitted data in the graph at the right, where we've kind of gone out of our way to fit the observations. The line has high variance, but low bias, because each individual point is pretty close to where it should be. So, this is an example of where we traded off variance for bias.

At the end of the day, you're not out to just reduce bias or just reduce variance, you want to reduce error. That's what really matters, and it turns out you can express error as a function of bias and variance:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-08/steps/3/4.png)

Looking at this, error is equal to bias squared plus variance. So, these things both contribute to the overall error, with bias actually contributing more. But keep in mind, it's error you really want to minimize, not the bias or the variance specifically, and that an overly complex model will probably end up having a high variance and low bias, whereas a too simple model will have low variance and high bias. However, they could both end up having similar error terms at the end of the day. You just have to find the right happy medium of these two things when you're trying to fit your data. We'll talk about some more principled ways of actually avoiding overfitting in our forthcoming sections. But, it's just the concept of bias and variance that I want to get across, because people do talk about it and you're going to be expected to know what means.


Now let's tie that back to some earlier concepts in this book. For example, in k-nearest neighbors if we increase the value of K, we start to spread out our neighborhood that were averaging across to a larger area. That has the effect of decreasing variance because we're kind of smoothing things out over a larger space, but it might increase our bias because we'll be picking up a larger population that may be less and less relevant to the point we started from. By smoothing out KNN over a larger number of neighbors, we can decrease the variance because we're smoothing things out over more values. But, we might be introducing bias because we're introducing more and more points that are less than less related to the point we started with.

Decision trees is another example. We know that a single decision tree is prone to overfitting, so that might imply that it has a high variance. But, random forests seek to trade off some of that variance for bias reduction, and it does that by having multiple trees that are randomly variant and averages all their solutions together. It's like when we average things out by increasing K in KNN: we can average out the results of a decision tree by using more than one decision tree using random forests similar idea.

This is bias-variance trade-off. You know the decision you have to make between how overall accurate your values are, and how spread out they are or how tightly clustered they are. That's the bias-variance trade-off and they both contribute to the overall error, which is the thing you really care about minimizing. So, keep those terms in mind!

### K-fold cross-validation to avoid overfitting

Earlier in the book, we talked about train and test as a good way of preventing overfitting and actually measuring how well your model can perform on data it's never seen before. We can take that to the next level with a technique called k-fold cross-validation. So, let's talk about this powerful tool in your arsenal for fighting overfitting; k-fold cross-validation and learn how that works.

To recall from train/test, the idea was that we split all of our data that we're building a machine learning model based off of into two segments: a training dataset, and a test dataset. The idea is that we train our model only using the data in our training dataset, and then we evaluate its performance using the data that we reserved for our test dataset. That prevents us from overfitting to the data that we have because we're testing the model against data that it's never seen before.

However, train/test still has its limitations: you could still end up overfitting to your specific train/test split. Maybe your training dataset isn't really representative of the entire dataset, and too much stuff ended up in your training dataset that skews things. So, that's where k-fold cross-validation comes in, it takes train/test and kicks it up a notch.

The idea, although it sounds complicated, is fairly simple:

- Instead of dividing our data into two buckets, one for training and one for testing, we divide it into K buckets.
- We reserve one of those buckets for testing purposes, for evaluating the results of our model.
- We train our model against the remaining buckets that we have, K-1, and then we take our test dataset and use that to evaluate how well our model did amongst all of those different training datasets.
- We average those resulting error metrics, that is, those r-squared values, together to get a final error metric from k-fold cross-validation.

That's all it is. It is a more robust way of doing train/test, and that's one way of doing it.

Now, you might think to yourself well, what if I'm overfitting to that one test dataset that I reserved? I'm still using the same test dataset for every one of those training datasets. What if that test dataset isn't really representative of things either?

There are variations of k-fold cross-validation that will randomize that as well. So, you could randomly pick what the training dataset is as well each time around, and just keep randomly assigning things to different buckets and measuring the results. But usually, when people talk about k-fold cross-validation, they're talking about this specific technique where you reserve one bucket for testing, and the remaining buckets for training, and you evaluate all of your training datasets against the test dataset when you build a model for each one.

### Example of k-fold cross-validation using scikit-learn

Fortunately, scikit-learn makes this really easy to do, and it's even easier than doing normal train/test! It's extremely simple to do k-fold cross-validation, so you may as well just do it.

Now, the way this all works in practice is you will have a model that you're trying to tune, and you will have different variations of that model, different parameters you might want to tweak on it, right?

Like, for example, the degree of polynomial for a polynomial fit. So, the idea is to try different values of your model, different variations, measure them all using k-fold cross-validation, and find the one that minimizes error against your test dataset. That's kind of your sweet spot there. In practice, you want to use k-fold cross-validation to measure the accuracy of your model against a test dataset, and just keep refining that model, keep trying different values within it, keep trying different variations of that model or maybe even different models entirely, until you find the technique that reduces error the most, using k-fold cross validation.

Let's go dive into an example and see how it works. We're going to apply this to our Iris dataset again, revisiting SVC, and we'll play with k-fold cross-validation and see how simple it is. Let's actually put k-fold cross-validation and train/test into practice here using some real Python code. You'll see it's actually very easy to use, which is a good thing because this is a technique you should be using to measure the accuracy, the effectiveness of your models in supervised learning.

Please go ahead and open up the `KFoldCrossValidation.ipynb` and follow along if you will. We're going to look at the Iris dataset again; remember we introduced this when we talk about dimensionality reduction?

#### Open Notebook
The Notebook opens in a new browser window. You can create a new notebook or open a local one. Check out the local folder `work` for several notebooks. Open and run `KFoldCrossValidation.ipynb` in the `work` folder.

You can also open the Jupyter Notebook at https://[[HOST_SUBDOMAIN]]-8888-[[KATACODA_HOST]].environments.katacoda.com/notebooks/work/KFoldCrossValidation.ipynb

### Example of k-fold cross-validation...


Just to refresh your memory, the Iris dataset contains a set of 150 Iris flower measurements, where each flower has a length and width of its petal, and a length and width of its sepal. We also know which one of 3 different species of Iris each flower belongs to. The challenge here is to create a model that can successfully predict the species of an Iris flower, just given the length and width of its petal and sepal. So, let's go ahead and do that.

We're going to use the SVC model. If you remember back again, that's just a way of classifying data that's pretty robust. There's a section on that if you need to go and refresh your memory:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-08/steps/6/1.png)

What we do is use the cross_validation library from scikit-learn, and we start by just doing a conventional train test split, just a single train/test split, and see how that will work.

To do that we have a train_test_split() function that makes it pretty easy. So, the way this works is we feed into train_test_split() a set of feature data. iris.data just contains all the actual measurements of each flower. iris.target is basically the thing we're trying to predict.

In this case, it contains all the species for each flower. test_size says what percentage do we want to train versus test. So, 0.4 means we're going to extract 40% of that data randomly for testing purposes, and use 60% for training purposes. What this gives us back is 4 datasets, basically, a training dataset and a test dataset for both the feature data and the target data. So, X_train ends up containing 60% of our Iris measurements, and X_test contains 40% of the measurements used for testing the results of our model. y_train and y_test contain the actual species for each one of those segments.

Then after that we go ahead and build an SVC model for predicting Iris species given their measurements, and we build that only using the training data. We fit this SVC model, using a linear kernel, using only the training feature data, and the training species data, that is, target data. We call that model clf. Then, we call the score() function on clf to just measure its performance against our test dataset. So, we score this model against the test data we reserved for the Iris measurements, and the test Iris species, and see how well it does:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-08/steps/6/2.png)

It turns out it does really well! Over 96% of the time, our model is able to correctly predict the species of an Iris that it had never seen before, just based on the measurements of that Iris. So that's pretty cool!

But, this is a fairly small dataset, about 150 flowers if I remember right. So, we're only using 60% of 150 flowers for training and only 40% of 150 flowers for testing. These are still fairly small numbers, so we could still be overfitting to our specific train/test split that we made. So, let's use k-fold cross-validation to protect against that. It turns out that using k-fold cross-validation, even though it's a more robust technique, is actually even easier to use than train/test. So, that's pretty cool! So, let's see how that works:

```
# We give cross_val_score a model, the entire data set and its "real" values, and the number of folds: 
scores = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=5) 
 
# Print the accuracy for each fold: 
print scores 
 
# And the mean accuracy of all 5 folds: 
print scores.mean() 
```

We have a model already, the SVC model that we defined for this prediction, and all you need to do is call cross_val_score() on the cross_validation package. So, you pass in this function a model of a given type (clf), the entire dataset that you have of all of the measurements, that is, all of my feature data (iris.data) and all of my target data (all of the species), iris.target.

I want cv=5 which means it's actually going to use 5 different training datasets while reserving 1 for testing. Basically, it's going to run it 5 times, and that's all we need to do. That will automatically evaluate our model against the entire dataset, split up five different ways, and give us back the individual results.

If we print back the output of that, it gives us back a list of the actual error metric from each one of those iterations, that is, each one of those folds. We can average those together to get an overall error metric based on k-fold cross-validation:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-08/steps/6/3.png)

When we do this over 5 folds, we can see that our results are even better than we thought! 98% accuracy. So that's pretty cool! In fact, in a couple of the runs we had perfect accuracy. So that's pretty amazing stuff.

Now let's see if we can do even better. We used a linear kernel before, what if we used a polynomial kernel and got even fancier? Will that be overfitting or will it actually better fit the data that we have? That kind of depends on whether there's actually a linear relationship or polynomial relationship between these petal measurements and the actual species or not. So, let's try that out:

```
clf = svm.SVC(kernel='poly', C=1).fit(X_train, y_train)
scores = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=5)
print scores
print scores.mean()
```

We'll just run this all again, using the same technique. But this time, we're using a polynomial kernel. We'll fit that to our training dataset, and it doesn't really matter where you fit to in this case, because cross_val_score() will just keep re-running it for you:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-08/steps/6/4.png)

It turns out that when we use a polynomial fit, we end up with an overall score that's even lower than our original run. So, this tells us that the polynomial kernel is probably overfitting. When we use k-fold cross-validation it reveals an actual lower score than with our linear kernel.

The important point here is that if we had just used a single train/test split, we wouldn't have realized that we were overfitting. We would have actually gotten the same result if we just did a single train/test split here as we did on the linear kernel. So, we might inadvertently be overfitting our data there, and not have even known it had we not use k-fold cross-validation. So, this is a good example of where k-fold comes to the rescue, and warns you of overfitting, where a single train/test split might not have caught that. So, keep that in your tool chest.

If you want to play around with this some more, go ahead and try different degrees. So, you can actually specify a different number of degrees. The default is 3 degrees for the polynomial kernel, but you can try a different one, you can try two.

Does that do better? If you go down to one, that degrades basically to a linear kernel, right? So, maybe there is still a polynomial relationship and maybe it's only a second degree polynomial. Try it out and see what you get back. That's k-fold cross-validation. As you can see, it's very easy to use thanks to scikit-learn. It's an important way to measure how good your model is in a very robust manner.

### Data cleaning and normalisation

Now, this is one of the simplest, but yet it might be the most important section in this whole book. We're going to talk about cleaning your input data, which you're going to spend a lot of your time doing.

How well you clean your input data and understand your raw input data is going to have a huge impact on the quality of your results - maybe even more so than what model you choose or how well you tune your models. So, pay attention; this is important stuff!

**Note:**

Cleaning your raw input data is often the most important, and time-consuming, part of your job as a data scientist!

Let's talk about an inconvenient truth of data science, and that's that you spend most of your time actually just cleaning and preparing your data, and actually relatively little of it analyzing it and trying out new algorithms. It's not quite as glamorous as people might make it out to be all the time. But, this is an extremely important thing to pay attention to.


There are a lot of different things that you might find in raw data. Data that comes in to you, just raw data, is going to be very dirty, it's going to be polluted in many different ways. If you don't deal with it it's going to skew your results, and it will ultimately end up in your business making the wrong decisions.

If it comes back that you made a mistake where you ingested a bunch of bad data and didn't account for it, didn't clean that data up, and what you told your business was to do something based on those results that later turn out to be completely wrong, you're going to be in a lot of trouble! So, pay attention!

There are a lot of different kinds of problems and data that you need to watch out for:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-08/steps/10/1.png)

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-08/steps/10/2.png)


There are a lot of different things that you might find in raw data. Data that comes in to you, just raw data, is going to be very dirty, it's going to be polluted in many different ways. If you don't deal with it it's going to skew your results, and it will ultimately end up in your business making the wrong decisions.

If it comes back that you made a mistake where you ingested a bunch of bad data and didn't account for it, didn't clean that data up, and what you told your business was to do something based on those results that later turn out to be completely wrong, you're going to be in a lot of trouble! So, pay attention!

There are a lot of different kinds of problems and data that you need to watch out for:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-08/steps/10/1.png)

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-08/steps/10/2.png)

### Cleaning web log data

We're going to show the importance of cleaning your data. I have some web log data from a little website that I own. We are just going to try to find the top viewed pages on that website. Sounds pretty simple, but as you'll see, it's actually quite challenging! So, if you want to follow along, the `TopPages.ipynb` is the notebook that we're working from here. Let's start!


#### Open Notebook
The Notebook opens in a new browser window. You can create a new notebook or open a local one. Check out the local folder `work` for several notebooks. Open and run `TopPages.ipynb` in the `work` folder.

You can also open the Jupyter Notebook at https://[[HOST_SUBDOMAIN]]-8888-[[KATACODA_HOST]].environments.katacoda.com/notebooks/work/TopPages.ipynb

I actually have an access log that I took from my actual website. It's a real HTTP access log from Apache and is included in your book materials. So, if you do want to play along here, make sure you update the path to move the access log to wherever you saved the book materials:

```
logPath = "E:\\sundog-consult\\Packt\\DataScience\\access_log.txt"
```

### Applying a regular expression on the web log

So, I went and got the following little snippet of code off of the Internet that will parse an Apache access log line into a bunch of fields:

```
format_pat= re.compile( 
    r"(?P<host>[\d\.]+)\s" 
    r"(?P<identity>\S*)\s" 
    r"(?P<user>\S*)\s" 
    r"\[(?P<time>.*?)\]\s" 
    r'"(?P<request>.*?)"\s' 
    r"(?P<status>\d+)\s" 
    r"(?P<bytes>\S*)\s" 
    r'"(?P<referer>.*?)"\s' 
    r'"(?P<user_agent>.*?)"\s*' 
) 
```

This code contains things like the host, the user, the time, the actual page request, the status, the referrer, user_agent (meaning which browser actually was used to view this page). It builds up what's called a regular expression, and we're using the re library to use it. That's basically a very powerful language for doing pattern matching on a large string. So, we can actually apply this regular expression to each line of our access log, and automatically group the bits of information in that access log line into these different fields. Let's go ahead and run this.

The obvious thing to do here, let's just whip up a little script that counts up each URL that we encounter that was requested, and keeps count of how many times it was requested. Then we can sort that list and get our top pages, right? Sounds simple enough!

So, we're going to construct a little Python dictionary called URLCounts. We're going to open up our log file, and for each line, we're going to apply our regular expression. If it actually comes back with a successful match for the pattern that we're trying to match, we'll say, Okay this looks like a decent line in our access log.

Let's extract the request field out of it, which is the actual HTTP request, the page which is actually being requested by the browser. We're going to split that up into its three components: it consists of an action, like get or post; the actual URL being requested; and the protocol being used. Given that information split out, we can then just see if that URL already exists in my dictionary. If so, I will increment the count of how many times that URL has been encountered by 1; otherwise, I'll introduce a new dictionary entry for that URL and initialize it to the value of 1. I do that for every line in the log, sort the results in reverse order, numerically, and print them out:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-08/steps/13/1.png)

So, let's go ahead and run that:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-08/steps/13/2.png)

Oops! We end up with this big old error here. It's telling us that, we need more than 1 value to unpack. So apparently, we're getting some requests fields that don't contain an action, a URL, and a protocol that they contain something else.

Let's see what's going on there! So, if we print out all the requests that don't contain three items, we'll see what's actually showing up. So, what we're going to do here is a similar little snippet of code, but we're going to actually do that split on the request field, and print out cases where we don't get the expected three fields.

```
URLCounts = {}

with open(logPath, "r") as f:
    for line in (l.rstrip() for l in f):
        match= format_pat.match(line)
        if match:
            access = match.groupdict()
            request = access['request']
            fields = request.split()
            if (len(fields) != 3):
                print fields
```

Let's see what's actually in there:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-08/steps/13/3.png)

So, we have a bunch of empty fields. That's our first problem. But, then we have the first field that's full just garbage. Who knows where that came from, but it's clearly erroneous data. Okay, fine, let's modify our script.


### Modification one - filtering the request field

We'll actually just throw out any lines that don't have the expected 3 fields in the request. That seems like a legitimate thing to do, because this does in fact have completely useless data inside of it, it's not like we're missing out on anything here by doing that. So, we'll modify our script to do that. We've introduced an if (len(fields) == 3) line before it actually tries to process it. We'll run that:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-08/steps/13/4.png)

Hey, we got a result!

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-08/steps/13/5.png)

But this doesn't really look like the top pages on my website. Remember, this is a news site. So, we're getting a bunch of PHP file hits, that's Perl scripts. What's going on there? Our top result is this xmlrpc.php script, and then WP_login.php, followed by the homepage. So, not very useful. Then there is robots.txt, then a bunch of XML files.

You know when I looked into this later on, it turned out that my site was actually under a malicious attack; someone was trying to break into it. This xmlrpc.php script was the way they were trying to guess at my passwords, and they were trying to log in using the login script. Fortunately, I shut them down before they could actually get through to this website.

This was an example of malicious data being introduced into my data stream that I have to filter out. So, by looking at that, we can see that not only was that malicious attack looking at PHP files, but it was also trying to execute stuff. It wasn't just doing a get request, it was doing a post request on the script to actually try to execute code on my website.

### Modification two - filtering post requests

Now, I know that the data that I care about, you know in the spirit of the thing I'm trying to figure out is, people getting web pages from my website. So, a legitimate thing for me to do is to filter out anything that's not a get request, out of these logs. So, let's do that next. We're going to check again if we have three fields in our request field, and then we're also going to check if the action is get. If it's not, we're just going to ignore that line entirely:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-08/steps/13/6.png)

We should be getting closer to what we want now, the following is the output of the preceding code:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-08/steps/13/7.png)


Yeah, this is starting to look more reasonable. But, it still doesn't really pass a sanity check. This is a news website; people go to it to read news. Are they really reading my little blog on it that just has a couple of articles? I don't think so! That seems a little bit fishy. So, let's dive in a little bit, and see who's actually looking at those blog pages. If you were to actually go into that file and examine it by hand, you would see that a lot of these blog requests don't actually have any user agent on them. They just have a user agent of -, which is highly unusual:


If a real human being with a real browser was trying to get this page, it would say something like Mozilla, or Internet Explorer, or Chrome or something like that. So, it seems that these requests are coming from some sort of a scraper. Again, potentially malicious traffic that's not identifying who it is.

### Modification three - checking the user agents

Maybe, we should be looking at the UserAgents too, to see if these are actual humans making requests, or not. Let's go ahead and print out all the different UserAgents that we're encountering. So, in the same spirit of the code that actually summed up the different URLs we were seeing, we can look at all the different UserAgents that we were seeing, and sort them by the most popular user_agent strings in this log:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-08/steps/13/8.png)

We get the following result:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-08/steps/13/9.png)

You can see most of it looks legitimate. So, if it's a scraper, and in this case it actually was a malicious attack but they were actually pretending to be a legitimate browser. But this dash user_agent shows up a lot too. So, I don't know what that is, but I know that it isn't an actual browser.

The other thing I'm seeing is a bunch of traffic from spiders, from web crawlers. So, there is Baidu which is a search engine in China, there is Googlebot just crawling the page. I think I saw Yandex in there too, a Russian search engine. So, our data is being polluted by a lot of crawlers that are just trying to mine our website for search engine purposes. Again, that traffic shouldn't count toward the intended purpose of my analysis, of seeing what pages these actual human beings are looking at on my website. These are all automated scripts.

### Filtering the activity of spiders/robots

Alright, so this gets a little bit tricky. There's no real good way of identifying spiders or robots just based on the user string alone. But we can at least take a legitimate crack at it, and filter out anything that has the word "bot" in it, or anything from my caching plugin that might be requesting pages in advance as well. We'll also strip out our friend single dash. So, we will once again refine our script to, in addition to everything else, strip out any UserAgents that look fishy:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-08/steps/13/10.png)

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-08/steps/13/11.png)

What do we get?

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-08/steps/13/12.png)


Alright, so here we go! This is starting to look more reasonable for the first two entries, the homepage is most popular, which would be expected. Orlando headlines is also popular, because I use this website more than anybody else, and I live in Orlando. But after that, we get a bunch of stuff that aren't webpages at all: a bunch of scripts, a bunch of CSS files. Those aren't web pages.

### Modification four - applying website-specific filters

I can just apply some knowledge about my site, where I happen to know that all the legitimate pages on my site just end with a slash in their URL. So, let's go ahead and modify this again, to strip out anything that doesn't end with a slash:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-08/steps/13/13.png)

Let's run that!

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-08/steps/13/14.png)

Finally, we're getting some results that seem to make sense! So, it looks like, that the top page requested from actual human beings on my little No-Hate News site is the homepage, followed by orlando-headlines, followed by world news, followed by the comics, then the weather, and the about screen. So, this is starting to look more legitimate.

If you were to dig even deeper though, you'd see that there are still problems with this analysis. For example, those feed pages are still coming from robots just trying to get RSS data from my website. So, this is a great parable in how a seemingly simple analysis requires a huge amount of pre-processing and cleaning of the source data before you get results that make any sense.

Again, make sure the things you're doing to clean your data along the way are principled, and you're not just cherry-picking problems that don't match with your preconceived notions. So, always question your results, always look at your source data, and look for weird things that are in it.

### Activity for web log data

Alright, if you want to mess with this some more you can solve that feed problem. Go ahead and strip out things that include feed because we know that's not a real web page, just to get some familiarity with the code. Or, go look at the log a little bit more closely, gain some understanding as to where those feed pages are actually coming from.

Maybe there's an even better and more robust way of identifying that traffic as a larger class. So, feel free to mess around with that. But I hope you learned your lesson: data cleaning - hugely important and it's going to take a lot of your time!

So, it's pretty surprising how hard it was to get some reasonable results on a simple question like "What are the top viewed pages on my website?" You can imagine if that much work had to go into cleaning the data for such a simple problem, think about all the nuanced ways that dirty data might actually impact the results of more complex problems, and complex algorithms.

It's very important to understand your source data, look at it, look at a representative sample of it, make sure you understand what's coming into your system. Always question your results and tie it back to the original source data to see where questionable results are coming from.

### Normalizing numerical data

This is a very quick section: I just want to remind you about the importance of normalizing your data, making sure that your various input feature data is on the same scale, and is comparable. And, sometimes it matters, and sometimes it doesn't. But, you just have to be cognizant of when it does. Just keep that in the back of your head because sometimes it will affect the quality of your results if you don't.

So, sometimes models will be based on several different numerical attributes. If you remember multivariant models, we might have different attributes of a car that we're looking at, and they might not be directly comparable measurements. Or, for example, if we're looking at relationships between ages and incomes, ages might range from 0 to 100, but incomes in dollars might range from 0 to billions, and depending on the currency it could be an even larger range! Some models are okay with that.

If you're doing a regression, usually that's not a big deal. But, other models don't perform so well unless those values are scaled down first to a common scale. If you're not careful, you can end up with some attributes counting more than others. Maybe the income would end up counting much more than the age, if you were trying to treat those two values as comparable values in your model.


So this can introduce also a bias in the attributes, which can also be a problem. Maybe one set of your data is skewed, you know, sometimes you need to normalize things versus the actual range seen for that set of values and not just to a 0 to whatever the maximum is scale. There's no set rule as to when you should and shouldn't do this sort of normalization. All I can say is always read the documentation for whatever technique you're using.

So, for example, in scikit-learn their PCA implementation has a whiten option that will automatically normalize your data for you. You should probably use that. It also has some preprocessing modules available that will normalize and scale things for you automatically as well.

Be aware too of textual data that should actually be represented numerically, or ordinally. If you have yes or no data you might need to convert that to 1 or 0 and do that in a consistent matter. So again, just read the documentation. Most techniques do work fine with raw, un-normalized data, but before you start using a new technique for the first time, just read the documentation and understand whether or not the inputs should be scaled or normalized or whitened first. If so, scikit-learn will probably make it very easy for you to do so, you just have to remember to do it! Don't forget to rescale your results when you're done if you are scaling the input data.

If you want to be able to interpret the results you get, sometimes you need to scale them back up to their original range after you're done. If you are scaling things and maybe even biasing them towards a certain amount before you input them into a model, make sure that you unscale them and unbias them before you actually present those results to somebody. Or else they won't make any sense! And just a little reminder, a little bit of a parable if you will, always check to see if you should normalize or whiten your data before you pass it into a given model.

No exercise associated with this section; it's just something I want you to remember. I'm just trying to drive the point home. Some algorithms require whitening, or normalization, some don't. So, always read the documentation! If you do need to normalize the data going into an algorithm it will usually tell you so, and it will make it very easy to do so. Please just be aware of that!

### Detecting outliers

A common problem with real-world data is outliers. You'll always have some strange users, or some strange agents that are polluting your data, that act abnormally and atypically from the typical user. They might be legitimate outliers; they might be caused by real people and not by some sort of malicious traffic, or fake data. So sometimes, it's appropriate to remove them, sometimes it isn't. Make sure you make that decision responsibly. So, let's dive into some examples of dealing with outliers.

For example, if I'm doing collaborative filtering, and I'm trying to make movie recommendations or something like that, you might have a few power users that have watched every movie ever made, and rated every movie ever made. They could end up having an inordinate influence on the recommendations for everybody else.

You don't really want a handful of people to have that much power in your system. So, that might be an example where it would be a legitimate thing to filter out an outlier, and identify them by how many ratings they've actually put into the system. Or, maybe an outlier would be someone who doesn't have enough ratings.

We might be looking at web log data, like we saw in our example earlier when we were doing data cleaning, outliers could be telling you that there's something very wrong with your data to begin with. It could be malicious traffic, it could be bots, or other agents that should be discarded that don't represent actual human beings that you're trying to model.


If someone really wanted the mean average income in the United States (and not the median), you shouldn't just throw out Donald Trump because you don't like him. You know the fact is, his billions of dollars are going to push that mean amount up, even if it doesn't budge the median. So, don't fudge your numbers by throwing out outliers. But throw out outliers if it's not consistent with what you're trying to model in the first place.

Now, how do we identify outliers? Well, remember our old friend standard deviation? We covered that very early in this book. It's a very useful tool for detecting outliers. You can, in a very principled matter, compute the standard deviation of a dataset that should have a more or less normal distribution. If you see a data point that's outside of one or two standard deviations, there you have an outlier.

Remember, we talked earlier too about the box and whisker diagrams too, and those also have a built-in way of detecting and visualizing outliers. Those diagrams define outliers as lying outside 1.5 the interquartile range.

What multiple do you choose? Well, you kind of have to use common sense, you know, there's no hard and fast rule as to what is an outlier. You have to look at your data and kind of eyeball it, look at the distribution, look at the histogram. See if there's actual things that stick out to you as obvious outliers, and understand what they are before you just throw them away.

### Dealing with outliers

So, let's take some example code, and see how you might handle outliers in practice. Let's mess around with some outliers. It's a pretty simple section. A little bit of review actually. If you want to follow along, we're in `Outliers.ipynb`. So, go ahead and open that up if you'd like:

#### Open Notebook
The Notebook opens in a new browser window. You can create a new notebook or open a local one. Check out the local folder `work` for several notebooks. Open and run `Outliers.ipynb` in the `work` folder.

You can also open the Jupyter Notebook at https://[[HOST_SUBDOMAIN]]-8888-[[KATACODA_HOST]].environments.katacoda.com/notebooks/work/Outliers.ipynb

```
import numpy as np

incomes = np.random.normal(27000, 15000, 10000)
incomes = np.append(incomes, [1000000000])

import matplotlib.pyplot as plt
plt.hist(incomes, 50)
plt.show()
```

We did something very similar, very early in the book, where we created a fake histogram of income distribution in the United States. What we're going to do is start off with a normal distribution of incomes here that are have a mean of $27,000 per year, with a standard deviation of 15,000. I'm going to create 10,000 fake Americans that have an income in that distribution. This is totally made-up data, by the way, although it's not that far off from reality.

Then, I'm going to stick in an outlier - call it Donald Trump, who has a billion dollars. We're going to stick this guy in at the end of our dataset. So, we have a normally distributed dataset around $27,000, and then we're going to stick in Donald Trump at the end.

We'll go ahead and plot that as a histogram:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-08/steps/22/1.png)

Wow! That's not very helpful! We have the entire normal distribution of everyone else in the country squeezed into one bucket of the histogram. On the other hand, we have Donald Trump out at the right side screwing up the whole thing at a billion dollars.


The other problem too is that if I'm trying to answer the question how much money does the typical American make. If I take the mean to try and figure that out, it's not going to be a very good, useful number:

```
incomes.mean ()
```

The output of the preceding code is as follows:

```
126892.66469341301
```

Donald Trump has pushed that number up all by himself to $126,000 and some odd of change, when I know that the real mean of my normally distributed data that excludes Donald Trump is only $27,000. So, the right thing to do there would be to use the median instead of the mean.

But, let's say we had to use the mean for some reason, and the right way to deal with this would be to exclude these outliers like Donald Trump. So, we need to figure out how do we identify these people. Well, you could just pick some arbitrary cutoff, and say, "I'm going to throw out all the billionaires", but that's not a very principled thing to do. Where did 1 billion come from?

It's just some accident of how we count numbers. So, a better thing to do would be to actually measure the standard deviation of your dataset, and identify outliers as being some multiple of a standard deviation away from the mean.

So, following is a little function that I wrote that does just that. It's called reject_outliers():

```
def reject_outliers(data): 
    u = np.median(data) 
    s = np.std(data) 
    filtered = [e for e in data if (u - 2 * s < e < u + 2 * s)] 
    return filtered 
 
filtered = reject_outliers(incomes) 
 
plt.hist(filtered, 50) 
plt.show() 
```

It takes in a list of data and finds the median. It also finds the standard deviation of that dataset. So, I filter that out, so I only preserve data points that are within two standard deviations of the median for my data. So, I can use this handy dandy reject_outliers() function on my income data, to actually strip out weird outliers automatically:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-08/steps/22/2.png)

Sure enough, it works! I get a much prettier graph now that excludes Donald Trump and focuses in on the more typical dataset here in the center. So, pretty cool stuff!

So, that's one example of identifying outliers, and automatically removing them, or dealing with them however you see fit. Remember, always do this in a principled manner. Don't just throw out outliers because they're inconvenient. Understand where they're coming from, and how they actually affect the thing you're trying to measure in spirit.

By the way, our mean is also much more meaningful now; much closer to 27,000 that it should be, now that we've gotten rid of that outlier.

### Activity for outliers

So, if you want to play around with this, you know just fiddle around with it like I normally ask you to do. Try different multiples of the standard deviation, try adding in more outliers, try adding in outliers that aren't quite as outlier-ish as Donald Trump. You know, just fabricate some extra fake data there and play around with it, see if you can identify those people successfully.

So there you have it! Outliers; pretty simple concept. So, that's an example of identifying outliers by looking at standard deviations, and just looking at the number of standard deviations from the mean or median that you care about. Median is probably a better choice actually, given that the outliers might be skewing the mean in and of themselves, right? So, by using the standard deviation, that's a good way of identifying outliers in a more principled manner than just picking some arbitrary cutoff. Again, you need to decide what the right thing to do is with those outliers. What are you actually trying to measure? Is it appropriate to actually discard them or not? So, keep that in your head!

